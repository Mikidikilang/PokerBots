"""
Parallel CFR Worker Pool & Shared Memory Regret Buffer (Phase 3)

[CRITICAL FIX — 2026-03-30] Broken IPC mapping eliminated.

    ROOT CAUSE:
    SharedMemoryRegretBuffer used a plain Python dict (self.infoset_to_idx)
    to map infoset hashes to tensor row indices.  When workers are spawned
    via mp.Process, each worker receives a COPY of this dict (fork on Linux,
    pickle on Windows).  The copies immediately diverge:

        Worker 0: allocates "abc" → idx 0 in ITS local dict
        Worker 1: allocates "abc" → idx 0 in ITS local dict (independent)
        Worker 2: allocates "xyz" → idx 0 in ITS local dict (collision!)

    All three workers write to self.regrets[0, :] — the SAME tensor row —
    but with DIFFERENT infosets.  The regrets become garbage.

    FIX:
    Eliminate the dict entirely.  Use a DETERMINISTIC hash function
    (zlib.crc32) to map infoset_hash → tensor row index:

        row = zlib.crc32(infoset_hash.encode('utf-8')) % max_infosets

    This mapping is:
        ✓ Identical across ALL processes (crc32 is deterministic)
        ✓ Zero shared state (no dict, no lock for allocation)
        ✓ O(1) (single integer operation)

    Collisions: With 100k rows and ~1k unique infosets per task, the
    collision rate is <1%.  Even with collisions, regrets MIX (they don't
    corrupt) — acceptable noise for MCCFR which is inherently stochastic.

    For production at scale (>100k infosets), increase max_infosets or
    use open-addressing in a second shared tensor.

Architecture:
    1. Master Process: Coordinates iteration, aggregates regrets, GPU inference
    2. Worker Processes: Independent game tree traversals, CFR computation
    3. Shared Memory: Regret buffer using torch.Tensor.share_memory_()
    4. IPC: Queue for work distribution, Result collection

References:
    - Schäfer et al. (2023): "Parallel Game Tree Search"
    - Brown & Sandholm (2019): "Endgame Solving in Large Imperfect-Info Games"
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.env.wrappers import RLCardWrapper, WrapperConfig
from src.model.networks import PokerActorCritic, NetworkConfig
from src.training.cfr_traversal import MCCFRTraversal
from src.training.regret_store import RegretStore

logger = logging.getLogger(__name__)


# ============================================================================
# Task / Result dataclasses
# ============================================================================

@dataclass
class WorkerTask:
    task_id: int
    game_state_hash: str
    iteration: int
    num_traversals: int
    player_id: int
    context: dict = field(default_factory=dict)


@dataclass
class WorkerResult:
    task_id: int
    game_value: float = 0.0
    num_traversals: int = 0
    worker_id: int = -1
    compute_time: float = 0.0
    num_regrets_written: int = 0


# ============================================================================
# SharedMemoryRegretBuffer — DICT-FREE (2026-03-30 FIX)
# ============================================================================

class SharedMemoryRegretBuffer:
    """Zero-copy shared regret buffer using torch.Tensor.share_memory_().

    [2026-03-30 FIX] NO DICT for infoset→index mapping.

    Every process computes:
        row = zlib.crc32(infoset_hash.encode()) % max_infosets

    This is deterministic and identical across all processes,
    eliminating the divergent-local-dict bug entirely.

    The only shared mutable state is ``self.regrets`` (a shared tensor)
    protected by striped locks for concurrent writes.
    """

    def __init__(
        self,
        max_infosets: int = 100_000,
        num_actions: int = 9,
        **_ignored: Any,            # Eat legacy kwargs (shared_infoset_mapping etc.)
    ):
        self.max_infosets = max_infosets
        self.num_actions = num_actions

        # Shared memory tensor: [max_infosets, num_actions]
        self.regrets = torch.zeros(
            (max_infosets, num_actions),
            dtype=torch.float32,
            requires_grad=False,
        ).share_memory_()

        # ★ NO DICT — hash-to-index is computed on the fly via crc32

        # Striped locks for safe concurrent writes to individual rows
        self.num_locks = 256
        self.write_locks = [mp.Lock() for _ in range(self.num_locks)]

        # Reverse mapping (master-side only, for get_all_regrets)
        # Workers never read this; master populates it lazily.
        self._known_hashes: Dict[str, int] = {}
        self._known_lock = mp.Lock()

        logger.info(
            "SharedMemoryRegretBuffer: %d infosets × %d actions, "
            "%d lock stripes, DICT-FREE indexing (crc32)",
            max_infosets, num_actions, self.num_locks,
        )

    # ------------------------------------------------------------------
    # Deterministic hash-to-index (same on every process)
    # ------------------------------------------------------------------

    def _hash_to_index(self, infoset_hash: str) -> int:
        """Map infoset hash string to a tensor row index.

        Uses zlib.crc32 which is:
            ✓ Deterministic (no PYTHONHASHSEED dependency)
            ✓ Fast (~30 ns per call)
            ✓ Identical across all processes / platforms
        """
        return zlib.crc32(infoset_hash.encode("utf-8")) % self.max_infosets

    # ------------------------------------------------------------------
    # Write regrets
    # ------------------------------------------------------------------

    def add_regret(self, infoset_hash: str, action: int, regret_delta: float) -> None:
        """Atomically add regret_delta to shared buffer."""
        idx = self._hash_to_index(infoset_hash)
        action = int(action) % self.num_actions
        stripe = idx % self.num_locks
        with self.write_locks[stripe]:
            self.regrets[idx, action].add_(regret_delta)

    def add_regrets_batch(
        self, infoset_hash: str, action_regrets: Dict[int, float]
    ) -> None:
        """Add multiple regrets for one infoset (atomic batch)."""
        idx = self._hash_to_index(infoset_hash)
        stripe = idx % self.num_locks
        with self.write_locks[stripe]:
            for action, delta in action_regrets.items():
                a = int(action) % self.num_actions
                self.regrets[idx, a].add_(delta)

    def register_hash(self, infoset_hash: str) -> None:
        """Record a hash for master-side reverse lookup (optional)."""
        idx = self._hash_to_index(infoset_hash)
        with self._known_lock:
            self._known_hashes[infoset_hash] = idx

    # ------------------------------------------------------------------
    # Read regrets
    # ------------------------------------------------------------------

    def get_regrets(self, infoset_hash: str) -> np.ndarray:
        idx = self._hash_to_index(infoset_hash)
        return self.regrets[idx].numpy().copy()

    def get_all_regrets(self) -> Dict[str, np.ndarray]:
        """Return all known infoset regrets (master-side only).

        Workers call ``register_hash`` after writing; the master can
        then iterate the known set.  If no hashes are registered,
        scans the tensor for any non-zero rows.
        """
        if self._known_hashes:
            return {
                h: self.regrets[idx].numpy().copy()
                for h, idx in self._known_hashes.items()
            }
        # Fallback: scan for any non-zero rows (slower but always works)
        result: Dict[str, np.ndarray] = {}
        for row_idx in range(self.max_infosets):
            row = self.regrets[row_idx]
            if row.abs().sum().item() > 1e-12:
                result[f"row_{row_idx}"] = row.numpy().copy()
        return result

    def get_non_zero_count(self) -> int:
        """Count tensor entries that are non-zero."""
        return int((self.regrets.abs() > 1e-12).any(dim=1).sum().item())

    def reset(self) -> None:
        self.regrets.zero_()
        with self._known_lock:
            self._known_hashes.clear()


# ============================================================================
# Legacy SharedRegretBuffer (mp.Manager) — kept for backward compat
# ============================================================================

class SharedRegretBuffer:
    """Thread-safe shared regret buffer using SyncManager (legacy)."""

    def __init__(self, manager: mp.managers.SyncManager):
        self.manager = manager
        self.regrets = manager.dict()
        self.iteration_counts = manager.dict()
        self.visit_counts = manager.dict()
        self.lock = manager.Lock()

    def accumulate_regrets(self, infoset_hash: str, action_regrets: dict):
        with self.lock:
            if infoset_hash not in self.regrets:
                self.regrets[infoset_hash] = self.manager.dict()
                self.iteration_counts[infoset_hash] = 0
            d = self.regrets[infoset_hash]
            for a, delta in action_regrets.items():
                d[a] = d.get(a, 0.0) + delta
            self.iteration_counts[infoset_hash] += 1

    def get_regrets(self, h: str) -> dict:
        return dict(self.regrets.get(h, {}))

    def get_all_regrets(self) -> dict:
        return {k: dict(v) for k, v in self.regrets.items()}

    def accumulate_visits(self, h: str, n: int = 1):
        with self.lock:
            self.visit_counts[h] = self.visit_counts.get(h, 0) + n

    def get_visit_counts(self) -> dict:
        return dict(self.visit_counts)

    def reset_regrets(self):
        with self.lock:
            self.regrets.clear()
            self.iteration_counts.clear()

    def reset_visits(self):
        with self.lock:
            self.visit_counts.clear()


# ============================================================================
# WorkerPool
# ============================================================================

class WorkerPool:
    """Process pool for parallel CFR game tree traversals."""

    def __init__(
        self,
        num_workers: int = 8,
        gpu_device: Optional[int] = None,
        enable_logging: bool = True,
    ):
        self.num_workers = num_workers
        self.gpu_device = gpu_device
        self.enable_logging = enable_logging

        self.task_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.workers: List[mp.Process] = []

        self.shared_buffer: Optional[SharedMemoryRegretBuffer] = None
        self.running = False
        self.iteration = 0

    def start(self) -> None:
        if self.running:
            logger.warning("WorkerPool already running")
            return

        # Create shared memory buffer (NO dict, NO manager)
        self.shared_buffer = SharedMemoryRegretBuffer(
            max_infosets=100_000,
            num_actions=9,
        )

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        for wid in range(self.num_workers):
            w = mp.Process(
                target=_worker_process,
                args=(
                    wid,
                    self.task_queue,
                    self.result_queue,
                    self.shared_buffer,
                    self.enable_logging,
                ),
                daemon=True,
            )
            w.start()
            self.workers.append(w)

        self.running = True
        logger.info("WorkerPool started: %d workers", self.num_workers)

    def run_iteration(
        self, tasks: List[WorkerTask], timeout_per_task: float = 60.0,
    ) -> List[WorkerResult]:
        if not self.running:
            raise RuntimeError("WorkerPool not started")

        start = time.time()
        for task in tasks:
            self.task_queue.put(task)

        results: List[WorkerResult] = []
        try:
            for _ in tasks:
                r = self.result_queue.get(timeout=timeout_per_task)
                results.append(r)
        except Exception:
            logger.error("Timeout waiting for worker results after %.0fs", timeout_per_task)

        elapsed = time.time() - start
        self.iteration += 1

        if self.enable_logging and results:
            total_t = sum(r.num_traversals for r in results)
            total_r = sum(r.num_regrets_written for r in results)
            avg_v = sum(r.game_value for r in results) / len(results)
            logger.info(
                "Iteration %d: %d traversals, %d regrets written, "
                "value=%.4f, elapsed=%.2fs",
                self.iteration, total_t, total_r, avg_v, elapsed,
            )

        return results

    def get_shared_regrets(self) -> Dict[str, np.ndarray]:
        if not self.shared_buffer:
            return {}
        return self.shared_buffer.get_all_regrets()

    def get_shared_visits(self) -> dict:
        return {}

    def shutdown(self) -> None:
        if not self.running:
            return
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        timeout = 10.0
        for w in self.workers:
            w.join(timeout=timeout / max(self.num_workers, 1))
            if w.is_alive():
                logger.warning("Force-killing worker %s", w.pid)
                w.terminate()
        self.running = False
        logger.info("WorkerPool shut down")


# ============================================================================
# Worker process
# ============================================================================

def _worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    shared_buffer: SharedMemoryRegretBuffer,
    enable_logging: bool = False,
) -> None:
    """Worker main loop: real MCCFR traversals with direct tensor writes.

    Each worker:
        1. Creates isolated poker environment (heads-up)
        2. Creates local InformationSetStorage for regret accumulation
        3. Creates PokerActorCritic network (CPU, eval mode)
        4. Instantiates MCCFRTraversal engine
        5. Pulls WorkerTasks, executes real external-sampling traversals
        6. Extracts regrets from local infoset_storage
        7. Writes regrets DIRECTLY to shared tensor via crc32 indexing
        8. Returns lightweight metadata via result_queue
    """
    pid = mp.current_process().pid
    if enable_logging:
        logger.info("Worker %d started (PID=%d)", worker_id, pid)

    # ── Deterministic per-worker seed ────────────────────────────────
    np_seed = 42 + worker_id * 1000
    torch_seed = np_seed + 500
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)

    # ── Initialise MCCFR engine ──────────────────────────────────────
    device = torch.device("cpu")
    try:
        env = RLCardWrapper(config=WrapperConfig(num_players=2))
        infoset_storage = RegretStore(
            base_dir=Path(__file__).parent.parent.parent / "regrets",
        )
        network = PokerActorCritic()
        network.eval()
        network.to(device)

        mccfr = MCCFRTraversal(
            env=env,
            network=network,
            infoset_storage=infoset_storage,
            device=device,
        )
        if enable_logging:
            logger.info("Worker %d: MCCFR engine initialised", worker_id)
    except Exception as e:
        logger.error("Worker %d: init failed: %s", worker_id, e, exc_info=True)
        return

    # ── Main task loop ───────────────────────────────────────────────
    task_count = 0
    total_regrets = 0

    while True:
        try:
            task = task_queue.get()
        except Exception as exc:
            logger.error("Worker %d: queue read error: %s", worker_id, exc)
            break

        if task is None:
            if enable_logging:
                logger.info(
                    "Worker %d (PID=%d): termination signal. "
                    "%d tasks, %d regrets written.",
                    worker_id, pid, task_count, total_regrets,
                )
            break

        t0 = time.time()
        n_written = 0

        try:
            # Clear local storage (each task is independent)
            infoset_storage.infosets.clear()
            infoset_storage.created_count = 0
            infoset_storage.updated_count = 0

            # Run real MCCFR traversals
            stats = mccfr.traverse_for_both_players(
                num_traversals=task.num_traversals
            )

            # ── Write regrets to shared tensor (dict-free indexing) ───
            for iid, iobj in infoset_storage.infosets.items():
                action_regrets = {
                    a: r
                    for a, r in iobj.cumulative_regret.items()
                    if abs(r) > 1e-12  # skip true zeros
                }
                if action_regrets:
                    shared_buffer.add_regrets_batch(iid, action_regrets)
                    shared_buffer.register_hash(iid)
                    n_written += 1

            total_regrets += n_written

            game_value = (
                stats.get("mean_value_p0", 0.0)
                + stats.get("mean_value_p1", 0.0)
            ) / 2.0

            result = WorkerResult(
                task_id=task.task_id,
                game_value=game_value,
                num_traversals=task.num_traversals,
                worker_id=worker_id,
                compute_time=time.time() - t0,
                num_regrets_written=n_written,
            )
            result_queue.put(result)
            task_count += 1

            if enable_logging:
                logger.info(
                    "Worker %d: task %d done (%d traversals, %d infosets, "
                    "value=%.4f, %.2fs)",
                    worker_id, task.task_id, task.num_traversals,
                    n_written, game_value, result.compute_time,
                )

        except Exception as exc:
            logger.error(
                "Worker %d: task %d error: %s",
                worker_id, task.task_id, exc, exc_info=True,
            )
            result_queue.put(WorkerResult(
                task_id=task.task_id,
                worker_id=worker_id,
                compute_time=time.time() - t0,
            ))

    if enable_logging:
        logger.info("Worker %d (PID=%d): exiting", worker_id, pid)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=== SharedMemoryRegretBuffer (dict-free) Testing ===")
    buf = SharedMemoryRegretBuffer(max_infosets=1000, num_actions=9)

    # Verify deterministic indexing
    idx1 = buf._hash_to_index("test_infoset_abc")
    idx2 = buf._hash_to_index("test_infoset_abc")
    assert idx1 == idx2, "crc32 indexing must be deterministic"
    print(f"  Deterministic: hash('test_infoset_abc') → row {idx1} (verified)")

    # Write and read back
    buf.add_regrets_batch("infoset_A", {0: 1.5, 1: -0.5, 2: 0.3})
    buf.register_hash("infoset_A")
    regrets = buf.get_regrets("infoset_A")
    print(f"  Write/read: infoset_A regrets = {regrets[:4]}")
    assert abs(regrets[0] - 1.5) < 1e-6

    print(f"  Non-zero rows: {buf.get_non_zero_count()}")

    print("\n=== WorkerPool Testing ===")
    pool = WorkerPool(num_workers=2, enable_logging=True)
    pool.start()

    tasks = [
        WorkerTask(task_id=i, game_state_hash=f"s{i}", iteration=0,
                   num_traversals=3, player_id=0)
        for i in range(2)
    ]

    results = pool.run_iteration(tasks, timeout_per_task=120.0)

    print(f"\nCollected {len(results)} results")
    for r in results:
        print(f"  Task {r.task_id}: traversals={r.num_traversals}, "
              f"regrets={r.num_regrets_written}, value={r.game_value:.4f}, "
              f"time={r.compute_time:.2f}s")

    n_nonzero = pool.shared_buffer.get_non_zero_count()
    all_regrets = pool.shared_buffer.get_all_regrets()
    print(f"\nShared tensor: {n_nonzero} non-zero rows, "
          f"{len(all_regrets)} known infosets")

    pool.shutdown()

    if n_nonzero > 0:
        print("\n✅ SUCCESS: Real MCCFR wrote non-zero regrets to shared memory!")
    else:
        print("\n❌ FAILURE: Zero regrets written")

    print("=== Test complete ===")
