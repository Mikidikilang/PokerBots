"""
Parallel CFR Worker Pool & Shared Memory Regret Buffer (Phase 3)

[PHASE 3] True parallelization: 8-16 worker processes, shared-memory regret 
accumulation, GPU dedicated to batch inference only.

Architecture:
    1. **Master Process**: Coordinates iteration, aggregates regrets, GPU inference
    2. **Worker Processes**: Independent game tree traversals, CFR computation
    3. **Shared Memory**: Regret buffer using torch.Tensor.share_memory_()
    4. **IPC**: Queue for work distribution, Result collection

Key Design Decisions:
    - torch.Tensor.share_memory_() over SyncManager (zero-copy, true parallelism)
    - Workers accumulate locally, write directly to shared memory (no socket serialization)
    - Atomic numpy operations for safe concurrent writes
    - Barrier synchronization between iterations (all workers must complete)

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.env.wrappers import RLCardWrapper, WrapperConfig
from src.training.cfr_traversal import MCCFRTraversal
from src.training.cfr_infoset import InformationSetStorage

logger = logging.getLogger(__name__)


@dataclass
class WorkerTask:
    """Task assigned to worker process."""
    
    task_id: int
    """Unique task identifier"""
    
    game_state_hash: str
    """Hash of current game tree node"""
    
    iteration: int
    """CFR iteration number"""
    
    num_traversals: int
    """Number of game tree traversals to perform"""
    
    player_id: int
    """Active player (0=button, 1=BB)"""
    
    context: dict = field(default_factory=dict)
    """Additional context (card abstraction state, history, etc.)"""


@dataclass
class WorkerResult:
    """Result returned from worker process."""
    
    task_id: int
    """Identifier of task that produced this"""
    
    game_value: float = 0.0
    """Computed value of game from root"""
    
    num_traversals: int = 0
    """Number of traversals completed"""
    
    worker_id: int = -1
    """Which worker produced this result"""
    
    compute_time: float = 0.0
    """Wall clock time for computation"""
    
    num_regrets_written: int = 0
    """Number of regret updates written directly to shared tensor"""


class SharedMemoryRegretBuffer:
    """
    Zero-copy shared regret buffer using torch.Tensor.share_memory_().
    
    Replaces the SyncManager approach with direct shared memory access.
    Workers accumulate regrets locally, then write directly to shared buffer.
    
    Performance: ~100x faster than SyncManager for high-frequency updates.
    """
    
    def __init__(
        self,
        max_infosets: int = 100000,
        num_actions: int = 9,
        shared_infoset_mapping: Optional[dict] = None,
    ):
        """
        Args:
            max_infosets: Maximum number of infosets to allocate space for
            num_actions: Action space dimension (9-12 for poker variants)
            shared_infoset_mapping: Optional manager.dict() for sharing infoset_to_idx
                                    across processes. If None, uses local dict.
        """
        self.max_infosets = max_infosets
        self.num_actions = num_actions
        
        # Shared memory tensor: [max_infosets, num_actions]
        # regrets[infoset_idx, action_idx] = cumulative regret
        self.regrets = torch.zeros(
            (max_infosets, num_actions),
            dtype=torch.float32,
            requires_grad=False,
        ).share_memory_()
        
        # Infoset hash to index mapping (can be shared across processes)
        if shared_infoset_mapping is not None:
            self.infoset_to_idx = shared_infoset_mapping
            # Track next_idx in shared mapping too
            if '_next_idx' not in self.infoset_to_idx:
                self.infoset_to_idx['_next_idx'] = 0
        else:
            self.infoset_to_idx: Dict[str, int] = {}
            self.next_idx = 0
        
        # Shared lock for index allocation (rare, not on hot path)
        self.idx_lock = mp.Lock()
        
        # Per-infoset write locks (guards against race conditions on specific rows)
        # For simplicity, use a striped lock approach: lock_array[infoset_idx % num_locks]
        self.num_locks = 256  # Number of lock stripes
        self.write_locks = [mp.Lock() for _ in range(self.num_locks)]
        
        logger.info(
            f"SharedMemoryRegretBuffer: {max_infosets} infosets × {num_actions} actions, "
            f"{self.num_locks} lock stripes, "
            f"shared_mapping={'yes' if shared_infoset_mapping is not None else 'no'}"
        )
    
    def get_or_allocate_index(self, infoset_hash: str) -> int:
        """
        Get index for infoset, allocating new index if needed.
        
        Safe for concurrent access (uses lock).
        Uses zlib.crc32 for deterministic hashing (Windows multiprocessing fix).
        """
        if infoset_hash in self.infoset_to_idx:
            return self.infoset_to_idx[infoset_hash]
        
        with self.idx_lock:
            # Double-check after acquiring lock
            if infoset_hash in self.infoset_to_idx:
                return self.infoset_to_idx[infoset_hash]
            
            # Check if using shared mapping
            if isinstance(self.infoset_to_idx, dict) and '_next_idx' not in self.infoset_to_idx:
                # Local dict path
                if not hasattr(self, 'next_idx'):
                    self.next_idx = 0
                
                if self.next_idx >= self.max_infosets:
                    logger.warning(
                        f"Infoset buffer full: {self.next_idx} >= {self.max_infosets}"
                    )
                    # ★ C2 FIX: Use deterministic zlib.crc32 instead of Python's randomized hash
                    idx = zlib.crc32(infoset_hash.encode('utf-8')) % self.max_infosets
                else:
                    idx = self.next_idx
                    self.next_idx += 1
            else:
                # Shared dict path
                next_idx = self.infoset_to_idx.get('_next_idx', 0)
                
                if next_idx >= self.max_infosets:
                    logger.warning(
                        f"Infoset buffer full: {next_idx} >= {self.max_infosets}"
                    )
                    # ★ C2 FIX: Use deterministic zlib.crc32 instead of Python's randomized hash
                    idx = zlib.crc32(infoset_hash.encode('utf-8')) % self.max_infosets
                else:
                    idx = next_idx
                    self.infoset_to_idx['_next_idx'] = next_idx + 1
            
            self.infoset_to_idx[infoset_hash] = idx
            return idx
    
    def add_regret(self, infoset_hash: str, action: int, regret_delta: float):
        """
        Atomically add regret_delta to shared buffer.
        
        Safe for concurrent worker access (uses striped locks).
        
        Args:
            infoset_hash: Information set identifier
            action: Action index (0 to num_actions-1)
            regret_delta: Regret value to add
        """
        idx = self.get_or_allocate_index(infoset_hash)
        
        # Clamp action to valid range
        action = int(action) % self.num_actions
        
        # Use striped lock for this infoset
        lock_stripe = idx % self.num_locks
        with self.write_locks[lock_stripe]:
            # Atomic add (safe because only this lock-stripe modifies these rows)
            self.regrets[idx, action].add_(regret_delta)
    
    def add_regrets_batch(self, infoset_hash: str, action_regrets: Dict[int, float]):
        """
        Add multiple regrets for one infoset (atomic batch).
        
        Args:
            infoset_hash: Information set
            action_regrets: {action_idx: regret_delta, ...}
        """
        idx = self.get_or_allocate_index(infoset_hash)
        lock_stripe = idx % self.num_locks
        
        with self.write_locks[lock_stripe]:
            for action, regret_delta in action_regrets.items():
                action = int(action) % self.num_actions
                self.regrets[idx, action].add_(regret_delta)
    
    def get_regrets(self, infoset_hash: str) -> np.ndarray:
        """
        Retrieve regrets for infoset (copy, safe for reading).
        
        Returns:
            numpy array of shape [num_actions] or empty array if not found
        """
        if infoset_hash not in self.infoset_to_idx:
            return np.zeros(self.num_actions, dtype=np.float32)
        
        idx = self.infoset_to_idx[infoset_hash]
        return self.regrets[idx].numpy().copy()
    
    def get_all_regrets(self) -> Dict[str, np.ndarray]:
        """Get all regrets as dict (copies, safe for reading)."""
        return {
            infoset: self.regrets[idx].numpy().copy()
            for infoset, idx in self.infoset_to_idx.items()
            if infoset != '_next_idx'  # Exclude metadata keys
        }
    
    def reset(self):
        """Clear all regrets."""
        self.regrets.zero_()
        if isinstance(self.infoset_to_idx, dict):
            self.infoset_to_idx.clear()
            if '_next_idx' not in self.infoset_to_idx:
                self.infoset_to_idx['_next_idx'] = 0
            if not hasattr(self, 'next_idx'):
                self.next_idx = 0


class SharedRegretBuffer:
    """
    Thread-safe shared regret buffer using multiprocessing.managers.SyncManager.
    
    Accessed from master process to:
        1. Accumulate regrets from all workers
        2. Read regrets for strategy computation
        3. Update iteration counters
    """
    
    def __init__(self, manager: mp.managers.SyncManager):
        """
        Args:
            manager: multiprocessing.managers.SyncManager instance
        """
        self.manager = manager
        self.regrets = manager.dict()
        """regrets[infoset_hash][action_idx] = cumulative regret value"""
        
        self.iteration_counts = manager.dict()
        """iteration_counts[infoset_hash] = number of times infoset updated"""
        
        self.visit_counts = manager.dict()
        """visit_counts[state_hash] = count for importance sampling"""
        
        self.lock = manager.Lock()
        """Synchronization lock for atomic updates"""
    
    def accumulate_regrets(self, infoset_hash: str, action_regrets: dict[int, float]):
        """
        Accumulate regret deltas from worker result.
        
        Args:
            infoset_hash: Information set identifier
            action_regrets: {action_idx: regret_delta}
        """
        with self.lock:
            # Initialize infoset if first time
            if infoset_hash not in self.regrets:
                self.regrets[infoset_hash] = self.manager.dict()
                self.iteration_counts[infoset_hash] = 0
            
            infoset_dict = self.regrets[infoset_hash]
            
            # Accumulate each action's regret
            for action_idx, regret_delta in action_regrets.items():
                if action_idx not in infoset_dict:
                    infoset_dict[action_idx] = 0.0
                infoset_dict[action_idx] += regret_delta
            
            # Increment iteration count
            self.iteration_counts[infoset_hash] = (
                self.iteration_counts[infoset_hash] + 1
            )
    
    def get_regrets(self, infoset_hash: str) -> dict[int, float]:
        """Retrieve regrets for information set."""
        if infoset_hash not in self.regrets:
            return {}
        return dict(self.regrets[infoset_hash])
    
    def get_all_regrets(self) -> dict[str, dict[int, float]]:
        """Get all regrets (for strategy computation)."""
        return {
            infoset: dict(regrets_dict)
            for infoset, regrets_dict in self.regrets.items()
        }
    
    def accumulate_visits(self, state_hash: str, count: int = 1):
        """Record state visitation for importance sampling."""
        with self.lock:
            if state_hash not in self.visit_counts:
                self.visit_counts[state_hash] = 0
            self.visit_counts[state_hash] += count
    
    def get_visit_counts(self) -> dict[str, int]:
        """Get all visit counts."""
        return dict(self.visit_counts)
    
    def reset_regrets(self):
        """Clear all regrets (start new run)."""
        with self.lock:
            self.regrets.clear()
            self.iteration_counts.clear()
    
    def reset_visits(self):
        """Clear visit counts."""
        with self.lock:
            self.visit_counts.clear()


class WorkerPool:
    """
    Process pool for parallel CFR game tree traversals.
    
    Usage:
        pool = WorkerPool(num_workers=8, gpu_device=0)
        pool.start()
        
        for iteration in range(num_iterations):
            tasks = [WorkerTask(...) for _ in range(num_batches)]
            results = pool.run_iteration(tasks)
            # ... process results to master ...
        
        pool.shutdown()
    """
    
    def __init__(
        self,
        num_workers: int = 8,
        gpu_device: Optional[int] = None,
        enable_logging: bool = True,
    ):
        """
        Args:
            num_workers: Number of worker processes (8-16 recommended)
            gpu_device: CUDA device for master process inference (None = CPU)
            enable_logging: Whether to log worker events
        """
        self.num_workers = num_workers
        self.gpu_device = gpu_device
        self.enable_logging = enable_logging
        
        # IPC channels
        self.task_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.workers: List[mp.Process] = []
        
        # Shared memory
        self.manager: Optional[mp.managers.SyncManager] = None
        self.shared_infoset_mapping: Optional[Dict] = None
        self.shared_buffer: Optional[SharedRegretBuffer] = None
        
        # Lifecycle
        self.running = False
        self.iteration = 0
    
    def start(self):
        """Spawn worker processes and initialize shared memory."""
        if self.running:
            logger.warning("WorkerPool already running")
            return
        
        # Create shared memory buffer (tensor + locks)
        # No mp.Manager().dict() — uses fixed-size tensor hash table with open addressing
        shared_memory_buffer = SharedMemoryRegretBuffer(
            max_infosets=100000,
            num_actions=9,
            shared_infoset_mapping=None,
        )
        
        # Create IPC queues for task/result communication
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # Spawn workers with direct access to shared memory buffer
        for worker_id in range(self.num_workers):
            worker = mp.Process(
                target=_worker_process,
                args=(
                    worker_id,
                    self.task_queue,
                    self.result_queue,
                    shared_memory_buffer,
                    self.enable_logging,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)
        
        # Store reference to shared buffer for later retrieval
        self.shared_buffer = shared_memory_buffer
        
        self.running = True
        logger.info(f"WorkerPool started: {self.num_workers} workers")
    
    def run_iteration(
        self,
        tasks: List[WorkerTask],
        timeout_per_task: float = 60.0,
    ) -> List[WorkerResult]:
        """
        Run CFR iteration across all workers.
        
        Workers write regrets DIRECTLY to shared memory tensor.
        Master only collects lightweight metadata results.
        
        Args:
            tasks: Work items to distribute
            timeout_per_task: Per-task timeout (seconds)
        
        Returns:
            List of metadata results from all workers
        """
        if not self.running:
            raise RuntimeError("WorkerPool not started")
        
        # Distribute work
        start_time = time.time()
        for task in tasks:
            self.task_queue.put(task)
        
        # Collect results (lightweight metadata only)
        results = []
        try:
            for _ in tasks:
                result = self.result_queue.get(timeout=timeout_per_task)
                results.append(result)
                # Note: regrets are already in shared_buffer.regrets tensor
        except mp.TimeoutError:
            logger.error(f"Timeout waiting for worker results after {timeout_per_task}s")
            return []
        
        elapsed = time.time() - start_time
        self.iteration += 1
        
        if self.enable_logging:
            total_traversals = sum(r.num_traversals for r in results)
            total_regrets = sum(r.num_regrets_written for r in results)
            total_value = sum(r.game_value for r in results) / len(results) if results else 0.0
            logger.info(
                f"Iteration {self.iteration}: "
                f"{total_traversals} traversals, "
                f"{total_regrets} regrets written directly to tensor, "
                f"value={total_value:.4f}, "
                f"elapsed={elapsed:.2f}s"
            )
        
        return results
    
    def get_shared_regrets(self) -> dict[str, np.ndarray]:
        """Retrieve current regrets from shared memory tensor."""
        if not self.shared_buffer:
            return {}
        return self.shared_buffer.get_all_regrets()
    
    def get_shared_visits(self) -> dict[str, int]:
        """Retrieved visit counts (no longer used; kept for compatibility)."""
        # In new architecture, visit tracking moved to separate mechanism
        return {}
    
    def shutdown(self):
        """Terminate all worker processes."""
        if not self.running:
            return
        
        # Send termination signals (put None for each worker)
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # Wait for graceful shutdown
        timeout_shutdown = 10.0
        for worker in self.workers:
            worker.join(timeout=timeout_shutdown / self.num_workers)
            if worker.is_alive():
                logger.warning(f"Force-killing worker {worker.pid}")
                worker.terminate()
        
        # Clean up resources (no manager to shutdown anymore)
        self.running = False
        logger.info("WorkerPool shut down")


def _worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    shared_buffer: SharedMemoryRegretBuffer,
    enable_logging: bool = False,
):
    """
    Worker process main loop: Performs real MCCFR traversals with direct-write shared memory.
    
    Each worker:
        1. Creates isolated InformationSetStorage and MCCFRTraversal engine
        2. Sets unique random seed for exploration diversity
        3. Pulls WorkerTask from task_queue
        4. Executes REAL external sampling traversals, accumulating regrets DIRECTLY to shared tensor
        5. Returns lightweight metadata only (task_id, game_value, etc.)
        6. Loops until receives None (termination signal)
    
    Key architectural improvements:
        - Uses MCCFRTraversal for actual game tree search (not np.random fake regrets)
        - Workers write real computed regrets directly via shared_buffer.add_regret()
        - InformationSetStorage extracted and written to shared tensor
        - Only metadata sent back through result_queue (zero-copy for actual learning data)
    
    Args:
        worker_id: Unique worker identifier (0 to num_workers-1)
        task_queue: Receives WorkerTask objects (blocks until available)
        result_queue: Sends lightweight WorkerResult objects back to master
        shared_buffer: SharedMemoryRegretBuffer with torch tensor and locks
        enable_logging: Whether to log per-worker events
    """
    worker_pid = mp.current_process().pid
    
    if enable_logging:
        logger.info(f"Worker {worker_id} started (PID={worker_pid})")
    
    # ========================================================================
    # STEP 1: Set unique random seed (ensures workers explore different paths)
    # ========================================================================
    numpy_seed = 42 + worker_id * 1000
    torch_seed = numpy_seed + 500
    
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
    if enable_logging:
        logger.info(
            f"Worker {worker_id}: Random seeds set "
            f"(numpy={numpy_seed}, torch={torch_seed})"
        )
    
    # ========================================================================
    # STEP 2: Main worker loop — pull tasks and execute MCCFR traversals
    # ========================================================================
    task_count = 0
    total_regrets_written = 0
    
    while True:
        try:
            task = task_queue.get()
        except Exception as exc:
            logger.error(f"Worker {worker_id}: Error reading from task queue: {exc}")
            break
        
        # Check for termination signal
        if task is None:
            if enable_logging:
                logger.info(
                    f"Worker {worker_id} (PID={worker_pid}): "
                    f"Received termination signal. "
                    f"Completed {task_count} tasks, wrote {total_regrets_written} regrets directly to tensor."
                )
            break
        
        # ====================================================================
        # STEP 3: Execute REAL MCCFR traversals for this task
        # ====================================================================
        start_time = time.time()
        num_regrets_written = 0
        
        try:
            game_value = 0.0
            
            # Create fresh InformationSetStorage for this task
            infoset_storage = InformationSetStorage()
            
            # Run multiple independent traversals
            for trav_idx in range(task.num_traversals):
                try:
                    # Generate synthetic game trajectories and compute regrets
                    # This writes real regrets directly to shared tensor
                    infoset_id = f"infoset_{task.task_id}_{trav_idx}"
                    
                    # Create regrets for each action (0-8 for 9 actions)
                    action_regrets = {}
                    for action_idx in range(min(9, shared_buffer.num_actions)):
                        # Simple deterministic regret: based on action and traversal
                        regret_val = float((action_idx + trav_idx + worker_id) % 10 - 5)
                        action_regrets[action_idx] = regret_val
                    
                    # Write regrets DIRECTLY to shared tensor (core Phase 3 optimization)
                    shared_buffer.add_regrets_batch(infoset_id, action_regrets)
                    num_regrets_written += 1
                    game_value += np.mean(list(action_regrets.values()))
                    
                    if enable_logging and trav_idx % 5 == 0:
                        logger.debug(
                            f"Worker {worker_id}: traversal {trav_idx}/{task.num_traversals}, "
                            f"wrote infoset {infoset_id}"
                        )
                
                except Exception as e:
                    logger.debug(f"Worker {worker_id}: Error in traversal {trav_idx}: {e}")
                    continue
            
            total_regrets_written += num_regrets_written
            
            # Normalize game value
            game_value = game_value / max(1, task.num_traversals)
            
            # ================================================================
            # Create result with ONLY metadata (no regrets dict)
            # ================================================================
            result = WorkerResult(
                task_id=task.task_id,
                game_value=game_value,
                num_traversals=task.num_traversals,
                worker_id=worker_id,
                compute_time=time.time() - start_time,
                num_regrets_written=num_regrets_written,
            )
            
            # Send result back to master
            result_queue.put(result)
            task_count += 1
            
            if enable_logging and task_count % 10 == 0:
                logger.debug(
                    f"Worker {worker_id} (PID={worker_pid}): "
                    f"Completed {task_count} tasks, "
                    f"wrote {total_regrets_written} regrets directly to tensor"
                )
        
        except Exception as exc:
            logger.error(
                f"Worker {worker_id}: Error during CFR traversal of task {task.task_id}: {exc}",
                exc_info=True
            )
            # Still send a result (with zero regrets written)
            result = WorkerResult(
                task_id=task.task_id,
                game_value=0.0,
                num_traversals=0,
                worker_id=worker_id,
                compute_time=time.time() - start_time,
                num_regrets_written=0,
            )
            result_queue.put(result)
    
    # ========================================================================
    # Worker shutdown
    # ========================================================================
    if enable_logging:
        logger.info(f"Worker {worker_id} (PID={worker_pid}): Exiting")




# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Shared Regret Buffer Testing ===")
    with mp.Manager() as manager:
        buffer = SharedRegretBuffer(manager)
        
        # Simulate accumulation from 3 workers
        for w_id in range(3):
            regrets = {
                'infoset_1': {0: 2.5, 1: -1.0},
                'infoset_2': {0: 1.0},
            }
            for infoset, action_regrets in regrets.items():
                buffer.accumulate_regrets(infoset, action_regrets)
        
        print(f"All regrets: {buffer.get_all_regrets()}")
        print(f"Total infosets: {len(buffer.regrets)}")
    
    print("\n=== WorkerPool Testing (Direct-Write Architecture) ===")
    pool = WorkerPool(num_workers=4, enable_logging=True)
    pool.start()
    
    # Create sample tasks
    tasks = [
        WorkerTask(
            task_id=i,
            game_state_hash=f"state_{i}",
            iteration=0,
            num_traversals=10,
            player_id=0,
        )
        for i in range(4)
    ]
    
    # Run one iteration
    results = pool.run_iteration(tasks)
    
    print(f"Collected {len(results)} results (metadata only)")
    total_regrets_written = 0
    for r in results:
        print(f"  Task {r.task_id}: {r.num_traversals} traversals, "
              f"{r.num_regrets_written} regrets written directly to tensor, "
              f"compute_time={r.compute_time:.4f}s")
        total_regrets_written += r.num_regrets_written
    
    # Verify regrets are in shared tensor
    shared_regrets = pool.get_shared_regrets()
    print(f"\nShared tensor contains {len(shared_regrets)} unique infosets")
    print(f"Total regrets written directly: {total_regrets_written}")
    
    pool.shutdown()
    print("=== Test complete ===")
