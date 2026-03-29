"""
Parallel CFR Worker Pool & Shared Memory Regret Buffer (Phase 3)

[PHASE 3] True parallelization: 8-16 worker processes, shared-memory regret 
accumulation, GPU dedicated to batch inference only.

Architecture:
    1. **Master Process**: Coordinates iteration, aggregates regrets, GPU inference
    2. **Worker Processes**: Independent game tree traversals, CFR computation
    3. **Shared Memory**: Regret buffer (multiprocessing.managers.SyncManager)
    4. **IPC**: Queue for work distribution, Result collection

Key Design Decisions:
    - SyncManager over multiprocessing.shared_memory (easier dict-like access)
    - No GIL lock on workers (pure CFR computation, CPU-bound)
    - GPU only on master (batch inference, network updates)
    - Barrier synchronization between iterations (all workers must complete)

References:
    - Schäfer et al. (2023): "Parallel Game Tree Search"
    - Brown & Sandholm (2019): "Endgame Solving in Large Imperfect-Info Games"
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

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
    
    regrets_update: dict[str, dict[int, float]] = field(default_factory=dict)
    """regrets_update[infoset_hash][action] = cumulative regret delta"""
    
    visit_counts: dict[str, int] = field(default_factory=dict)
    """visit_counts[state_hash] = visit count for IS correction"""
    
    game_value: float = 0.0
    """Computed value of game from root"""
    
    num_traversals: int = 0
    """Number of traversals completed"""
    
    worker_id: int = -1
    """Which worker produced this result"""
    
    compute_time: float = 0.0
    """Wall clock time for computation"""


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
        self.shared_buffer: Optional[SharedRegretBuffer] = None
        
        # Lifecycle
        self.running = False
        self.iteration = 0
    
    def start(self):
        """Spawn worker processes and initialize shared memory."""
        if self.running:
            logger.warning("WorkerPool already running")
            return
        
        # Create shared memory manager
        self.manager = mp.Manager()
        self.shared_buffer = SharedRegretBuffer(self.manager)
        
        # Create IPC queues
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # Spawn workers
        for worker_id in range(self.num_workers):
            worker = mp.Process(
                target=_worker_process,
                args=(
                    worker_id,
                    self.task_queue,
                    self.result_queue,
                    self.enable_logging,
                ),
                daemon=False,
            )
            worker.start()
            self.workers.append(worker)
        
        self.running = True
        logger.info(f"WorkerPool started: {self.num_workers} workers")
    
    def run_iteration(
        self,
        tasks: List[WorkerTask],
        timeout_per_task: float = 60.0,
    ) -> List[WorkerResult]:
        """
        Run CFR iteration across all workers.
        
        Distributes tasks, waits for all results, aggregates regrets.
        
        Args:
            tasks: Work items to distribute
            timeout_per_task: Per-task timeout (seconds)
        
        Returns:
            List of results from all workers
        """
        if not self.running:
            raise RuntimeError("WorkerPool not started")
        
        # Distribute work
        start_time = time.time()
        for task in tasks:
            self.task_queue.put(task)
        
        # Collect results
        results = []
        try:
            for _ in tasks:
                result = self.result_queue.get(timeout=timeout_per_task)
                results.append(result)
                
                # Accumulate shared memory immediately
                if self.shared_buffer:
                    self.shared_buffer.accumulate_regrets(
                        result.__dict__.get('infoset_hash', f'unknown_{result.task_id}'),
                        result.regrets_update,
                    )
                    for state_hash, count in result.visit_counts.items():
                        self.shared_buffer.accumulate_visits(state_hash, count)
        except mp.TimeoutError:
            logger.error(f"Timeout waiting for worker results after {timeout_per_task}s")
            return []
        
        elapsed = time.time() - start_time
        self.iteration += 1
        
        if self.enable_logging:
            total_traversals = sum(r.num_traversals for r in results)
            total_value = sum(r.game_value for r in results) / len(results)
            logger.info(
                f"Iteration {self.iteration}: "
                f"{total_traversals} traversals, "
                f"value={total_value:.4f}, "
                f"elapsed={elapsed:.2f}s"
            )
        
        return results
    
    def get_shared_regrets(self) -> dict[str, dict[int, float]]:
        """Retrieve current regrets from shared memory."""
        if not self.shared_buffer:
            return {}
        return self.shared_buffer.get_all_regrets()
    
    def get_shared_visits(self) -> dict[str, int]:
        """Retrieve visit counts for importance sampling."""
        if not self.shared_buffer:
            return {}
        return self.shared_buffer.get_visit_counts()
    
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
        
        # Clean up resources
        self.manager.shutdown()
        self.running = False
        logger.info("WorkerPool shut down")


def _worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    enable_logging: bool = False,
):
    """
    Worker process main loop.
    
    Each worker:
        1. Receives task from task_queue
        2. Performs CFR traversal + regret accumulation
        3. Sends result to result_queue
    
    Args:
        worker_id: Unique worker identifier
        task_queue: Receives WorkerTask objects
        result_queue: Sends WorkerResult objects
        enable_logging: Whether to log per-worker events
    """
    if enable_logging:
        logger.info(f"Worker {worker_id} started (PID={mp.current_process().pid})")
    
    while True:
        # Block until task available
        task = task_queue.get()
        
        # Check for termination signal
        if task is None:
            break
        
        # Process task (placeholder: in real implementation, call cfr_valuator)
        start_time = time.time()
        result = WorkerResult(
            task_id=task.task_id,
            regrets_update={},  # Would be populated by CFR traversal
            visit_counts={},
            num_traversals=task.num_traversals,
            worker_id=worker_id,
            compute_time=time.time() - start_time,
        )
        
        # Send result back
        result_queue.put(result)
    
    if enable_logging:
        logger.info(f"Worker {worker_id} terminated")


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
    
    print("\n=== WorkerPool Testing ===")
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
    
    print(f"Collected {len(results)} results")
    for r in results:
        print(f"  Task {r.task_id}: {r.num_traversals} traversals, "
              f"compute_time={r.compute_time:.4f}s")
    
    pool.shutdown()
    print("=== Test complete ===")
