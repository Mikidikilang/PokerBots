#!/usr/bin/env python3
"""
Strict verification: Real MCCFR traversal writing valid regrets to shared memory.

This script tests the worker pool directly with aggressive timeout.
"""

import sys
import time
import logging

from src.training.parallel_cfr import WorkerPool, WorkerTask

# Set up logging to see only warnings and errors (suppress debug spam)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

def main():
    print("=" * 80)
    print("STRICT VERIFICATION: Real MCCFR Traversal -> Shared Memory Regrets")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Initialize worker pool
    # ========================================================================
    print("[STEP 1] Initializing WorkerPool with 1 worker...")
    pool = WorkerPool(
        num_workers=1,
        enable_logging=True,
    )
    print("  [OK] WorkerPool initialized")
    print()

    # ========================================================================
    # Step 2: Start the pool (creates internal shared buffer)
    # ========================================================================
    print("[STEP 2] Starting worker pool...")
    sys.stdout.flush()
    pool.start()
    time.sleep(2)  # Give worker time to start
    print("  [OK] Pool started, worker process spawned")
    print(f"  [OK] Internal shared buffer: {pool.shared_buffer.max_infosets} infosets x {pool.shared_buffer.num_actions} actions")
    print()
    sys.stdout.flush()

    # ========================================================================
    # Step 3: Create ONE task
    # ========================================================================
    print("[STEP 3] Creating 1 task (num_traversals=3)...")
    task = WorkerTask(
        task_id=1,
        game_state_hash="root_game_state",
        iteration=1,
        num_traversals=3,
        player_id=0,
    )
    print(f"  [OK] Task created: task_id={task.task_id}, num_traversals={task.num_traversals}")
    print()
    sys.stdout.flush()

    # ========================================================================
    # Step 4: Submit the task with SHORT timeout
    # ========================================================================
    print("[STEP 4] Running iteration with 1 task (timeout=90s)...")
    sys.stdout.flush()
    
    try:
        results = pool.run_iteration(tasks=[task], timeout_per_task=90.0)
        print(f"  [OK] Received {len(results)} result(s)")
        for result in results:
            print(f"    - Task {result.task_id}: value={result.game_value:.4f}, "
                  f"regrets_written={result.num_regrets_written}, "
                  f"compute_time={result.compute_time:.2f}s")
    except Exception as e:
        print(f"  [FAIL] TIMEOUT or ERROR: {type(e).__name__}: {e}")
        print(f"\n  Waiting 5 more seconds to see if worker produces output...")
        time.sleep(5)
        sys.stdout.flush()
    
    print()
    sys.stdout.flush()

    # ========================================================================
    # Step 5: Shutdown pool
    # ========================================================================
    print("[STEP 5] Shutting down worker pool...")
    sys.stdout.flush()
    pool.shutdown()
    print("  [OK] Pool shut down")
    print()

    # ========================================================================
    # Step 6: Inspect shared tensor
    # ========================================================================
    print("[STEP 6] Analyzing shared memory tensor...")
    non_zero_count = (pool.shared_buffer.regrets != 0.0).sum().item()
    print(f"  Non-zero entries in tensor: {non_zero_count}")
    print(f"  Total tensor size: {pool.shared_buffer.regrets.shape}")
    print()

    # ========================================================================
    # Step 7: Dump all regrets
    # ========================================================================
    print("[STEP 7] Full regret contents from shared buffer:")
    print("-" * 80)
    all_regrets = pool.shared_buffer.get_all_regrets()
    
    if not all_regrets:
        print("  (Empty - no regrets written)")
    else:
        print(f"  Total infosets with regrets: {len(all_regrets)}")
        print()
        for infoset_hash, regret_array in sorted(all_regrets.items()):
            print(f"  Infoset: '{infoset_hash}'")
            print(f"    Regrets: {regret_array}")
            non_zero_in_iset = (regret_array != 0.0).sum()
            print(f"    Non-zero actions: {non_zero_in_iset}/9")
            print()
    
    print("-" * 80)
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("[SUMMARY]")
    print(f"  Non-zero entries in shared tensor: {non_zero_count}")
    print(f"  Infosets with regrets: {len(all_regrets)}")
    
    if non_zero_count > 0:
        print("  [OK] SUCCESS: Real MCCFR traversal wrote valid regrets to shared memory!")
        return True
    else:
        print("  [FAIL] FAILURE: No regrets were written to shared memory")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] Script failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
