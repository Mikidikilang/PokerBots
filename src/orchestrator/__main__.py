"""
__main__.py - Tabular MCCFR Orchestrator with Parallel WorkerPool

Pure Monte Carlo Counterfactual Regret Minimization WITHOUT neural networks:
- SharedMemoryRegretBuffer: Zero-copy regret accumulation (dict-free, crc32 hashing)
- WorkerPool: Parallel game traversals across multiple processes
- Tabular strategy: Regret matching from accumulated regrets (no NN policy)
- 6-Player No-Limit Texas Hold'em

Usage:
    python -m src.orchestrator --config config.yaml --iterations 100 --num_workers 4
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
import importlib

# Setup logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Tabular MCCFR training with parallel WorkerPool (NO neural networks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--iterations", type=int, default=100, help="Number of MCCFR iterations")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel worker processes")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("TABULAR MCCFR ORCHESTRATOR (PARALLEL WORKERPOOL)")
    print("6-Player No-Limit Texas Hold'em | Zero-Copy Shared Memory | NO Neural Networks")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Iterations: {args.iterations}")
    print(f"Workers (parallel processes): {args.num_workers}")
    print()
    
    # ===== LOAD CONFIG =====
    print("[1] Loading configuration...")
    try:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"    [OK] Configuration loaded")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # ===== IMPORT TABULAR MCCFR COMPONENTS =====
    print("\n[2] Importing Tabular MCCFR infrastructure...")
    try:
        import torch
        from src.env.wrappers import make_env
        from src.training.parallel_cfr import (
            WorkerPool, WorkerTask, SharedMemoryRegretBuffer
        )
        print("    [OK] All imports successful")
        print("    [OK] WorkerPool, SharedMemoryRegretBuffer (dict-free, crc32)")
    except Exception as e:
        logger.error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== INIT W&B (OPTIONAL) =====
    print("\n[3] Initializing Weights & Biases (optional)...")
    run = None
    try:
        import wandb
        
        # Check if W&B is enabled in config
        wandb_cfg = config.get("mlops", {}).get("wandb", {})
        wandb_enabled = wandb_cfg.get("enabled", True)  # Default to enabled if not specified
        
        if wandb_enabled and wandb_cfg.get("project"):
            try:
                # Prepare W&B init parameters
                init_params = {
                    "project": wandb_cfg.get("project", "poker-mccfr-v5"),
                    "config": config,
                    "tags": ["tabular-mccfr", "parallel-workerpool", "6-player"],
                    "notes": f"Tabular MCCFR: {args.iterations} iterations, {args.num_workers} parallel workers"
                }
                
                # Only add entity if it's specified and not None
                entity = wandb_cfg.get("entity")
                if entity:
                    init_params["entity"] = entity
                
                run = wandb.init(**init_params)
                
                if run:
                    url = getattr(run, 'url', 'N/A')
                    run_id = getattr(run, 'id', 'N/A')
                    print(f"    [OK] W&B initialized (run_id={run_id}, {url})")
                else:
                    print(f"    [INFO] W&B init returned None (offline mode)")
                    run = None
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")
                print(f"    [WARN] W&B init failed: {e}")
                run = None
        else:
            print(f"    [SKIP] W&B disabled or project name missing")
    except ImportError as e:
        logger.info(f"W&B not installed, skipping: {e}")
        print(f"    [SKIP] W&B not installed")
        run = None
    except Exception as e:
        logger.warning(f"W&B init error: {e}")
        print(f"    [WARN] W&B init error: {e}")
        run = None
    
    # ===== INIT ENVIRONMENT (6-PLAYER ONLY) =====
    print("\n[4] Initializing 6-player environment...")
    try:
        env = make_env(cfg=config)
        env_cfg = config.get("environment", {})
        num_players = env_cfg.get("num_players", 6)
        if num_players != 6:
            logger.warning(f"Config has num_players={num_players}, forcing 6 for MCCFR")
            num_players = 6
        print(f"    [OK] Environment created (num_players={num_players})")
    except Exception as e:
        logger.error(f"Environment init failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== INIT SHARED MEMORY REGRET BUFFER =====
    print("\n[5] Initializing SharedMemoryRegretBuffer (dict-free, crc32 hashing)...")
    try:
        cfr_cfg = config.get("cfr", {})
        max_infosets = cfr_cfg.get("max_infosets", 100_000)
        num_actions = env_cfg.get("action_space", {}).get("num_actions", 9)
        
        regret_buffer = SharedMemoryRegretBuffer(
            max_infosets=max_infosets,
            num_actions=num_actions,
        )
        
        print(f"    [OK] Regret buffer initialized")
        print(f"        Max infosets: {max_infosets}")
        print(f"        Actions per infoset: {num_actions}")
        print(f"        Deterministic indexing: zlib.crc32 (no dict, no manager)")
    except Exception as e:
        logger.error(f"Regret buffer init failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== INIT WORKERPOOL =====
    print(f"\n[6] Initializing WorkerPool ({args.num_workers} parallel processes)...")
    try:
        worker_pool = WorkerPool(
            num_workers=args.num_workers,
            gpu_device=None,  # MCCFR is CPU-bound
            enable_logging=True,
        )
        worker_pool.start()
        print(f"    [OK] WorkerPool started ({args.num_workers} workers)")
    except Exception as e:
        logger.error(f"WorkerPool init failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== MAIN MCCFR LOOP =====
    print("\n" + "="*80)
    print(f"STARTING TABULAR MCCFR: {args.iterations} iterations")
    print(f"Parallel workers: {args.num_workers}")
    print("Algorithm: Monte Carlo Counterfactual Regret Minimization (TABULAR)")
    print("Strategy: Regret Matching (from accumulated regrets, no neural network)")
    print("="*80 + "\n")
    
    try:
        traversals_per_iteration = cfr_cfg.get("traversals_per_iteration", 100)
        save_interval = config.get("training", {}).get("save_interval_iterations", 10)
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        for iteration in range(1, args.iterations + 1):
            iter_start = time.time()
            
            logger.info(f"\n[ITERATION {iteration}/{args.iterations}] Starting MCCFR traversals")
            
            try:
                # Create tasks for workers (parallel game tree traversals)
                tasks = [
                    WorkerTask(
                        task_id=i,
                        game_state_hash=f"iter_{iteration}_task_{i}",
                        iteration=iteration,
                        num_traversals=traversals_per_iteration,
                        player_id=i % num_players,
                    )
                    for i in range(args.num_workers)
                ]
                
                # Run tasks in parallel via WorkerPool
                results = worker_pool.run_iteration(tasks, timeout_per_task=300.0)
                
                # Analyze regrets written to shared memory
                non_zero_count = regret_buffer.get_non_zero_count()
                
                iter_elapsed = time.time() - iter_start
                total_traversals = args.num_workers * traversals_per_iteration
                logger.info(f"  [SUMMARY] Iteration {iteration}: {total_traversals} traversals, {non_zero_count} infosets, {iter_elapsed:.1f}s")
                
                # Log to W&B if available
                if run is not None:
                    try:
                        run.log({
                            "iteration": iteration,
                            "traversals": total_traversals,
                            "regrets_written": non_zero_count,
                            "num_results": len(results),
                            "elapsed_time_s": iter_elapsed,
                            "algorithm": "Tabular-MCCFR",
                        })
                    except Exception as e:
                        logger.debug(f"W&B logging failed: {e}")
                
            except Exception as e:
                logger.error(f"MCCFR iteration {iteration} failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
            
            # Save checkpoint at intervals
            if iteration % save_interval == 0:
                try:
                    checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration:06d}.pt"
                    state = {
                        "iteration": iteration,
                        "regrets": regret_buffer.get_all_regrets(),
                        "config": config,
                        "algorithm": "Tabular-MCCFR",
                    }
                    torch.save(state, checkpoint_path)
                    logger.info(f"  [CHECKPOINT] Saved: {checkpoint_path}")
                    
                    # Attempt HF Hub sync (if configured)
                    if config.get("mlops", {}).get("huggingface", {}).get("enabled", False):
                        try:
                            from huggingface_hub import HfApi
                            api = HfApi()
                            repo_id = config.get("mlops", {}).get("huggingface", {}).get("repo_id", "")
                            if repo_id:
                                api.upload_file(
                                    path_or_fileobj=str(checkpoint_path),
                                    path_in_repo=checkpoint_path.name,
                                    repo_id=repo_id,
                                    repo_type="model",
                                )
                                logger.info(f"  [HF-HUB] Synced to {repo_id}")
                        except Exception as e:
                            logger.debug(f"HF Hub sync failed: {e}")
                
                except Exception as e:
                    logger.error(f"Checkpoint save failed at iteration {iteration}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("\n" + "="*80)
        print("TABULAR MCCFR TRAINING COMPLETED SUCCESSFULLY")
        print(f"Final regret information sets: {regret_buffer.get_non_zero_count()}")
        print("="*80 + "\n")
        
        # Close W&B run
        if run is not None:
            try:
                run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.debug(f"W&B finish failed: {e}")
        
        # Cleanup worker pool
        try:
            worker_pool.shutdown()
            logger.info("WorkerPool shut down")
        except Exception as e:
            logger.debug(f"WorkerPool shutdown failed: {e}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training stopped by user")
        try:
            worker_pool.shutdown()
        except:
            pass
        return 1
    except Exception as e:
        logger.error(f"MCCFR training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
