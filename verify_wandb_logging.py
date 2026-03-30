#!/usr/bin/env python3
"""
Quick verification script for W&B logging in the orchestrator.
Shows that W&B is successfully initialized and logging metrics.
"""

import wandb
import yaml
import time

print("\n" + "="*80)
print("W&B LOGGING VERIFICATION")
print("="*80 + "\n")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

wandb_cfg = config.get("mlops", {}).get("wandb", {})
project = wandb_cfg.get("project", "poker-mccfr-v5")
entity = wandb_cfg.get("entity")

print(f"[1] Configuration:")
print(f"    Project: {project}")
print(f"    Entity: {entity if entity else '(using logged-in user)'}")
print(f"    Enabled: {wandb_cfg.get('enabled', True)}")

# Initialize W&B
print(f"\n[2] Initializing W&B...")
try:
    init_params = {
        "project": project,
        "config": {"test": "verification_run", "algorithm": "Tabular-MCCFR"},
        "tags": ["verification", "logging-test"],
    }
    if entity:
        init_params["entity"] = entity
    
    run = wandb.init(**init_params)
    print(f"    ✓ W&B initialized")
    print(f"      Run ID: {run.id}")
    print(f"      Run URL: {run.url}")
except Exception as e:
    print(f"    ✗ W&B init failed: {e}")
    exit(1)

# Test logging metrics (simulating training loop)
print(f"\n[3] Testing metric logging...")
try:
    for i in range(1, 3):
        # Simulate training metrics
        metrics = {
            "iteration": i,
            "traversals": 4000,
            "regrets_written": 13000 + i*100,
            "elapsed_time_s": 30 + i*5,
            "algorithm": "Tabular-MCCFR",
        }
        
        run.log(metrics)
        print(f"    ✓ Logged iteration {i}: {metrics}")
        time.sleep(0.5)
    
    print(f"\n    ✓ All metrics logged successfully!")
except Exception as e:
    print(f"    ✗ Logging failed: {e}")
    run.finish()
    exit(1)

# Close run
print(f"\n[4] Finishing W&B run...")
try:
    run.finish()
    print(f"    ✓ W&B run completed")
except Exception as e:
    print(f"    ✗ Finish failed: {e}")
    exit(1)

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80 + "\n")
print("✓ W&B logging is working correctly!")
print(f"  View your run at: {run.url}")
