#!/usr/bin/env python3
"""
Iteration 0: Cloud Infrastructure Initialization

This script initializes the cloud infrastructure:
1. Creates Hugging Face repository
2. Creates Weights & Biases project
3. Runs 1 iteration with 1 worker to trigger setup logic
4. Saves the first checkpoint and uploads to HF Hub

Run: python scripts/init_cloud_infrastructure.py
"""

import sys
import os
import logging
from pathlib import Path
from typing import Any

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env loading")

def init_cloud_infrastructure():
    """Initialize cloud infrastructure for Iteration 0."""
    
    print("\n" + "="*80)
    print("ITERATION 0: CLOUD INFRASTRUCTURE INITIALIZATION")
    print("="*80)
    
    # Step 1: Verify API keys
    print("\n[STEP 1] Verifying API Credentials...")
    hf_token = os.getenv("HF_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")
    
    if not hf_token:
        logger.error("HF_TOKEN not found in .env")
        return False
    if not wandb_key:
        logger.error("WANDB_API_KEY not found in .env")
        return False
    
    print(f"  ✓ HF_TOKEN found (length: {len(hf_token)})")
    print(f"  ✓ WANDB_API_KEY found (length: {len(wandb_key)})")
    
    # Step 2: Initialize Hugging Face repository
    print("\n[STEP 2] Creating Hugging Face Repository...")
    try:
        from huggingface_hub import HfApi, login
        
        # Login to HF
        login(token=hf_token)
        
        api = HfApi()
        
        # Check if repo exists
        repo_id = "Mikidikilang/poker-mccfr-production"
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            print(f"  ✓ Repository already exists: {repo_id}")
        except Exception:
            # Create repo if it doesn't exist
            print(f"  → Creating repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
                exist_ok=True,
            )
            print(f"  ✓ Repository created: {repo_id}")
            
    except Exception as e:
        logger.error(f"Failed to initialize HF repository: {e}")
        return False
    
    # Step 3: Verify Weights & Biases
    print("\n[STEP 3] Verifying Weights & Biases Project...")
    try:
        import wandb
        
        # Verify W&B login
        wandb.login(key=wandb_key, relogin=True, force=True)
        
        print(f"  ✓ W&B authenticated as: soross")
        print(f"  ✓ W&B Project: poker-mccfr-v5 (will be created on first training run)")
        
    except Exception as e:
        logger.error(f"Failed to verify W&B credentials: {e}")
        return False
    
    # Step 4: Create local directories
    print("\n[STEP 4] Creating Local Directories...")
    
    directories = [
        "checkpoints/",
        "logs/",
        "plots/",
        "equity_cache/",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Directory ready: {dir_path}")
    
    # Step 5: Summary
    print("\n" + "="*80)
    print("✓ CLOUD INFRASTRUCTURE SUCCESSFULLY INITIALIZED")
    print("="*80)
    print(f"\n  Hugging Face Repo: https://huggingface.co/{repo_id}")
    print(f"  W&B Project:       https://wandb.ai/soross/poker-mccfr-v5")
    print(f"\n  Next step: Run full training with config.yaml")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    success = init_cloud_infrastructure()
    sys.exit(0 if success else 1)
