#!/usr/bin/env python3
"""
Smoke Test: API Credentials Validation
======================================

This script tests whether the API keys in .env are valid and
can establish connections to the respective services.

Run: python scripts/test_api_credentials.py
"""

import os
import sys
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install: pip install python-dotenv")
    sys.exit(1)

def test_wandb():
    """Test W&B API connectivity."""
    print("\n" + "="*70)
    print("TEST 1: Weights & Biases (W&B)")
    print("="*70)
    
    api_key = os.getenv("WANDB_API_KEY")
    
    if not api_key:
        print("❌ WANDB_API_KEY not found in .env")
        return False
    
    print(f"✓ API key found (length: {len(api_key)} chars)")
    print(f"  First 20 chars: {api_key[:20]}...")
    
    try:
        import wandb
        print(f"✓ wandb library available (v{wandb.__version__})")
        
        # Try to initialize (this validates the API key)
        try:
            wandb.login(key=api_key, relogin=True, force=True)
            print("✓ Successfully authenticated with W&B!")
            return True
        except Exception as e:
            print(f"❌ W&B authentication failed: {e}")
            return False
            
    except ImportError:
        print("⚠️  wandb not installed. Install: pip install wandb")
        return False

def test_huggingface():
    """Test Hugging Face API connectivity."""
    print("\n" + "="*70)
    print("TEST 2: Hugging Face (HF)")
    print("="*70)
    
    api_key = os.getenv("HF_TOKEN")
    
    if not api_key:
        print("❌ HF_TOKEN not found in .env")
        return False
    
    print(f"✓ API key found (length: {len(api_key)} chars)")
    print(f"  First 20 chars: {api_key[:20]}...")
    
    try:
        from huggingface_hub import HfApi, login
        print(f"✓ huggingface_hub library available")
        
        try:
            # Authenticate
            login(token=api_key)
            print("✓ Successfully authenticated with Hugging Face!")
            
            # Try to get user info (validates token)
            api = HfApi()
            user = api.whoami(token=api_key)
            print(f"✓ Authenticated as: {user.get('name', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ HF authentication failed: {e}")
            return False
            
    except ImportError:
        print("⚠️  huggingface_hub not installed. Install: pip install huggingface-hub")
        return False

def main():
    """Run all API credential tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  SMOKE TEST: API CREDENTIALS VALIDATION".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check if .env exists
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print(f"\n❌ .env file not found at: {env_path}")
        print(f"   Please create it with WANDB_API_KEY and HF_TOKEN")
        sys.exit(1)
    
    print(f"\n✓ Using .env file: {env_path}")
    
    results = {
        "W&B": test_wandb(),
        "Hugging Face": test_huggingface(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for service, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{service:20} : {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - API credentials are valid!")
        print("="*70 + "\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Check your API keys in .env")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
