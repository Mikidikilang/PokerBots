#!/usr/bin/env python3
"""Minimal test of MCCFR traversal to find recursion issue."""

import sys
import traceback
from src.env.wrappers import RLCardWrapper, WrapperConfig
from src.training.cfr_traversal import MCCFRTraversal
from src.training.parallel_cfr import InformationSetStorage
import torch

print("Minimal MCCFR traversal test...")

try:
    # Initialize components
    config = WrapperConfig(num_players=2)
    env = RLCardWrapper(config=config)
    
    # Create a simple network
    network = torch.nn.Identity()
    device = torch.device('cpu')
    
    # Create infoset storage
    infoset_storage = InformationSetStorage()
    
    # Create MCCFR instance
    mccfr = MCCFRTraversal(
        env=env,
        network=network,
        infoset_storage=infoset_storage,
        device=device,
    )
    
    print("[OK] MCCFR engine created")
    
    # Try one traversal
    print("[TEST] Starting traversal for player 0...")
    root_state = env.reset()
    print(f"  Root state: hand={root_state['hand']}, board={root_state['public_cards']}")
    
    try:
        value = mccfr.external_sampling_traversal(
            state=root_state,
            player_to_update=0,
            reach_probs={0: 1.0, 1: 1.0},
            action_count=0,
        )
        print(f"[OK] Traversal completed! Value={value}")
    
    except RecursionError as e:
        print(f"[ERROR] RecursionError: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
        
except Exception as e:
    print(f"[FATAL] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] Minimal traversal test passed")
