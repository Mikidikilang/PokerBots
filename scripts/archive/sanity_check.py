#!/usr/bin/env python3
"""Sanity check - test if imports and initialization work."""

import sys
print("[1] Starting script...", flush=True)

sys.path.insert(0, "/home/user/poker_ai_v5")

print("[2] Importing modules...", flush=True)
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("[3] Setting up logging...", flush=True)
import logging
logging.getLogger('src.env.wrappers').setLevel(logging.WARNING)
logging.getLogger('src.env.action_mapper').setLevel(logging.WARNING)

print("[4] Importing env wrapper...", flush=True)
from src.env.wrappers import RLCardWrapper, WrapperConfig

print("[5] Creating env config...", flush=True)
config = WrapperConfig(num_players=2, initial_stack_bb=200.0)

print("[6] Creating RLCardWrapper...", flush=True)
env = RLCardWrapper(config)

print("[7] Resetting env...", flush=True)
state = env.reset()

print("[8] Checking state...", flush=True)
print(f"  hand={state['hand']}, board={state['public_cards']}")

print("\n[SUCCESS] All basic operations completed!")
