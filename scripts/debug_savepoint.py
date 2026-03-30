#!/usr/bin/env python3
"""Debug script to test if savepoint restore is working correctly."""

import sys
import torch
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to suppress annoying messages
logging.getLogger('src.env.wrappers').setLevel(logging.WARNING)
logging.getLogger('src.env.action_mapper').setLevel(logging.WARNING)
logging.getLogger('src.model.networks').setLevel(logging.WARNING)
logging.getLogger('src.env.sequential_history').setLevel(logging.WARNING)

from src.env.wrappers import RLCardWrapper, WrapperConfig
from src.training.cfr_env_state import EnvStateManager

print("=" * 80)
print("[TEST 1] Savepoint Restore - Simple Steps")
print("=" * 80)

# Create a simple environment
config = WrapperConfig(num_players=2, initial_stack_bb=200.0)
env = RLCardWrapper(config)
state_manager = EnvStateManager(env)

# Reset to get initial state
initial_state = env.reset()
print(f"Initial state: player=0, hand={initial_state['hand']}, board={initial_state['public_cards']}")

# Get legal actions
legal_actions = list(initial_state['legal_actions'].keys()) if hasattr(initial_state['legal_actions'], 'keys') else list(initial_state['legal_actions'])
print(f"Legal actions: {legal_actions}")

# Test 1: Take FOLD and then restore
print("\n[STEP 1] Taking FOLD action (action=0)...")
with state_manager.savepoint() as snap1:
    next_state1, reward1 = env.step(0)  # FOLD
    print(f"  After FOLD: hand={next_state1['hand']}, board={next_state1['public_cards']}")
    print(f"  env.is_over()={env.is_over()}")

print(f"[STEP 2] After savepoint context (state restored): env.is_over()={env.is_over()}")

# Test 2: Reset and check if we can take another action
print("\n[STEP 3] Attempting CHECK action (action=1)...")
try:
    next_state2, reward2 = env.step(1)  # CHECK
    print(f"  After CHECK: hand={next_state2['hand']}, board={next_state2['public_cards']}")
    print(f"  env.is_over()={env.is_over()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Full loop simulation
print("\n" + "=" * 80)
print("[TEST 2] Full Action Loop Simulation")
print("=" * 80)

env = RLCardWrapper(config)
state_manager = EnvStateManager(env)
initial_state = env.reset()
legal_actions = list(initial_state['legal_actions'].keys()) if hasattr(initial_state['legal_actions'], 'keys') else list(initial_state['legal_actions'])

print(f"Initial: hand={initial_state['hand']}, actions={legal_actions}")
print("\nSimulating loop through ALL actions with savepoints:")

action_values = {}
for i, action in enumerate(legal_actions):
    print(f"\n  Iteration {i}: action={action}")
    
    with state_manager.savepoint() as snap:
        try:
            next_state, reward = env.step(action)
            print(f"    Stepped: reward={reward}, is_over={env.is_over()}")
            action_values[action] = reward
        except Exception as e:
            print(f"    ERROR in step: {e}")
            action_values[action] = 0.0
    
    print(f"    After restore: env.is_over()={env.is_over()}")

print(f"\n[RESULTS] Completed {len(action_values)} actions with values:")
for a, v in sorted(action_values.items()):
    print(f"  action={a}: value={v}")

print("\n[TEST COMPLETE]")
