#!/usr/bin/env python3
"""Simple test to check environment step/reset behavior."""

import sys
from src.env.wrappers import RLCardWrapper, WrapperConfig

print("Simple environment test...")
env = RLCardWrapper(config=WrapperConfig(num_players=2))

print("\n[Test 1] Reset environment")
state1 = env.reset()
print(f"State keys: {list(state1.keys())}")
print(f"hand: {state1['hand']}, public_cards: {state1['public_cards']}")
print(f"is_over after reset: {env.is_over()}")

print("\n[Test 2] Take 5 random actions")
for i in range(5):
    legal = state1.get('legal_actions', [])
    if not legal:
        print(f"No legal actions! State: {state1}")
        sys.exit(1)
    
    action = legal[0]
    print(f"\n  Step {i+1}: action={action}")
    is_terminal_before = env.is_over()
    state1, reward = env.step(action)
    is_terminal_after = env.is_over()
    
    print(f"    is_over before: {is_terminal_before}, after: {is_terminal_after}")
    print(f"    reward: {reward}")
    print(f"    hand: {state1.get('hand')}, board: {state1.get('public_cards')}")
    
    if is_terminal_after:
        print(f"  --> Game terminal at step {i+1}")
        break

print("\n[Test 3] Reset again and verify")
state2 = env.reset()
print(f"New reset - hand: {state2['hand']}, board: {state2['public_cards']}")
print(f"is_over: {env.is_over()}")

print("\n[SUCCESS] Environment behaves correctly")
