#!/usr/bin/env python3
"""
Test that mimics the training pipeline to identify the issue.
"""
import yaml
import torch
from src.env.wrappers import make_env
from src.env.features import ObservationBuilder, ObservationConfig

# Load config
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=== Training Pipeline Diagnostic ===\n")

# Create environment
env = make_env(cfg)
print("OK: Environment created")

# Create ObservationBuilder
num_players = cfg["environment"]["num_players"]
obs_config = ObservationConfig(num_players=num_players)
obs_builder = ObservationBuilder(obs_config)
print(f"OK: ObservationBuilder created (dim={obs_builder.get_observation_dim()})\n")

# Reset environment
print("Calling env.reset()...")
obs_dict = env.reset()
print(f"OK: reset() returned: {type(obs_dict).__name__}")
print(f"  Keys: {list(obs_dict.keys())}")
print(f"  'hand' value: {obs_dict.get('hand', 'MISSING')}\n")

# Now try to build observation
print("Calling obs_builder.build(obs_dict)...")
try:
    obs_tensor = obs_builder.build(obs_dict)
    print("OK: build() succeeded")
    print(f"  Output type: {type(obs_tensor).__name__}")
    print(f"  Keys: {list(obs_tensor.keys())}")
except Exception as e:
    print(f"ERROR: build() FAILED: {e}")
    print(f"  Exception type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# Try a few more steps
print("\nTrying a few more steps...")
for i in range(3):
    print(f"  Step {i+1}: ", end="")
    action = 1  # check/call
    next_obs, reward = env.step(action)
    print(f"action={action}, done={env.is_over()}", end=" -> ")
    try:
        obs_tensor = obs_builder.build(next_obs)
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        break
    if env.is_over():
        print("Hand ended, resetting...")
        obs_dict = env.reset()
        print(f"  Reset: {list(obs_dict.keys())}")
