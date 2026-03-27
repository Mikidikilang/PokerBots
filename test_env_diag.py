#!/usr/bin/env python3
"""
Quick diagnostic script to test environment wrapper.
"""
import yaml
from src.env.wrappers import make_env

# Load config
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=== Environment Diagnostic ===\n")

# Create environment
env = make_env(cfg)
print(f"✓ Environment created: {type(env).__name__}\n")

# Reset and check obs
obs = env.reset()
print(f"✓ reset() returned: {type(obs).__name__}")
print(f"  Keys in obs: {list(obs.keys())}\n")

# Check for 'hand' key
if 'hand' in obs:
    print(f"✓ 'hand' key present: {obs['hand']}")
else:
    print(f"✗ 'hand' key MISSING!")
    print(f"  Available keys: {list(obs.keys())}")
    print(f"  Full obs structure: {obs}")

# Try a step
print(f"\nPerforming one step...")
action = 1  # check/call
next_obs, reward = env.step(action)
print(f"✓ step({action}) returned: obs={type(next_obs).__name__}, reward={reward}")
print(f"  Keys in next_obs: {list(next_obs.keys())}")

if 'hand' in next_obs:
    print(f"✓ 'hand' key present in next_obs: {next_obs['hand']}")
else:
    print(f"✗ 'hand' key MISSING in next_obs!")
