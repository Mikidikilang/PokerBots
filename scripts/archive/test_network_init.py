#!/usr/bin/env python3
"""Test NetworkConfig and MCCFR initialization."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.getLogger('src').setLevel(logging.WARNING)

print("[1] Importing NetworkConfig...", flush=True)
from src.model.networks import NetworkConfig, PokerActorCritic

print("[2] Creating NetworkConfig...", flush=True)
net_config = NetworkConfig()

print("[3] Creating PokerActorCritic network...", flush=True, end=' ')
try:
    network = PokerActorCritic(net_config)
    print("OK", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    sys.exit(1)

print("[4] Importing traversal engine...", flush=True)
from src.training.cfr_traversal import MCCFRTraversal
from src.training.cfr_infoset import InformationSetStorage
from src.env.wrappers import RLCardWrapper, WrapperConfig

print("[5] Creating environment and infoset storage...", flush=True)
env = RLCardWrapper()
infoset_storage = InformationSetStorage()

print("[6] Creating MCCFRTraversal...", flush=True, end=' ')
try:
    traversal = MCCFRTraversal(
        env=env,
        network=network,
        infoset_storage=infoset_storage,
    )
    print("OK", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[7] Resetting env and getting root state...", flush=True)
root_state = env.reset()

print("[8] Starting traversal...", flush=True, end=' ')
sys.stdout.flush()

# Add timeout protection
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Traversal took too long!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10 second timeout

try:
    value = traversal.external_sampling_traversal(
        state=root_state,
        player_to_update=0,
        reach_probs={0: 1.0, 1: 1.0},
        action_count=0,
    )
    signal.alarm(0)  # Cancel alarm
    print(f"Completed, value={value}", flush=True)
except TimeoutError as e:
    print(f"TIMEOUT: {e}", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    signal.alarm(0)  # Cancel alarm
    sys.exit(1)

print("\n[SUCCESS]")
