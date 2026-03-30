#!/usr/bin/env python3
"""Test traversal without signal-based timeout."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.getLogger('src').setLevel(logging.WARNING)
logging.getLogger('src.training.cfr_traversal').setLevel(logging.DEBUG)

print("[1] Importing modules...", flush=True)
from src.model.networks import NetworkConfig, PokerActorCritic
from src.training.cfr_traversal import MCCFRTraversal
from src.training.cfr_infoset import InformationSetStorage
from src.env.wrappers import RLCardWrapper, WrapperConfig

print("[2] Creating environment and network...", flush=True)
env = RLCardWrapper(WrapperConfig(num_players=2))
network = PokerActorCritic(NetworkConfig())
infoset_storage = InformationSetStorage()

print("[3] Creating MCCFRTraversal...", flush=True)
traversal = MCCFRTraversal(
    env=env,
    network=network,
    infoset_storage=infoset_storage,
)

print("[4] Resetting env...", flush=True)
root_state = env.reset()

print("[5] Starting traversal (may take a while)...", flush=True)
import time
start_time = time.time()

try:
    value = traversal.external_sampling_traversal(
        state=root_state,
        player_to_update=0,
        reach_probs={0: 1.0, 1: 1.0},
        action_count=0,
    )
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Traversal completed in {elapsed:.2f}s, value={value}", flush=True)
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[ERROR] After {elapsed:.2f}s: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
