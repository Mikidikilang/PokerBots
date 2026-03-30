#!/usr/bin/env python3
"""Debug MCCFRTraversal to see what values are being computed."""

import sys
import torch
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Reduce logging noise
logging.getLogger('src.env.wrappers').setLevel(logging.WARNING)
logging.getLogger('src.env.action_mapper').setLevel(logging.WARNING)
logging.getLogger('src.model.networks').setLevel(logging.WARNING)
logging.getLogger('src.env.sequential_history').setLevel(logging.WARNING)
logging.getLogger('src.training.cfr_infoset').setLevel(logging.WARNING)

# Enable debug for traversal
logging.getLogger('src.training.cfr_traversal').setLevel(logging.DEBUG)

from src.env.wrappers import RLCardWrapper, WrapperConfig
from src.training.cfr_infoset import InformationSetStorage
from src.training.cfr_traversal import MCCFRTraversal
from src.model.networks import PokerActorCritic, NetworkConfig

print("=" * 80)
print("[MINIMAL TRAVERSAL TEST]")
print("=" * 80)

# Create environment
config = WrapperConfig(num_players=2, initial_stack_bb=200.0)
env = RLCardWrapper(config)

# Create minimal network
net_config = NetworkConfig()
network = PokerActorCritic(net_config)

# Create infoset storage
infoset_storage = InformationSetStorage()

# Create traversal engine
traversal = MCCFRTraversal(
    env=env,
    network=network,
    infoset_storage=infoset_storage,
)

# Reset environment
root_state = env.reset()
print(f"\nInitial state: hand={root_state['hand']}, board={root_state['public_cards']}")
print(f"Legal actions: {list(root_state['legal_actions'].keys()) if hasattr(root_state['legal_actions'], 'keys') else root_state['legal_actions']}")

# Run ONE traversal for player 0
print("\n[TRAVERSAL] Running external_sampling_traversal for player 0...")
value = traversal.external_sampling_traversal(
    state=root_state,
    player_to_update=0,
    reach_probs={0: 1.0, 1: 1.0},
    action_count=0,
)

print(f"\n[RESULT] Game value for player 0: {value}")

# Check what infosets were discovered
print(f"\n[INFOSETS] Discovered {len(infoset_storage.infosets)} infosets:")
for infoset_id, infoset_obj in list(infoset_storage.infosets.items())[:5]:
    cumulative = infoset_obj.cumulative_regret
    print(f"  {infoset_id[:8]}...: {len(cumulative)} actions with regrets")
    for action_idx, regret in list(cumulative.items())[:3]:
        print(f"    action {action_idx}: regret={regret:.4f}")

print("\n[TEST COMPLETE]")
