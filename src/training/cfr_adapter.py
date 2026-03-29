"""
CFR Trajectory Adapter (cfr_adapter.py).

[PHASE 2] Data format conversion from PPO (RolloutBuffer) → CFR (CFREngine).

PROBLEM: Existing PPO infrastructure (collector, buffer) produces mini-batch dicts
optimized for PPO training (advantages, log_probs, returns). CFR needs a different
format (trajectories, infoset IDs, per-action counterfactual values).

SOLUTION: Adapter module
    - Input: mini-batch dict from buffer.get_mini_batches()
    - Output: list of trajectory dicts compatible with cfr_engine.train_on_rollouts()
    - Side effect: populates InformationSetStorage with discovered infosets

KEY INSIGHT: We don't need to change collector/buffer. We adapt their output
at the trainer level, allowing smooth transition from PPOTrainer → CFREngine.

---

References:
    - PPO buffer: src/training/buffer.py
    - CFR engine: src/training/cfr_engine.py
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CFRTrajectoryAdapter:
    """
    Converts mini-batches from PPO buffer to CFR trajectory format.
    
    Handles:
        - Flattening obs_dict to single tensor
        - Extracting infoset IDs from observation keys
        - Breaking batch-of-timesteps into episode trajectories
        - Populating InformationSetStorage
    """
    
    def __init__(self, infoset_storage=None):
        """
        Args:
            infoset_storage: Optional InformationSetStorage instance to populate
        """
        self.infoset_storage = infoset_storage
        self.obs_keys_seen = set()
    
    def batch_to_cfr_trajectories(
        self,
        batch: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Convert a single PPO mini-batch to list of CFR trajectories.
        
        [IMPORTANT] PPO mini-batches are already shuffled collections of
        individual timesteps from potentially multiple episodes. For CFR,
        we need to group these back into episode-wise trajectories for
        proper counterfactual value computation.
        
        This function is a COMPROMISE for Phase 2:
            - We treat each timestep as a 1-step "trajectory"
            - This underutilizes CFR's ability to use multi-step lookahead
            - BUT it allows immediate integration with existing buffer
        
        Phase 3 TODO: Modify collector to preserve trajectory boundaries
                      (episode start/end markers) through buffer to enable
                      proper multi-step counterfactual computation.
        
        Args:
            batch: Dict from buffer.get_mini_batches():
                {
                    "observations": {key: tensor[batch_size, ...]},
                    "actions": tensor[batch_size],
                    "returns": tensor[batch_size],
                    ... (advantages, old_log_probs, old_values not used)
                }
        
        Returns:
            List[dict] where each dict is a single-step trajectory:
            [
                {
                    "states": [obs_flattened],  # Single-element list
                    "actions": [action_int],
                    "infoset_ids": [infoset_hash],
                    "legal_actions_per_node": [legal_actions],
                    "reward": float,  # final_reward (returns[i])
                }
                ...
            ]
        """
        trajectories = []
        obs_dicts = batch["observations"]
        actions = batch["actions"]
        returns = batch["returns"]  # Returns, not advantages (CFR uses real outcomes)
        
        batch_size = actions.shape[0]
        
        # Flatten observation dict to ensure we can work with it
        obs_flat = self._flatten_obs_dict(obs_dicts)
        
        for i in range(batch_size):
            # Extract single timestep
            obs_single = {k: v[i:i+1] for k, v in obs_flat.items()}
            action_int = int(actions[i].item())
            reward = float(returns[i].item())
            
            # Generate infoset ID from observation
            infoset_id = self._generate_infoset_id(obs_dicts, i)
            
            # Infer legal actions (all for now; better: extract from obs)
            # TODO: Store legal actions in buffer or obs to avoid this
            legal_actions = list(range(12))  # All 12 actions (heads-up)
            
            # Convert single obs_flat entry back to tensor for CFREngine
            obs_tensor = self._obs_dict_to_tensor(obs_single)
            
            trajectory = {
                "states": [obs_tensor],  # Single state (1-step trajectory)
                "actions": [action_int],
                "infoset_ids": [infoset_id],
                "legal_actions_per_node": [legal_actions],
                "reward": reward,
            }
            
            trajectories.append(trajectory)
            
            # Register infoset in storage if provided
            if self.infoset_storage:
                # Extract game info from observation
                hero_cards = self._extract_cards(obs_dicts, i, "hero")
                board_cards = self._extract_cards(obs_dicts, i, "board")
                # TODO: Extract action_history from obs
                
                self.infoset_storage.get_or_create_infoset(
                    player=0,  # Always hero in our setup
                    hole_cards=hero_cards,
                    board_cards=board_cards,
                    action_history=(),  # TODO: Extract from obs
                )
        
        return trajectories
    
    def _flatten_obs_dict(self, obs_dicts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Ensure all obs tensors are properly shaped."""
        flattened = {}
        for key, tensor in obs_dicts.items():
            if tensor.dim() > 1 and tensor.shape[0] == 1:
                # Remove batch dimension if present
                flattened[key] = tensor
            else:
                flattened[key] = tensor
            
            self.obs_keys_seen.add(key)
        
        return flattened
    
    def _generate_infoset_id(self, obs_dicts: dict[str, torch.Tensor], idx: int) -> str:
        """
        Generate infoset ID from observation at batch index idx.
        
        Infoset = hash of (player, hero_cards, board_cards, action_history)
        
        For now: use simplified hash of available obs keys.
        TODO: Extract actual card information and action history.
        """
        # Simple approach: hash the concatenated features
        # In production, extract actual cards from obs_dicts["hero_cards"] etc.
        from src.training.cfr_infoset import hash_infoset
        
        # Placeholder: extract from obs (requires knowledge of obs structure)
        hero_cards = ("A", "K")  # TODO: Parse from obs_dicts
        board_cards = ()  # TODO: Parse from obs_dicts
        action_history = ()  # TODO: Parse from obs_dicts
        
        return hash_infoset(
            player=0,
            hole_cards=hero_cards,
            board_cards=board_cards,
            action_history=action_history,
        )
    
    def _extract_cards(
        self,
        obs_dicts: dict[str, torch.Tensor],
        idx: int,
        card_type: str,  # "hero", "board", etc.
    ) -> tuple[str, ...]:
        """
        Extract card information from observation dict.
        
        TODO: This requires understanding the exact encoding of cards in obs.
        Current structure (from features.py) encodes cards as:
            - Integer indices (0-51 in standard 52-card deck)
            - Normalized to [0,1] range
        
        For CFR, we need the STRING representation (e.g., "A♠", "K♦").
        
        Args:
            obs_dicts: All observation tensors
            idx: Batch index
            card_type: Which cards to extract ("hero", "board", "opponent")
        
        Returns:
            Tuple of card strings, e.g., ("A♠", "K♦")
        """
        # Placeholder: return dummy cards
        if card_type == "hero":
            return ("A", "K")
        elif card_type == "board":
            return ()
        else:
            return ()
    
    def _obs_dict_to_tensor(
        self,
        obs_single: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert single observation dict to flat tensor.
        
        CFREngine expects: obs_tensor of shape [obs_dim]
        
        Args:
            obs_single: Dict with single-entry tensors (shape [..., 1] or [...])
        
        Returns:
            Flattened tensor [obs_dim]
        """
        # Concatenate all observation components
        tensors = []
        
        # Standard order (from features.py):
        # 1. hero_cards (encoded)
        # 2. board_cards (encoded)
        # 3. opponent_cards (if known)
        # 4. env_metrics (pot_odds, stacks, etc.)
        
        for key in sorted(obs_single.keys()):  # Deterministic order
            tensor = obs_single[key]
            
            # Flatten and move to CPU if needed
            if tensor.dim() > 1:
                tensor = tensor.flatten()
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            
            tensors.append(tensor)
        
        # Concatenate all into single tensor
        if tensors:
            flat = torch.cat(tensors, dim=0)
            return flat
        else:
            logger.warning("Empty observation dict, returning zero tensor")
            return torch.tensor([], dtype=torch.float32)
    
    def get_summary(self) -> dict[str, any]:
        """Return statistics about processed batches."""
        return {
            "obs_keys_seen": sorted(self.obs_keys_seen),
            "total_obs_key_types": len(self.obs_keys_seen),
        }


class CFRIntegrationBridge:
    """
    [PHASE 2.5] Wrapper that replaces PPOTrainer.train_on_buffer() with CFR.
    
    Usage in runner.py:
        >>> bridge = CFRIntegrationBridge(cfr_engine, infoset_storage)
        >>> stats = bridge.train_on_buffer(buffer)
    
    This maintains the same interface as PPOTrainer, allowing minimal changes
    to runner.py.
    """
    
    def __init__(self, cfr_engine, infoset_storage=None):
        """
        Args:
            cfr_engine: CFREngine instance
            infoset_storage: InformationSetStorage (usually cfr_engine.infoset_storage)
        """
        self.cfr_engine = cfr_engine
        self.infoset_storage = infoset_storage or cfr_engine.infoset_storage
        self.adapter = CFRTrajectoryAdapter(self.infoset_storage)
    
    def train_on_buffer(self, buffer: Any) -> dict[str, float]:
        """
        Bridge method: adapts PPO buffer interface to CFR engine interface.
        
        Steps:
            1. Compute GAE (same as PPO)
            2. Get mini-batches from buffer
            3. Convert each batch to CFR trajectories
            4. Call cfr_engine.train_on_rollouts()
            5. Aggregate stats
        
        Args:
            buffer: RolloutBuffer instance (from PPO pipeline)
        
        Returns:
            Stats dict in CFR format (matches cfr_engine.train_on_rollouts())
        """
        # PPO still computes GAE (used for value targets in CFR)
        buffer.compute_gae()
        
        all_stats = {
            "cfr_loss": 0.0,
            "avg_regret": 0.0,
            "num_infosets": 0,
            "num_batches": 0,
        }
        
        batch_count = 0
        
        # Process each mini-batch
        for batch in buffer.get_mini_batches():
            # Convert PPO batch format → CFR trajectory format
            trajectories = self.adapter.batch_to_cfr_trajectories(batch)
            
            # Wrap as rollouts (CFREngine expects list of {trajectory, reward})
            rollouts = [
                {
                    "trajectory": traj,
                    "reward": traj["reward"],
                }
                for traj in trajectories
            ]
            
            # Train CFR on this batch
            batch_stats = self.cfr_engine.train_on_rollouts(rollouts)
            
            # Accumulate stats
            all_stats["cfr_loss"] += batch_stats.get("cfr_loss", 0.0)
            all_stats["avg_regret"] += batch_stats.get("avg_regret", 0.0)
            all_stats["num_infosets"] = max(
                all_stats["num_infosets"],
                batch_stats.get("num_infosets", 0),
            )
            batch_count += 1
        
        # Normalize by batch count
        if batch_count > 0:
            all_stats["cfr_loss"] /= batch_count
            all_stats["avg_regret"] /= batch_count
        
        all_stats["num_batches"] = batch_count
        
        logger.info(
            "[CFR Bridge] Processed %d batches: loss=%.4f, regret=%.4f, infosets=%d",
            batch_count,
            all_stats["cfr_loss"],
            all_stats["avg_regret"],
            all_stats["num_infosets"],
        )
        
        return all_stats
