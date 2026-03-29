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
            
            # Extract legal actions from observation if available
            # Otherwise fall back to all actions (12 for heads-up, 9 for 6-max preflop)
            legal_actions = list(range(12))  # Default: all 12 legal actions
            if "action_mask" in obs_dicts:
                mask = obs_dicts["action_mask"][i]
                # action_mask is binary [12]: 1.0 = legal, 0.0 = illegal
                legal_actions = torch.nonzero(mask == 1.0, as_tuple=False).squeeze(-1).tolist()
                if not legal_actions:  # Safety: if no legal actions detected, allow all
                    logger.warning(f"No legal actions in action_mask at batch index {i}, using all")
                    legal_actions = list(range(12))
            elif "legal_actions" in batch:
                # If batch dict has a legal_actions field directly
                legal_actions = batch["legal_actions"][i]
                if isinstance(legal_actions, torch.Tensor):
                    legal_actions = legal_actions.tolist()
            
            # Convert single obs_flat entry back to tensor for CFREngine
            # [PHASE 2.5B] IMPORTANT: Keep obs as dict, not flattened tensor!
            # CFREngine.network expects dict{hole_cards, community_cards, env_metrics, betting_history, position, action_mask}
            
            trajectory = {
                "states": [obs_single],  # Keep as dict, NOT flattened!
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
        
        Phase 2.5 Simplified (no action history):
            Use hash(player, hero_cards, board_cards, action_count)
            where action_count is approximated from betting_history length
        
        Args:
            obs_dicts: Observation dict at batch index idx
            idx: Batch index
        
        Returns:
            Unique infoset identifier string
        """
        from src.training.cfr_infoset import hash_infoset
        
        # Extract actual cards from observation tensors (FIXED from hardcoded "A", "K")
        hero_cards = self._extract_cards(obs_dicts, idx, "hero")
        board_cards = self._extract_cards(obs_dicts, idx, "board")
        
        # Approximate action history from betting_history tensor
        # Each action takes 13 dims (if using extended history)
        # So action_count ≈ num_nonzero / 13
        action_history = ()
        if "betting_history" in obs_dicts:
            betting_hist_tensor = obs_dicts["betting_history"][idx].flatten()
            action_count = int(betting_hist_tensor.nonzero().shape[0] / 13)
            action_history = tuple(str(i) for i in range(action_count))  # String actions
        
        return hash_infoset(
            player=0,  # Always hero in our setup (training poker player)
            hole_cards=hero_cards,
            board_cards=board_cards,
            action_history=action_history,
        )
    
    def _decode_card_tensor(self, card_tensor: torch.Tensor) -> tuple[str, ...]:
        """
        Decode 52-dim multi-hot card tensor to card string tuple.
        
        Card encoding (from features.py._encode_cards):
            card_index = rank_idx * 4 + suit_idx
            where rank_idx ∈ [0,12] (2-A), suit_idx ∈ [0,3] (S,H,D,C)
        
        Args:
            card_tensor: Shape [52] with 1.0 at card indices, 0.0 elsewhere
        
        Returns:
            Tuple of card strings, e.g., ("As", "Kd")
        """
        RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        SUIT_NAMES = ["S", "H", "D", "C"]
        
        # Find all indices where card_tensor == 1.0
        card_tensor = card_tensor.flatten()
        indices = torch.nonzero(card_tensor == 1.0, as_tuple=False).squeeze(-1)
        
        # Handle scalar output from nonzero
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        
        cards = []
        for idx in indices.tolist():
            rank_idx = idx // 4
            suit_idx = idx % 4
            
            if 0 <= rank_idx < 13 and 0 <= suit_idx < 4:
                card_str = SUIT_NAMES[suit_idx] + RANK_NAMES[rank_idx]
                cards.append(card_str)
        
        return tuple(cards)
    
    def _extract_cards(
        self,
        obs_dicts: dict[str, torch.Tensor],
        idx: int,
        card_type: str,  # "hero", "board"
    ) -> tuple[str, ...]:
        """
        Extract card information from observation dict.
        
        Uses multi-hot card encoding from features.py:
            - hole_cards: 52-dim vector for hero's 2 hole cards
            - community_cards: 52-dim vector for 0-5 board cards
        
        Args:
            obs_dicts: Observation dict with "hole_cards", "community_cards" tensors
            idx: Batch index
            card_type: "hero" (hole cards) or "board" (community cards)
        
        Returns:
            Tuple of card strings, e.g., ("As", "Kd")
        """
        try:
            if card_type == "hero":
                tensor = obs_dicts["hole_cards"][idx]
            elif card_type == "board":
                tensor = obs_dicts["community_cards"][idx]
            else:
                logger.warning(f"Unknown card_type: {card_type}, returning empty tuple")
                return ()
            
            return self._decode_card_tensor(tensor)
        
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to extract {card_type} cards: {e}, returning empty tuple")
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
