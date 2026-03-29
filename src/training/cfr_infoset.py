"""
Information Set Management (cfr_infoset.py).

[PHASE 2] Game state abstraction and regret persistence.

WHAT IS AN INFORMATION SET?
-----------------------------

In imperfect information games, players don't see all cards. An INFOSET abstracts
all game histories that are indistinguishable to the current player.

Example (Texas Hold'em heads-up):
    Player sees: hole cards {A♠ K♠}, board {Q♣ J♦ T♠}
    Action history: [check, raise 3BB, call]
    
    There might be 1000+ different opponent hole card combinations that lead
    to identical information from this player's perspective.
    
    All 1000 histories = SAME INFOSET for this player
    
This allows:
    - Compact strategy representation (one policy per infoset, not per history)
    - Regret matching across similar situations
    - Generalization via neural networks

INFOSET HASHING
----------------

We identify infosets by hashing:
    hash(player, hole_cards, board_state, action_history)

This represents: "What does this player see and what has happened so far?"

Benefits:
    - Fast lookup
    - Collision-resistant (unlikely same hash = same infoset)
    - Can be persisted to disk for multi-session training

REGRET ACCUMULATION
--------------------

For each infoset, we track:
    cumulative_regret[action_a] = Σ_t regret^t(action_a)
    
Where regret^t(action_a) = (value from taking a) - (value from actual action)

Over sufficient iterations:
    regret_matched_strategy(action_a) = max(cumulative_regret[a], 0) / Σ_a' max(regret[a'], 0)
    
This strategy converges to Nash equilibrium.

---

References
    - Lanctot et al. (2009): "An Introduction to Counterfactual Regret Minimization"
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InformationSet:
    """
    Game state abstraction: one infoset = all histories indistinguishable to a player.
    
    Stores:
        - Player's visible information (cards, board, history)
        - Regret for each action
        - Iteration count (for averaging)
    """
    
    infoset_id: str                          # Hash of (player, cards, board, history)
    player: int                              # 0 = hero, 1 = opponent (heads-up)
    hole_cards: tuple[str, str]              # e.g., ('A', 'K')
    board_cards: tuple[str, ...]             # e.g., ('Q', 'J', 'T') or ()
    action_history: tuple[str, ...]          # e.g., ('check', 'raise')
    
    # Regret tracking
    cumulative_regret: dict[int, float] = field(default_factory=dict)
    regret_sum_squared: dict[int, float] = field(default_factory=dict)
    action_counts: dict[int, int] = field(default_factory=dict)
    visit_count: int = 0
    
    # For strategy computation
    positive_regret_sum: dict[int, float] = field(default_factory=dict)
    last_strategy: dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure all action dicts have consistent keys."""
        if not self.cumulative_regret:
            logger.debug(f"Created new infoset: {self.infoset_id}")
    
    def add_regret(self, action: int, regret_value: float):
        """
        Accumulate regret for an action.
        
        Args:
            action: Action index (0-11 in poker)
            regret_value: Counterfactual regret (can be negative)
        """
        if action not in self.cumulative_regret:
            self.cumulative_regret[action] = 0.0
            self.regret_sum_squared[action] = 0.0
            self.action_counts[action] = 0
        
        self.cumulative_regret[action] += regret_value
        self.regret_sum_squared[action] += regret_value ** 2
        self.action_counts[action] += 1
        self.visit_count += 1
    
    def get_strategy(self, legal_actions: Optional[list[int]] = None) -> dict[int, float]:
        """
        Compute current regret-matched strategy via regret matching formula.
        
        Formula:
            σ_t(a|h) = max(R^t(a|h), 0) / Σ_a' max(R^t(a'|h), 0)
        
        If sum of positive regrets = 0 (untrained action), use uniform.
        
        Args:
            legal_actions: If provided, zero out probability for illegal actions
        
        Returns:
            {action_idx: probability}
        """
        if not legal_actions:
            legal_actions = list(self.cumulative_regret.keys())
        
        # Compute positive regrets
        positive_regrets = {
            action: max(self.cumulative_regret.get(action, 0.0), 0.0)
            for action in legal_actions
        }
        
        total_positive_regret = sum(positive_regrets.values())
        
        if total_positive_regret <= 1e-8:
            # No action has positive regret → use uniform strategy
            num_actions = len(legal_actions)
            strategy = {action: 1.0 / num_actions for action in legal_actions}
        else:
            # Normalize positive regrets
            strategy = {
                action: positive_regrets[action] / total_positive_regret
                for action in legal_actions
            }
        
        self.last_strategy = strategy
        return strategy
    
    def get_regret_stats(self) -> dict[str, float]:
        """
        Summary statistics for this infoset's regret evolution.
        
        Returns:
            {
                'mean_regret': average regret across actions,
                'max_regret': maximum regret,
                'regret_variance': variance of regrets,
                'visit_count': number of times visited,
            }
        """
        if not self.cumulative_regret:
            return {'mean_regret': 0.0, 'max_regret': 0.0, 'regret_variance': 0.0, 'visit_count': 0}
        
        regrets = list(self.cumulative_regret.values())
        mean_regret = sum(regrets) / len(regrets)
        max_regret = max(regrets)
        
        variance = sum((r - mean_regret) ** 2 for r in regrets) / len(regrets)
        
        return {
            'mean_regret': mean_regret,
            'max_regret': max_regret,
            'regret_variance': variance,
            'visit_count': self.visit_count,
        }


class InformationSetStorage:
    """
    Central repository for all infosets encountered during training.
    
    Provides:
        - O(1) lookup by infoset hash
        - Batch strategy queries
        - Regret persistence (save/load between training runs)
    """
    
    def __init__(self):
        self.infosets: dict[str, InformationSet] = {}
        self.created_count = 0
        self.updated_count = 0
    
    def get_or_create_infoset(
        self,
        player: int,
        hole_cards: tuple[str, str],
        board_cards: tuple[str, ...],
        action_history: tuple[str, ...],
    ) -> InformationSet:
        """
        Get existing infoset or create new one.
        
        Args:
            player: 0 or 1 (heads-up)
            hole_cards: Tuple of two card strings
            board_cards: Tuple of 0-5 card strings
            action_history: Tuple of action names
        
        Returns:
            InformationSet object
        """
        infoset_id = hash_infoset(player, hole_cards, board_cards, action_history)
        
        if infoset_id not in self.infosets:
            self.infosets[infoset_id] = InformationSet(
                infoset_id=infoset_id,
                player=player,
                hole_cards=hole_cards,
                board_cards=board_cards,
                action_history=action_history,
            )
            self.created_count += 1
        
        return self.infosets[infoset_id]
    
    def get_infoset(self, infoset_id: str) -> Optional[InformationSet]:
        """Lookup infoset by ID."""
        return self.infosets.get(infoset_id)
    
    def add_regret(
        self,
        infoset_id: str,
        action: int,
        regret_value: float,
    ):
        """Add regret to existing infoset (must exist)."""
        if infoset_id in self.infosets:
            self.infosets[infoset_id].add_regret(action, regret_value)
            self.updated_count += 1
        else:
            logger.warning(f"Infoset {infoset_id} not found when adding regret")
    
    def get_strategy_batch(
        self,
        infoset_ids: list[str],
        legal_actions_batch: list[list[int]],
    ) -> list[dict[int, float]]:
        """
        Get strategies for multiple infosets.
        
        Args:
            infoset_ids: List of infoset hashes
            legal_actions_batch: Corresponding legal action lists
        
        Returns:
            List of strategy dicts
        """
        strategies = []
        
        for infoset_id, legal_actions in zip(infoset_ids, legal_actions_batch):
            infoset = self.get_infoset(infoset_id)
            
            if infoset:
                strategy = infoset.get_strategy(legal_actions)
            else:
                # Unseen infoset → uniform strategy
                num_actions = len(legal_actions)
                strategy = {a: 1.0 / num_actions for a in legal_actions}
            
            strategies.append(strategy)
        
        return strategies
    
    def get_summary(self) -> dict[str, any]:
        """Statistics about storage."""
        if not self.infosets:
            return {
                'total_infosets': 0,
                'created_this_session': 0,
                'updated_this_session': 0,
            }
        
        regrets = [
            regret_value
            for infoset in self.infosets.values()
            for regret_value in infoset.cumulative_regret.values()
        ]
        
        return {
            'total_infosets': len(self.infosets),
            'created_this_session': self.created_count,
            'updated_this_session': self.updated_count,
            'mean_cumulative_regret': sum(regrets) / len(regrets) if regrets else 0.0,
            'max_cumulative_regret': max(regrets) if regrets else 0.0,
        }


def hash_infoset(
    player: int,
    hole_cards: tuple[str, str],
    board_cards: tuple[str, ...],
    action_history: tuple[str, ...],
) -> str:
    """
    Create stable hash for infoset identification.
    
    Hash should be:
        - Deterministic (same input → same hash)
        - Collision-resistant (different infosets → different hashes)
        - Fast to compute
    
    Args:
        player: 0 or 1
        hole_cards: Tuple of 2 card strings
        board_cards: Tuple of 0-5 card strings
        action_history: Tuple of actions taken
    
    Returns:
        40-character hex string
    """
    # Construct canonical string representation
    parts = [
        str(player),
        '|'.join(hole_cards),
        '|'.join(board_cards) if board_cards else '_',
        '|'.join(action_history) if action_history else '_',
    ]
    
    canonical = ';'.join(parts)
    
    # SHA1 is fast and sufficient for infoset identity (not security)
    hash_obj = hashlib.sha1(canonical.encode('utf-8'))
    return hash_obj.hexdigest()


def parse_infoset_id(infoset_id: str) -> dict[str, any]:
    """
    Reverse-lookup infoset contents from hash (if you kept original mapping).
    
    For now, we rely on InformationSet storing original components.
    This function is a placeholder for future persistent storage.
    
    Args:
        infoset_id: Hex hash
    
    Returns:
        Would return {player, hole_cards, board_cards, action_history}
        For now: returns None (need external mapping)
    """
    # TODO: Implement persistent infoset ID → contents mapping
    # This would require:
    #   1. Dump all infosets to file (JSON or pickle)
    #   2. On load, rebuild InformationSetStorage from file
    return None
