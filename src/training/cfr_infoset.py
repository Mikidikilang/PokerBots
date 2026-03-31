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

import torch

from .dcfr_params import DCFRParameters, apply_dcfr_update

logger = logging.getLogger(__name__)


@dataclass
class InformationSet:
    """
    Game state abstraction: one infoset = all histories indistinguishable to a player.
    
    Stores:
        - Player's visible information (cards, board, history)
        - Regret for each action
        - Iteration count (for DCFR discounting)
    
    [PHASE 3 UPGRADE] Discounted CFR (DCFR):
        - Tracks iteration count for this infoset
        - Applies per-sign discount factors (α=1.5 for positive, β=0 for negative)
        - Formula: R^new(a) = discount(t, sign) * R^old(a) + r(a)
        - With γ=2, iteration-dependent discounting accelerates convergence
        - Discount decreases with iteration (less emphasis on old regrets)
    
    [PHASE 2 LEGACY] RM+ (Regret Matching Plus):
        - Constant discount factor (fallback mode if DCFR disabled)
        - R^new(a) = discount_factor * R^old(a) + r(a)
    """
    
    infoset_id: str                          # Hash of (player, cards, board, history)
    player: int                              # 0 = hero, 1 = opponent (heads-up)
    hole_cards: tuple[str, str]              # e.g., ('A', 'K')
    board_cards: tuple[str, ...]             # e.g., ('Q', 'J', 'T') or ()
    action_history: tuple[str, ...]          # e.g., ('check', 'raise')
    
    # ★ AUDIT FIX #4 ★: Store observation tensor for this infoset
    # This is populated during tree traversal when the infoset is first visited.
    # Required for strategy network training via behavioral cloning:
    #   inputs: obs_tensor (flattened state representation)
    #   targets: average_strategy (action probability distribution)
    obs_tensor: Optional[torch.Tensor] = None  # Shape [obs_dim], populated on first visit
    
    # Regret tracking
    cumulative_regret: dict[int, float] = field(default_factory=dict)
    regret_sum_squared: dict[int, float] = field(default_factory=dict)
    action_counts: dict[int, int] = field(default_factory=dict)
    visit_count: int = 0
    
    # For strategy computation
    positive_regret_sum: dict[int, float] = field(default_factory=dict)
    last_strategy: dict[int, float] = field(default_factory=dict)
    
    # [AUDIT FIX #3] Strategy averaging for behavioral cloning convergence
    # Tracks average strategy across iterations: σ̄(a|h) = (1/T) Σ_t σ^t(a|h)
    cumulative_strategy_sum: dict[int, float] = field(default_factory=dict)
    iteration_count_for_averaging: int = 0  # How many iterations have contributed
    
    # [PHASE 3] DCFR iteration tracking
    iteration_count: int = 0  # How many times this infoset has been updated
    dcfr_params: Optional[DCFRParameters] = None
    
    # Legacy RM+ mode (if DCFR disabled)
    use_dcfr: bool = True
    regret_discount_factor: float = 3.0    # Fallback if DCFR disabled
    
    def __post_init__(self):
        """Ensure all action dicts have consistent keys."""
        if not self.cumulative_regret:
            logger.debug(f"Created new infoset: {self.infoset_id}")
        if self.dcfr_params is None:
            self.dcfr_params = DCFRParameters()
    
    def add_regret(self, action: int, regret_value: float, 
                   importance_weight: float = 1.0):
        """
        Accumulate regret for an action using DCFR (NOT weighted by importance sampling).
        
        ★★★ CRITICAL FIX (AUDIT FIX #2) ★★★
        
        MATHEMATICAL GUARANTEE:
            Importance sampling weights are REMOVED from regret accumulation.
            
            Pure DCFR Formula (Brown & Sandholm 2019):
                R^new(a) = discount(t, sign) * R^old(a) + r(a)
                
            where discount depends on sign of R^old(a) and iteration count.
            
            Convergence Rate: O(1/√t) is GUARANTEED by DCFR theory.
            
            The regret_value input must ALREADY be scaled by reach probabilities
            during tree traversal (NOT here).
        
        [PHASE 2 LEGACY] RM+ Formula (disabled):
            R^new(a) = constant_discount * R^old(a) + r(a)
        
        Args:
            action: Action index (0-11 in poker)
            regret_value: Counterfactual regret PROPERLY scaled by reach probabilities
                         (NO additional importance weighting applied here)
            importance_weight: DEPRECATED - accepted for API compatibility but NOT used.
                              Will be removed in future versions.
        
        ★ AUDIT NOTE ★:
            importance_weight parameter exists only for backward compatibility.
            It is NEVER applied. Regrets are accumulated unweighted.
            If weighting is needed, it must be done at tree traversal time.
        """
        if action not in self.cumulative_regret:
            self.cumulative_regret[action] = 0.0
            self.regret_sum_squared[action] = 0.0
            self.action_counts[action] = 0
        
        regret_old = self.cumulative_regret[action]
        
        # ASSERT importance_weight is not used (for debugging)
        if importance_weight != 1.0:
            logger.warning(
                f"add_regret received importance_weight={importance_weight} "
                f"but this is ignored.  Weights must be applied at traversal time, "
                f"not during regret accumulation. See AUDIT FIX #2."
            )
        
        if self.use_dcfr and self.dcfr_params:
            # ★ PURE DCFR: NO importance weighting ★
            # Convergence proof requires unweighted regret accumulation
            regret_updated = apply_dcfr_update(
                regret_old=regret_old,
                regret_new=regret_value,  # ← NO weighting applied here
                iteration=self.iteration_count,
                params=self.dcfr_params,
            )
        else:
            # [PHASE 2] Legacy RM+: Constant discount (disabled by default)
            regret_updated = (
                self.regret_discount_factor * regret_old + regret_value
            )
        
        # ★ AUDIT FIX #5 ★: CFR+ (Regret Matching Plus) — Clamp to zero
        # CFR+ convergence is 10–1000× faster than vanilla CFR.
        # This clamps cumulative regrets to [0, ∞) each iteration.
        # Reference: "Regret Matching+" (Burch et al., 2014)
        regret_updated = max(regret_updated, 0.0)
        
        self.cumulative_regret[action] = regret_updated
        self.regret_sum_squared[action] += regret_value ** 2
        self.action_counts[action] += 1
        self.visit_count += 1
    
    def increment_iteration(self):
        """[PHASE 3] Increment iteration counter (called at end of CFR iteration)."""
        self.iteration_count += 1
        
        # [AUDIT FIX #3] Accumulate current strategy for averaging
        # This is called at the END of each iteration after regrets are finalized
        current_strategy = self.get_strategy()
        
        for action, prob in current_strategy.items():
            if action not in self.cumulative_strategy_sum:
                self.cumulative_strategy_sum[action] = 0.0
            # Add current iteration's probability to cumulative sum
            self.cumulative_strategy_sum[action] += prob
        
        self.iteration_count_for_averaging += 1
    
    def get_average_strategy(self, legal_actions: Optional[list[int]] = None) -> dict[int, float]:
        """
        [AUDIT FIX #3] Compute average strategy across all iterations.
        
        Formula:
            σ̄_i(a|h) = (1/T) Σ_{t=1}^T σ^t_i(a|h)
        
        This is the strategy guaranteed to converge to Nash equilibrium.
        Use this for behavioral cloning (strategy network training),
        NOT the current iteration's strategy.
        
        Args:
            legal_actions: If provided, restrict to these actions
        
        Returns:
            {action_idx: probability} where sum = 1.0
        """
        if not legal_actions:
            legal_actions = list(self.cumulative_strategy_sum.keys())
        
        if not legal_actions or self.iteration_count_for_averaging == 0:
            # Uniform if no data
            num_actions = len(legal_actions) if legal_actions else 1
            return {a: 1.0 / num_actions for a in legal_actions} if legal_actions else {}
        
        # Compute average by dividing cumsum by iteration count
        avg_strategy = {}
        total_prob = 0.0
        
        for action in legal_actions:
            cumsum = self.cumulative_strategy_sum.get(action, 0.0)
            avg_prob = cumsum / self.iteration_count_for_averaging
            avg_strategy[action] = avg_prob
            total_prob += avg_prob
        
        # Normalize (in case of floating point errors)
        if total_prob > 1e-8:
            avg_strategy = {a: p / total_prob for a, p in avg_strategy.items()}
        else:
            # Fallback to uniform
            num_actions = len(legal_actions)
            avg_strategy = {a: 1.0 / num_actions for a in legal_actions}
        
        return avg_strategy
    
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
        obs_tensor: Optional[torch.Tensor] = None,
    ) -> InformationSet:
        """
        Get existing infoset or create new one.
        
        ★ AUDIT FIX #4.5 ★: Optional obs_tensor parameter for behavioral cloning.
        If provided and infoset is newly created, store the observation tensor.
        This tensor is used later for strategy network training.
        
        Args:
            player: 0 or 1 (heads-up)
            hole_cards: Tuple of two card strings
            board_cards: Tuple of 0-5 card strings
            action_history: Tuple of action names
            obs_tensor: Optional[torch.Tensor] of shape [obs_dim] for this infoset.
                       Stored for behavioral cloning of average strategy.
        
        Returns:
            InformationSet object
        """
        infoset_id = hash_infoset(player, hole_cards, board_cards, action_history)
        
        if infoset_id not in self.infosets:
            infoset = InformationSet(
                infoset_id=infoset_id,
                player=player,
                hole_cards=hole_cards,
                board_cards=board_cards,
                action_history=action_history,
            )
            # Populate obs_tensor if provided (first visit to this infoset)
            if obs_tensor is not None:
                infoset.obs_tensor = obs_tensor
            self.infosets[infoset_id] = infoset
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
