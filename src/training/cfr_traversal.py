"""
MCCFR Traversal Engine (cfr_traversal.py).

[PHASE 2] Monte Carlo Counterfactual Regret Minimization (MCCFR) traversal.

MCCFR is a stochastic approximation of Counterfactual Regret Minimization that:
    1. Samples game trajectories via external sampling
    2. Updates regrets incrementally during traversal
    3. Converges to Nash equilibrium as iterations → ∞

EXTERNAL SAMPLING variant:
    - Only traverse actions of the current-player-to-update
    - Sample opponent actions from their current strategy
    - More efficient than outcome sampling (lower variance)
    - Still converges to Nash (provably)

ALGORITHM OUTLINE
------------------

external_sampling_traversal(h, p, σ, cfr_state):
    if h is terminal:
        return u_p(h)  # Terminal utility for player p
    
    P(h) = player whose turn it is at h
    
    if P(h) == p:  # Update player p's regrets
        v_h = 0
        v_h_a = {}
        
        for action a in legal_actions(h):
            # Recursively evaluate this action
            h' = h + a
            v_h_a[a] = external_sampling_traversal(h', p, σ, cfr_state)
            v_h += π_p(a | h) * v_h_a[a]  # Weight by current strategy
        
        # Compute counterfactual regrets
        for action a in legal_actions(h):
            regret_a = v_h_a[a] - v_h  # Value of this action - avg value
            cfr_state.add_regret(infoset(h), a, regret_a)
        
        return v_h
    
    else:  # Opponent's turn - sample action
        π_opponent_a = σ[infoset(h)][?]  # Current opponent strategy
        a ~ π_opponent_a
        h' = h + a
        return external_sampling_traversal(h', p, σ, cfr_state) * π_opponent_a[a]

---

References:
    - Lanctot et al. (2009): "An Introduction to Counterfactual Regret Minimization"
    - Bowling et al. (2015): "Heads-up Limit Hold'em Poker is Solved" (uses CFR+)
    - Burch et al. (2014): "Playing Imperfect Information Games through Machine Learning"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TraversalState:
    """Mutable state passed through MCCFR traversal."""
    
    player_to_update: int                       # 0 or 1 (whose regrets to update)
    cfr_state: Any                              # InformationSetStorage
    reach_probs: dict[str, float] = field(default_factory=dict)
    # reach_probs[infoset_id] = P(reach this infoset | both players' strategies)
    
    counterfactual_values: dict[str, float] = field(default_factory=dict)
    # counterfactual_values[infoset_id] = V(infoset from player's perspective)
    
    regret_updates: list[tuple[str, int, float]] = field(default_factory=list)
    # List of (infoset_id, action, regret) for batch updates


class MCCFRTraversal:
    """
    External sampling Monte Carlo CFR traversal.
    
    Recursively traverses game tree, updating regrets for current player while
    sampling opponent actions from their strategy.
    """
    
    def __init__(
        self,
        env: Any,                              # RLCard environment
        network: torch.nn.Module,              # Actor-Critic network
        infoset_storage: Any,                  # InformationSetStorage
        get_obs_tensor: Callable[[dict], torch.Tensor] = None,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            env: PokerEnvironment with reset(), step(), is_over()
            network: Neural network outputting (action_logits, value)
            infoset_storage: CFR information set storage
            get_obs_tensor: Function to convert obs_dict → tensor for network
            device: PyTorch device
        """
        self.env = env
        self.network = network
        self.infoset_storage = infoset_storage
        self.get_obs_tensor = get_obs_tensor or (lambda x: torch.tensor([]))
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.traversal_count = 0
        self.max_depth = 20  # Prevent infinite recursion
    
    def external_sampling_traversal(
        self,
        state: dict[str, Any],
        player_to_update: int,
        reach_probs: dict[int, float],
        depth: int = 0,
    ) -> float:
        """
        [CORE MCCFR ALGORITHM]
        
        Recursively traverse game tree, updating regrets only for player_to_update.
        
        Algorithm:
            1. If terminal: return terminal utility
            2. If current player = player_to_update:
               a. Evaluate each legal action
               b. Compute regrets (value_a - average_value)
               c. Record regrets for storage
            3. If opponent's turn:
               a. Sample one action from opponent strategy
               b. Recursively evaluate that branch
        
        Args:
            state: Current game state (obs_dict from environment)
            player_to_update: Which player's regrets to update (0 or 1)
            reach_probs: {player_id: probability of reaching this state}
            depth: Current recursion depth (prevent infinite loops)
        
        Returns:
            Utility value from player_to_update's perspective
        """
        # Termination conditions
        if depth > self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached, returning 0")
            return 0.0
        
        if self.env.is_over():
            # Terminal state: return the reward
            # TODO: Extract actual reward from terminal state
            # For now, placeholder
            return 0.0
        
        # Determine whose turn it is
        # TODO: Extract current player from state
        # This is environment-specific
        current_player = 0  # Placeholder
        
        # Get legal actions
        # TODO: Extract from environment
        legal_actions = list(range(12))  # Placeholder: all 12 actions
        
        # Get or create infoset ID
        infoset_id = self._get_infoset_id(state, current_player)
        
        if current_player == player_to_update:
            # ================================================================
            # UPDATE PLAYER'S REGRETS
            # ================================================================
            # For each action, recursively evaluate and compute regrets
            
            # Get current strategy for this information set
            strategy = self.infoset_storage.get_strategy_batch(
                [infoset_id],
                [legal_actions],
            )[0]  # Returns dict[action_idx: probability]
            
            # Evaluate each action
            action_values = {}
            avg_value = 0.0
            
            for action in legal_actions:
                # Take action and recurse
                next_state, reward = self.env.step(action)
                
                # Update reach probabilities
                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= strategy.get(action, 1.0 / len(legal_actions))
                
                # Recursive call with opponent as player_to_update next
                value = self.external_sampling_traversal(
                    state=next_state,
                    player_to_update=player_to_update,
                    reach_probs=new_reach_probs,
                    depth=depth + 1,
                )
                
                action_values[action] = value
                avg_value += strategy.get(action, 1.0 / len(legal_actions)) * value
                
                # Undo action (reset environment to current state)
                # TODO: Implement proper state saving/loading
                self._undo_action()
            
            # Compute and store counterfactual regrets
            for action in legal_actions:
                regret = action_values[action] - avg_value
                
                # Scale regret by reach probability of opponent reaching this state
                # (only opponent's actions affect counterfactual probability)
                opposing_reach_prob = reach_probs.get(1 - current_player, 1.0)
                scaled_regret = regret * opposing_reach_prob
                
                # Store regret
                self.infoset_storage.add_regret(
                    infoset_id=infoset_id,
                    action=action,
                    regret_value=scaled_regret,
                )
            
            return avg_value
        
        else:
            # ================================================================
            # OPPONENT'S TURN - SAMPLE ACTION
            # ================================================================
            # Sample one action from opponent's strategy
            # Only that branch is traversed (external sampling efficiency)
            
            strategy = self.infoset_storage.get_strategy_batch(
                [infoset_id],
                [legal_actions],
            )[0]
            
            # Sample action proportional to strategy
            action_probs = np.array([
                strategy.get(a, 1.0 / len(legal_actions))
                for a in legal_actions
            ])
            action_probs /= action_probs.sum()  # Normalize
            
            sampled_action = np.random.choice(legal_actions, p=action_probs)
            sampled_prob = action_probs[legal_actions.index(sampled_action)]
            
            # Take sampled action
            next_state, reward = self.env.step(sampled_action)
            
            # Update reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[1 - current_player] *= sampled_prob
            
            # Recursive traversal (only sampled branch)
            value = self.external_sampling_traversal(
                state=next_state,
                player_to_update=player_to_update,
                reach_probs=new_reach_probs,
                depth=depth + 1,
            )
            
            # Scale by probability of sampling this action (importance weighting)
            return value / sampled_prob if sampled_prob > 0 else 0.0
    
    def traverse_for_both_players(self, num_traversals: int = 1) -> dict[str, float]:
        """
        Run alternating traversals: update player 0, then player 1.
        
        This is the standard MCCFR algorithm for self-play:
            for t in 1..T:
                v_0 = external_sampling_traversal(root, player=0)
                v_1 = external_sampling_traversal(root, player=1)
        
        Args:
            num_traversals: Number of (p0, p1) traversal pairs
        
        Returns:
            {metric: value} with statistics
        """
        stats = {
            "total_traversals": 0,
            "mean_value_p0": 0.0,
            "mean_value_p1": 0.0,
            "infosets_discovered": 0,
        }
        
        values_p0 = []
        values_p1 = []
        
        for trav_idx in range(num_traversals):
            # Reset environment
            root_state = self.env.reset()
            
            # Traversal for player 0
            value_p0 = self.external_sampling_traversal(
                state=root_state,
                player_to_update=0,
                reach_probs={0: 1.0, 1: 1.0},
                depth=0,
            )
            values_p0.append(value_p0)
            
            # Reset environment again
            root_state = self.env.reset()
            
            # Traversal for player 1
            value_p1 = self.external_sampling_traversal(
                state=root_state,
                player_to_update=1,
                reach_probs={0: 1.0, 1: 1.0},
                depth=0,
            )
            values_p1.append(value_p1)
            
            self.traversal_count += 1
            
            if (trav_idx + 1) % 10 == 0:
                logger.info(
                    f"MCCFR Traversal {trav_idx + 1}: "
                    f"V_p0={value_p0:.4f}, V_p1={value_p1:.4f}"
                )
        
        # Aggregate stats
        if values_p0:
            stats["mean_value_p0"] = np.mean(values_p0)
        if values_p1:
            stats["mean_value_p1"] = np.mean(values_p1)
        stats["total_traversals"] = self.traversal_count
        stats["infosets_discovered"] = len(self.infoset_storage.infosets)
        
        return stats
    
    def _get_infoset_id(self, state: dict[str, Any], player: int) -> str:
        """
        Generate infoset ID from current state.
        
        TODO: Extract hero_cards, board_cards, action_history from state
        """
        from src.training.cfr_infoset import hash_infoset
        
        # Placeholder: use dummy values
        hero_cards = ("A", "K")
        board_cards = ()
        action_history = ()
        
        return hash_infoset(player, hero_cards, board_cards, action_history)
    
    def _undo_action(self) -> None:
        """
        Revert environment to previous state after evaluation.
        
        TODO: Implement proper state save/load mechanism
        For now, this is a placeholder.
        """
        pass


class ExternalSamplingMCCFR:
    """
    [INTEGRATION WRAPPER]
    
    Bundles MCCFR traversal with regret storage and strategy updates.
    
    Usage:
        >>> mccfr = ExternalSamplingMCCFR(env, network, infoset_storage)
        >>> for iteration in range(1000):
        ...     stats = mccfr.run_iteration()
        ...     if iteration % 100 == 0:
        ...         strategies = mccfr.get_current_strategies()
        ...         exploitability = mccfr.compute_exploitability()
    """
    
    def __init__(
        self,
        env: Any,
        network: torch.nn.Module,
        infoset_storage: Any,
        device: torch.device | str = "cpu",
    ):
        self.traversal = MCCFRTraversal(env, network, infoset_storage, device=device)
        self.infoset_storage = infoset_storage
        self.iteration = 0
    
    def run_iteration(self, num_traversals: int = 1) -> dict[str, float]:
        """
        Run one iteration of MCCFR (traverse for both players).
        
        Args:
            num_traversals: Number of (p0, p1) traversal pairs per iteration
        
        Returns:
            Statistics about this iteration
        """
        stats = self.traversal.traverse_for_both_players(num_traversals)
        stats["iteration"] = self.iteration
        
        self.iteration += 1
        
        return stats
    
    def get_current_strategies(self) -> dict[str, dict[int, float]]:
        """
        Extract current strategies from all infosets.
        
        Returns:
            {infoset_id: {action_idx: probability}}
        """
        strategies = {}
        
        for infoset_id, infoset in self.infoset_storage.infosets.items():
            strategies[infoset_id] = infoset.get_strategy()
        
        return strategies
    
    def compute_exploitability(self) -> float:
        """
        [PLACEHOLDER]
        
        Compute exploitability (distance from Nash equilibrium).
        
        Requires GTO baseline or opponent evaluation.
        TODO: Implement via best-response or GTO database lookup
        """
        return 0.0  # Placeholder
