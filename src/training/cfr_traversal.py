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

from src.training.cfr_env_state import EnvStateManager

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
        self.env_state_manager = EnvStateManager(env)  # ★ AUDIT FIX #2 ★
        
        self.traversal_count = 0
        self.max_actions = 60  # Maximum actions in NLHE hand (poker-theoretic limit)
    
    def external_sampling_traversal(
        self,
        state: dict[str, Any],
        player_to_update: int,
        reach_probs: dict[int, float],
        action_count: int = 0,
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
            action_count: Total actions taken so far (prevent infinite loops via stalemate guard)
        
        Returns:
            Utility value from player_to_update's perspective
        """
        # Stalemate guard: break if too many actions (max 60 in heads-up NLHE)
        if action_count >= self.max_actions:
            logger.warning(f"Max actions {self.max_actions} reached at action_count={action_count}, returning 0")
            return 0.0
        
        # Termination conditions
        if self.env.is_over():
            # Terminal state: return the reward
            logger.debug(f"[A{action_count}] Terminal state reached, returning 0.0")
            return 0.0
        
        # Determine whose turn it is
        # RLCard stores current player in the env, not state dict
        try:
            current_player = self.env.get_legal_action_agent()
        except:
            # Fallback if method doesn't exist
            current_player = 0
        
        # Get legal actions directly from state dict
        legal_actions = state.get('legal_actions', list(range(12)))
        # RLCard provides legal_actions as OrderedDict, extract keys
        if hasattr(legal_actions, 'keys'):
            legal_actions = list(legal_actions.keys())
        elif not isinstance(legal_actions, list):
            legal_actions = list(legal_actions) if legal_actions else list(range(12))
        
        # ★ CRITICAL FIX ★: Extract cards ONCE at start
        # RLCard stores cards in raw_obs['hand'] and raw_obs['public_cards'] as string lists
        raw_obs = state.get('raw_obs', {})
        if isinstance(raw_obs, dict):
            hero_cards = tuple(raw_obs.get('hand', []))
            board_cards = tuple(raw_obs.get('public_cards', []))
        else:
            # Fallback if raw_obs is not a dict
            hero_cards = ()
            board_cards = ()
        action_history = ()  # TODO: Extract full action history from state
        
        # Convert state dict to observation tensor for strategy network training
        obs_tensor = self._state_dict_to_tensor(state) if state else None
        
        # Get or create infoset - this computes infoset_id internally
        infoset = self.infoset_storage.get_or_create_infoset(
            player=current_player,
            hole_cards=hero_cards,
            board_cards=board_cards,
            action_history=action_history,
            obs_tensor=obs_tensor,  # ★ Pass obs_tensor on first creation
        )
        
        # ★ CRITICAL FIX ★: Use infoset object's ID, not our own computation
        # This ensures we use the EXACT same hash as get_or_create_infoset() computed
        infoset_id = infoset.infoset_id
        
        # Only log nodes at action_count 0-20 to reduce output volume
        if action_count <= 20:
            logger.debug(f"[A{action_count}] Node: player={current_player}, actions={len(legal_actions)}, "
                        f"infoset={infoset_id[:8]}..., hero={hero_cards}, board={board_cards}")
        
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
                # ★ AUDIT FIX #2 ★: Save state before action, auto-restore after eval
                # External Sampling MCCFR requires evaluating ALL actions at this node
                # for the updating player. Each action must start from the SAME state.
                with self.env_state_manager.savepoint():
                    # Take action
                    next_state, reward = self.env.step(action)
                    
                    # Update reach probabilities
                    new_reach_probs = reach_probs.copy()
                    new_reach_probs[current_player] *= strategy.get(action, 1.0 / len(legal_actions))
                    
                    # Recursive call with opponent as player_to_update next
                    value = self.external_sampling_traversal(
                        state=next_state,
                        player_to_update=player_to_update,
                        reach_probs=new_reach_probs,
                        action_count=action_count + 1,
                    )
                    # Auto-restore environment state on context exit
                
                action_values[action] = value
                avg_value += strategy.get(action, 1.0 / len(legal_actions)) * value
            
            # Compute and store counterfactual regrets
            # ★ CRITICAL FIX ★: Use SAME infoset_id computed at start of function
            for action in legal_actions:
                regret = action_values[action] - avg_value
                
                # ★ AUDIT FIX #2.5 ★: Scale regret by reach probability of opponent reaching this state
                # (only opponent's actions affect counterfactual probability)
                # 
                # In pure CFR with explicit game tree: reach probability is exact
                # In Deep CFR with card abstraction: reach probability should be weighted by bucket
                #
                # Formula (heads-up):
                #   counterfactual_regret = regret(a) * π_{-i}(reach this state)
                #
                # where π_{-i} = product of opponent actions that led to this state
                # Stored in reach_probs[1-current_player]
                
                opposing_reach_prob = reach_probs.get(1 - current_player, 1.0)
                scaled_regret = regret * opposing_reach_prob
                
                # TODO: If using card abstraction buckets, also weight by P(concrete | bucket)
                # bucket_weight = get_bucket_weight(infoset_id, action)
                # weighted_regret = scaled_regret * bucket_weight
                
                # Store regret (unweighted by importance, per AUDIT FIX #1)
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
            
            # Ensure legal_actions is a list for indexing
            if not isinstance(legal_actions, list):
                legal_actions_list = list(legal_actions)
            else:
                legal_actions_list = legal_actions
            
            sampled_action = np.random.choice(legal_actions_list, p=action_probs)
            sampled_action_idx = legal_actions_list.index(sampled_action)
            sampled_prob = action_probs[sampled_action_idx]
            
            logger.debug(f"[A{action_count}] Opponent turn: sampled action={sampled_action} (prob={sampled_prob:.3f})")
            
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
                action_count=action_count + 1,
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
            # Reset environment (returns dict, not tuple)
            root_state = self.env.reset()
            
            # Traversal for player 0
            value_p0 = self.external_sampling_traversal(
                state=root_state,
                player_to_update=0,
                reach_probs={0: 1.0, 1: 1.0},
                action_count=0,
            )
            values_p0.append(value_p0)
            
            # Reset environment again
            root_state = self.env.reset()
            
            # Traversal for player 1
            value_p1 = self.external_sampling_traversal(
                state=root_state,
                player_to_update=1,
                reach_probs={0: 1.0, 1: 1.0},
                action_count=0,
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
        
        ★ AUDIT FIX #3 ★: Extract real hole cards and board cards from state dict
        instead of using placeholder values ("A", "K").
        
        Infoset = hash(player, hole_cards, board_cards, action_history)
        
        Card encoding (from features.py._encode_cards):
            - hole_cards: 52-dim multi-hot vector for hero's hole cards
            - community_cards: 52-dim multi-hot vector for board cards
            
        Args:
            state: Current game state dict with 'hole_cards', 'community_cards'
            player: Player index (0 or 1 in heads-up)
        
        Returns:
            Unique infoset hash string
        """
        from src.training.cfr_infoset import hash_infoset
        
        # Extract real hole cards from state
        hole_cards = self._decode_card_tensor(state.get("hole_cards", torch.zeros(52)))
        if not hole_cards:
            hole_cards = ("A", "K")  # Fallback only if extraction fails
        
        # Extract board cards from state
        board_cards = self._decode_card_tensor(state.get("community_cards", torch.zeros(52)))
        
        # Approximate action history from betting history length
        # (Full action_history extraction requires detailed log parsing)
        action_history = ()
        if "betting_history" in state:
            betting_hist = state["betting_history"]
            if isinstance(betting_hist, torch.Tensor):
                # Each action is ~13 dims if extended history enabled
                action_count = int(betting_hist.nonzero().shape[0] / 13)
                action_history = tuple(str(i) for i in range(action_count))  # String actions
        
        return hash_infoset(player, hole_cards, board_cards, action_history)
    
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
        
        # Ensure tensor is 1D
        card_tensor = card_tensor.flatten()
        
        # Find all indices where card_tensor == 1.0
        indices = torch.nonzero(card_tensor == 1.0, as_tuple=False).squeeze(-1)
        
        # Handle scalar output from nonzero
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        
        cards = []
        for idx in indices.tolist():
            rank_idx = idx // 4
            suit_idx = idx % 4
            
            if 0 <= rank_idx < 13 and 0 <= suit_idx < 4:
                card_str = RANK_NAMES[rank_idx][0] + SUIT_NAMES[suit_idx].lower()
                cards.append(card_str)
        
        return tuple(cards)
    
    def _state_dict_to_tensor(self, state_dict: dict[str, Any]) -> torch.Tensor:
        """
        Convert observation state dict to flattened tensor.
        
        Concatenates all state components (hole cards, board cards, env metrics, etc.)
        into a single tensor suitable for network input.
        
        Args:
            state_dict: Dictionary with observation components
            - "hole_cards": [52] multi-hot vector
            - "community_cards": [52] multi-hot vector
            - "env_metrics": [...] environment variables (pot, stacks)
            - "betting_history": [...] action history
            - etc.
        
        Returns:
            torch.Tensor of shape [obs_dim] (flattened concatenation)
        """
        if not state_dict:
            return torch.tensor([], dtype=torch.float32)
        
        tensors = []
        
        # Standard order (deterministic for reproducibility):
        for key in sorted(state_dict.keys()):
            value = state_dict[key]
            
            if isinstance(value, torch.Tensor):
                tensor = value.clone().detach().flatten()
            elif isinstance(value, (list, tuple)):
                # Try to convert - skip if contains non-numeric values
                try:
                    tensor = torch.tensor(value, dtype=torch.float32).flatten()
                except (ValueError, TypeError):
                    # Skip fields with non-numeric data (e.g., strings, action records)
                    continue
            elif isinstance(value, (int, float)):
                tensor = torch.tensor([value], dtype=torch.float32)
            else:
                # Skip unsupported types (str, dict, etc.)
                continue
            
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            
            tensors.append(tensor)
        
        if tensors:
            return torch.cat(tensors, dim=0)
        else:
            return torch.tensor([], dtype=torch.float32)
    
    def _undo_action(self) -> None:
        """
        [DEPRECATED] This method is kept for backward compatibility but is no longer used.
        
        ★ AUDIT FIX #2 ★: Environment state is now restored automatically via
        EnvStateManager.savepoint() context manager in external_sampling_traversal().
        
        Each action evaluation is wrapped in: `with env_state_manager.savepoint():`
        which auto-restores the environment state on context exit. This is more
        reliable than explicit undo() calls.
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
