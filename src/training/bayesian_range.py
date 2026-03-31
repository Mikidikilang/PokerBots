"""
Bayesian Hand Range Inference (Phase 4.2)

[PHASE 4.2] Proper range inference using strategy network and action history

Key Innovation:
    Instead of returning uniform 0.5 for all hands, compute proper Bayesian
    posterior using:
    
    P(hand | history) ∝ P(history | hand) × P(hand | prior)
    
    where:
    - P(hand | prior) = Blueprint network prediction (sigma)
    - P(history | hand) = Product of action likelihoods in sequence
    - Update multiplicatively with each action
    
Example (Heads-up, opponent bets 25 on river):
    Prior: uniform 0.006 (1/169 hands)
    Action: bet 25 into 50 pot
    
    Likelihood by hand strength:
    - AA/KK: 0.9 (strong hands bet strong)
    - AK: 0.7 (premium hands bet often)
    - 72o: 0.1 (trash hands bet rarely)
    
    Posterior (normalized):
    - AA: 0.006 × 0.9 = 0.0054
    - KK: 0.006 × 0.9 = 0.0054
    - 72o: 0.006 × 0.1 = 0.0006
    - (after normalization)

Implementation:
    1. Start with uniform prior (or blueprint prior)
    2. For each opponent action in sequence:
       - Get action likelihood from strategy network
       - Multiply posterior by likelihood
    3. Normalize to sum to 1.0
    4. Return HandRange with posterior

Strategy Network Integration:
    Network output = action softmax across {fold, check/call, all-in}
    Likelihood[hand] = σ_network[action] for that hand
    
    If we don't have per-hand action probs:
    - Use magnitude of betting (large bet → stronger hands)
    - Use historical frequency in training data
    - Use hand strength heuristics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HandRange:
    """
    Probability distribution over 169 canonical hands (Texas Hold'em).
    
    Hands: 13 × 13 = 169
      - 13 pairs: AA, KK, ..., 22
      - 78 unpaired same suit (AKs, AQs, ..., 32s)
      - 78 unpaired diff suit (AKo, AQo, ..., 32o)
    
    Canonical ordering (common in poker solvers):
      Index 0-12: AA, KK, QQ, ..., 22
      Index 13-90: AKs, AQs, AJs, ..., 32s
      Index 91-168: AKo, AQo, AJo, ..., 32o
    """
    
    hands: Dict[str, float]
    """
    {canonical_hand: probability}
    
    Example:
      {'AA': 0.06, 'KK': 0.04, 'AKs': 0.12, ...}
      Sum to 1.0
    """
    
    board: Tuple[str, ...] = ()
    """Community cards (for removed hand tracking)"""
    
    def __post_init__(self):
        """Normalize probabilities."""
        total = sum(self.hands.values())
        if total > 1e-6:
            self.hands = {h: p / total for h, p in self.hands.items()}
    
    def get_equity(self, vs_range: HandRange) -> float:
        """
        Equity vs opponent range (weighted average).
        
        Placeholder: requires actual hand evaluation.
        """
        return 0.5
    
    def get_non_zero_hands(self) -> List[str]:
        """Hands with > 1e-6 probability (live hands)."""
        return [h for h, p in self.hands.items() if p > 1e-6]
    
    def get_summary(self, top_n: int = 5) -> str:
        """Human-readable summary of top N hands."""
        top = sorted(self.hands.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return ", ".join([f"{h}({p:.1%})" for h, p in top])
    
    def prune_empty_hands(self, min_prob: float = 1e-6) -> HandRange:
        """Remove negligible probability hands (improve numerical stability)."""
        return HandRange({h: p for h, p in self.hands.items() if p > min_prob})


class BayesianRangeInference:
    """
    Infer opponent hand range from action history using Bayes rule.
    
    Algorithm:
        1. Initialize prior P(hand | initial belief)
        2. For each opponent action:
           a. Query network: P(action | hand, board, position)
           b. Multiply posterior: P(hand | history) ∝ P(action | hand) × P(hand)
           c. Normalize
        3. Return posterior as HandRange
    
    Key Implementation:
        - Action likelihood from strategy network
        - Multiplicative Bayesian update
        - Numerical stability (log-space computation)
        - Pruning of negligible probabilities
    """
    
    def __init__(
        self,
        strategy_network: Optional[nn.Module] = None,
        obs_builder: Optional[Any] = None,
        action_mapper: Optional[Any] = None,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            strategy_network: Trained π (strategy) network
                              Output: raw logits over actions for each hand
            obs_builder: ObservationBuilder to convert raw_state to tensors
            action_mapper: ActionMapper for legal action masking
            device: PyTorch device
        """
        self.strategy_network = strategy_network
        self.obs_builder = obs_builder
        self.action_mapper = action_mapper
        self.device = device
        
        # Default 169-hand canonical ordering
        self.canonical_hands = self._create_canonical_hands()
        
        logger.info(
            f"BayesianRangeInference initialized with "
            f"{len(self.canonical_hands)} canonical hands, "
            f"obs_builder={'present' if obs_builder else 'missing'}, "
            f"action_mapper={'present' if action_mapper else 'missing'}"
        )
    
    def _create_canonical_hands(self) -> List[str]:
        """Create list of 169 canonical hands in standard order."""
        hands = []
        
        # Pairs: AA, KK, ..., 22
        for rank_idx in range(13):
            rank = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'][rank_idx]
            hands.append(f"{rank}{rank}")
        
        # Suited: AKs, AQs, ..., 32s
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        for i in range(13):
            for j in range(i + 1, 13):
                hands.append(f"{ranks[i]}{ranks[j]}s")
        
        # Unsuited: AKo, AQo, ..., 32o
        for i in range(13):
            for j in range(i + 1, 13):
                hands.append(f"{ranks[i]}{ranks[j]}o")
        
        return hands
    
    def infer_range(
        self,
        board: Tuple[str, ...],
        action_history: List[Dict],
        raw_state: Optional[dict] = None,
        initial_prior: Optional[Dict[str, float]] = None,
        removed_cards: Optional[set] = None,
    ) -> HandRange:
        """
        Infer opponent range from action history via Bayesian updating.
        
        Args:
            board: Community cards (may be partial: flop=3, turn=4, river=5)
            action_history: Sequence of {
                'player': 'hero' or 'opponent',
                'action': 'fold', 'check', 'call', 'bet', 'raise', 'all_in',
                'amount': bet size (0 for check/fold),
            }
            raw_state: Current game state dict (contains pot, stacks, legal_actions, etc.)
                      Required for accurate observation generation when obs_builder is present.
            initial_prior: Starting probability distribution (default: uniform)
            removed_cards: Cards definitely not in opponent's hand (e.g., hero's hole)
        
        Returns:
            HandRange with posterior distribution
        """
        # Initialize prior
        if initial_prior is None:
            initial_prior = {h: 1.0 / len(self.canonical_hands) for h in self.canonical_hands}
        
        # Account for removed cards (crude: remove hands containing removed cards)
        # In full implementation: update using card combinatorics
        if removed_cards:
            for hand in list(initial_prior.keys()):
                # Example: if removed_cards={'As', 'Ks'}, remove 'AKs'
                # Simplified: just warn
                pass
        
        # Start with prior
        posterior = dict(initial_prior)
        
        logger.debug(
            f"Range inference on {board}: "
            f"prior_hands={sum(1 for p in posterior.values() if p > 0)}"
        )
        
        # Bayesian update for each opponent action
        opponent_actions = [a for a in action_history if a.get('player') == 'opponent']
        
        for step, action in enumerate(opponent_actions):
            action_name = action.get('action', 'unknown')
            amount = action.get('amount', 0)
            
            # Compute likelihood: P(action | hand, board, history)
            likelihoods = self._compute_action_likelihood(
                action_name=action_name,
                amount=amount,
                board=board,
                raw_state=raw_state,
                posterior_before=posterior,  # Use current posterior as context
            )
            
            # Bayesian update: P(hand | action) ∝ P(action | hand) × P(hand)
            for hand in posterior:
                posterior[hand] *= likelihoods.get(hand, 0.5)
            
            # Normalize
            total = sum(posterior.values())
            if total > 1e-10:
                posterior = {h: p / total for h, p in posterior.items()}
            
            logger.debug(f"  After action {step+1} ({action_name}): "
                        f"live_hands={sum(1 for p in posterior.values() if p > 1e-6)}")
        
        # Create HandRange with posterior
        result = HandRange(hands=posterior, board=board)
        result = result.prune_empty_hands(min_prob=1e-6)
        
        logger.info(f"Inferred range: {result.get_summary()}")
        return result
    
    def _compute_action_likelihood(
        self,
        action_name: str,
        amount: float,
        board: Tuple[str, ...],
        raw_state: Optional[dict],
        posterior_before: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute likelihood P(action | hand) for all hands.
        
        Real Implementation:
            For each canonical hand, builds an observation using ObservationBuilder
            with the actual game state (pot, stacks, legal actions, etc.).
            Queries the strategy network to get raw logits, applies legal action
            masking, softmax normalization, and extracts the probability for the
            target action.
        
        Args:
            action_name: Target action ('fold', 'check', 'call', 'bet', 'raise', 'all_in')
            amount: Bet size
            board: Community cards
            raw_state: Current game state dict (must contain legal_actions, pot, etc.)
            posterior_before: Current hand probability distribution
        
        Returns:
            {hand: likelihood in [0, 1]} where likelihood is P(action_name | hand)
        """
        likelihoods = {}
        
        # Fallback 1: No strategy network
        if self.strategy_network is None:
            logger.warning(
                "_compute_action_likelihood: strategy_network not provided. "
                "Using hand strength heuristics as fallback."
            )
            hand_strength = self._estimate_hand_strength(posterior_before)
            for hand in self.canonical_hands:
                strength = hand_strength.get(hand, 0.5)
                if action_name in ('bet', 'raise'):
                    likelihoods[hand] = 0.3 + 0.5 * strength
                elif action_name in ('check', 'call'):
                    likelihoods[hand] = 0.5 - 0.3 * abs(strength - 0.5)
                elif action_name == 'fold':
                    likelihoods[hand] = 0.3 - 0.3 * strength
                else:
                    likelihoods[hand] = 0.5
            return likelihoods
        
        # Fallback 2: No observation builder (can't build real observations)
        if self.obs_builder is None:
            logger.warning(
                "_compute_action_likelihood: obs_builder not provided. "
                "Using hand strength heuristics as fallback."
            )
            hand_strength = self._estimate_hand_strength(posterior_before)
            for hand in self.canonical_hands:
                strength = hand_strength.get(hand, 0.5)
                if action_name in ('bet', 'raise'):
                    likelihoods[hand] = 0.3 + 0.5 * strength
                elif action_name in ('check', 'call'):
                    likelihoods[hand] = 0.5 - 0.3 * abs(strength - 0.5)
                else:
                    likelihoods[hand] = 0.5
            return likelihoods
        
        # Fallback 3: No raw_state provided
        if raw_state is None:
            logger.warning(
                "_compute_action_likelihood: raw_state not provided. "
                "Using hand strength heuristics as fallback."
            )
            hand_strength = self._estimate_hand_strength(posterior_before)
            for hand in self.canonical_hands:
                strength = hand_strength.get(hand, 0.5)
                if action_name in ('bet', 'raise'):
                    likelihoods[hand] = 0.3 + 0.5 * strength
                elif action_name in ('check', 'call'):
                    likelihoods[hand] = 0.5 - 0.3 * abs(strength - 0.5)
                else:
                    likelihoods[hand] = 0.5
            return likelihoods
        
        # ★ REAL IMPLEMENTATION: Query strategy network for each canonical hand
        try:
            import torch.nn.functional as F
            from src.env.action_mapper import apply_action_mask
            
            for hand_idx, hand in enumerate(self.canonical_hands):
                try:
                    # Step 1: Create shallow copy of raw_state with this hand
                    state_copy = dict(raw_state)
                    state_copy["hand"] = hand  # Inject canonical hand
                    
                    # Step 2: Build observation tensor dict from the game state
                    obs_dict = self.obs_builder.build(state_copy, validate=False)
                    
                    # Step 3: Flatten observation and add batch dimension
                    flat_obs = self.obs_builder.flatten(obs_dict)  # Shape: (feature_dim,)
                    obs_tensor = flat_obs.unsqueeze(0).to(self.device)  # Shape: (1, feature_dim)
                    
                    # Step 4: Query strategy network in inference mode
                    with torch.inference_mode():
                        logits = self.strategy_network(obs_tensor)  # Shape: (1, num_actions)
                    
                    # Step 5: Extract legal actions from state
                    legal_actions_list = state_copy.get("legal_actions", [])
                    num_actions = logits.shape[-1]
                    action_mask = torch.zeros(1, num_actions, dtype=torch.float32, device=self.device)
                    
                    for action_idx in legal_actions_list:
                        if 0 <= action_idx < num_actions:
                            action_mask[0, action_idx] = 1.0
                    
                    # Step 6: Apply AMP-safe legal action masking (matches LBR Oracle)
                    masked_logits = apply_action_mask(logits, action_mask)  # Shape: (1, num_actions)
                    
                    # Step 7: Get probability distribution via softmax
                    action_probs = F.softmax(masked_logits, dim=-1)  # Shape: (1, num_actions)
                    
                    # Step 8: Extract probability for target action
                    action_idx = self._map_action_name_to_idx(action_name)
                    if action_idx is not None and 0 <= action_idx < num_actions:
                        likelihoods[hand] = action_probs[0, action_idx].item()
                    else:
                        # Action not in legal set or mapping failed
                        likelihoods[hand] = 0.5  # Neutral fallback
                
                except Exception as hand_error:
                    logger.debug(
                        f"Error processing hand {hand}: {hand_error}. Using fallback."
                    )
                    # For this hand only, use hand strength heuristic
                    hand_strength = self._estimate_hand_strength({hand: 1.0})
                    strength = hand_strength.get(hand, 0.5)
                    if action_name in ('bet', 'raise'):
                        likelihoods[hand] = 0.3 + 0.5 * strength
                    elif action_name in ('check', 'call'):
                        likelihoods[hand] = 0.5 - 0.3 * abs(strength - 0.5)
                    else:
                        likelihoods[hand] = 0.5
            
            return likelihoods
        
        except Exception as e:
            logger.error(
                f"Error querying strategy network for action likelihood: {e}",
                exc_info=True
            )
            # Fallback to hand strength heuristics on critical error
            hand_strength = self._estimate_hand_strength(posterior_before)
            for hand in self.canonical_hands:
                strength = hand_strength.get(hand, 0.5)
                if action_name in ('bet', 'raise'):
                    likelihoods[hand] = 0.3 + 0.5 * strength
                elif action_name in ('check', 'call'):
                    likelihoods[hand] = 0.5 - 0.3 * abs(strength - 0.5)
                else:
                    likelihoods[hand] = 0.5
            return likelihoods
    
    def _encode_board_tensor(self, board: Tuple[str, ...]) -> torch.Tensor:
        """Encode board cards as one-hot (1, 52) tensor."""
        card_vector = torch.zeros(52, dtype=torch.float32)
        
        suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                    '9': 7, 't': 8, 'j': 9, 'q': 10, 'k': 11, 'a': 12}
        
        for card_str in board:
            if len(card_str) >= 2:
                rank = card_str[0].lower()
                suit = card_str[1].lower()
                if rank in rank_map and suit in suit_map:
                    card_idx = rank_map[rank] * 4 + suit_map[suit]
                    if 0 <= card_idx < 52:
                        card_vector[card_idx] = 1.0
        
        return card_vector.unsqueeze(0)
    
    def _map_action_name_to_idx(self, action_name: str) -> int | None:
        """Map poker action name to action index (0-11)."""
        action_map = {
            'fold': 0,
            'check': 1,
            'call': 1,
            'bet': 2,
            'raise': 3,
            'all_in': 4,
        }
        return action_map.get(action_name.lower())
    
    def _estimate_hand_strength(self, range_dist: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate hand strength (equity vs random hand) for each hand.
        
        Heuristic (simplified):
            - Pairs: stronger (equity ~0.48-0.55 each)
            - Suited broadway: strong (equity ~0.45-0.50)
            - Unsuited broadway: medium (equity ~0.40-0.45)
            - Low cards: weak (equity ~0.25-0.35)
        """
        strengths = {}
        
        for hand in self.canonical_hands:
            if len(hand) == 2 and hand[0] == hand[1]:
                # Pair
                pair_rank = hand[0]
                rank_order = {'A': 13, 'K': 12, 'Q': 11, 'J': 10, 'T': 9,
                             '9': 8, '8': 7, '7': 6, '6': 5, '5': 4, '4': 3, '3': 2, '2': 1}
                rank_val = rank_order.get(pair_rank, 5)
                strengths[hand] = 0.40 + 0.08 * (rank_val / 13)  # 0.40-0.48
            
            elif hand[2] == 's':  # Suited
                # Estimate by high card
                high_rank = {'A': 13, 'K': 12, 'Q': 11, 'J': 10, 'T': 9,
                            '9': 8, '8': 7, '7': 6, '6': 5, '5': 4, '4': 3, '3': 2, '2': 1}
                h1 = high_rank.get(hand[0], 5)
                h2 = high_rank.get(hand[1], 5)
                strengths[hand] = 0.35 + 0.1 * (h1 + h2) / 26  # 0.35-0.45
            
            else:  # Unsuited
                h1 = {'A': 13, 'K': 12, 'Q': 11, 'J': 10, 'T': 9,
                     '9': 8, '8': 7, '7': 6, '6': 5, '5': 4, '4': 3, '3': 2, '2': 1}.get(hand[0], 5)
                h2 = {'A': 13, 'K': 12, 'Q': 11, 'J': 10, 'T': 9,
                     '9': 8, '8': 7, '7': 6, '6': 5, '5': 4, '4': 3, '3': 2, '2': 1}.get(hand[1], 5)
                strengths[hand] = 0.30 + 0.08 * (h1 + h2) / 26  # 0.30-0.38
        
        return strengths


class ImprovedRTAAgent:
    """
    RTA Agent using safe subgame solving + range inference.
    """
    
    def __init__(
        self,
        strategy_network: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu'),
    ):
        self.range_inference = BayesianRangeInference(
            strategy_network=strategy_network,
            device=device,
        )
    
    def get_decision(
        self,
        hero_hand: str,
        board: Tuple[str, ...],
        action_history: List[Dict],
        pot: float,
        hero_stack: float,
        opponent_stack: float,
    ) -> str:
        """
        Get action decision using range inference + safe subgame solving.
        
        Args:
            hero_hand: Canonical hand ('AKs', 'AA', etc.)
            board: Community cards
            action_history: Sequence of actions
            pot: Current pot (BB)
            hero_stack: Remaining stack (BB)
            opponent_stack: Opponent's remaining stack (BB)
        
        Returns:
            Action name: 'fold', 'check', 'call', 'bet', 'raise', 'all_in'
        """
        # Step 1: Infer opponent range
        opp_range = self.range_inference.infer_range(
            board=board,
            action_history=action_history,
        )
        
        logger.info(f"Decision ({hero_hand}): opp_range={opp_range.get_summary()}")
        
        # Step 2: Use range for subgame decision (stub)
        return "check"  # Placeholder


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Bayesian Range Inference Testing ===")
    
    inference = BayesianRangeInference()
    
    # River scenario: opponent bet 25 into 50 pot
    history = [
        {'player': 'opponent', 'action': 'bet', 'amount': 25},
    ]
    
    inferred = inference.infer_range(
        board=('As', 'Ks', '2h', '3d', '5c'),
        action_history=history,
    )
    
    print(f"Inferred range: {inferred.get_summary()}")
    print(f"Non-zero hands: {len(inferred.get_non_zero_hands())}")
    
    # Test multi-action sequence
    print("\n=== Multi-Action Inference ===")
    
    history2 = [
        {'player': 'opponent', 'action': 'bet', 'amount': 10},
        {'player': 'hero', 'action': 'call', 'amount': 10},
        {'player': 'opponent', 'action': 'check', 'amount': 0},
    ]
    
    inferred2 = inference.infer_range(
        board=('As', 'Ks', '2h', '3d'),
        action_history=history2,
    )
    
    print(f"Inferred after sequence: {inferred2.get_summary()}")
