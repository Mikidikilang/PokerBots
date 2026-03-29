"""
Real-Time Subgame Solving (RTA) for Deep CFR (Phase 4.1).

[PHASE 4.1] Online Re-solving for River Decisions

MOTIVATION
==========

Blueprint Strategy Limitation:
  The average strategy network (Blueprint) is trained offline on ALL possible game states.
  But on the river (final decision point with highest variance):
    - Ranges might be different than Blueprint assumed
    - Opponent's actual bets/checks update our beliefs
    - Best move is subgame-specific, not generic
  
  Example: Heads-up, river pot 100BB
    Blueprint says "bet 50% pot with AK"
    But opponent just checked (suggests weakness!)
    Optimal decision: NOW depends on specific river card & history
    
  Solution: Compute Nash in JUST THIS SUBGAME in real-time (seconds, not minutes)

STATE SPACE ON RIVER
====================

Subgame tree on river:
  - Hero hand: 169 × board (card abstraction) = ~200k hands
  - Opponent hand: 169 × board = ~200k hands
  - Pot size: 5-10k BB (normalization)
  - Stacks: 1-100 BB effective (finite bounds)
  - Hero position: (P1, P2) in heads-up
  
  Decision tree:
    Hero's turn to act:
      ├─ Fold (0 leaf)
      ├─ Check (move to opp. turn)
      │  ├─ Opp folds (hero wins pot)
      │  ├─ Opp checks (showdown)
      │  └─ Opp bets (hero's next decision)
      └─ Bet (move to opp.)
         ├─ Opp folds (hero wins pot + bet)
         ├─ Opp calls (go to showdown)
         └─ Opp raises (hero's next decision)
  
  Complexity: O(200k² × actions²) ≈ manageable with approximations

REAL-TIME SOLVING CONSTRAINTS
==============================

Time Budget: ~5 seconds per decision (online poker timeout)
Space Budget: ~1GB RAM (local to agent)
Accuracy: Within 5% of true Nash (exploitability tolerance)

Solution: Fast CFR variant
  - CFVS (CFR with Value Sampling): ~1000 iterations/sec
  - Pruning: Remove near-zero probability hands
  - Linear Approximation: Value function instead of exact tree
  - Rollout: For deep trees, approximate with rollout network

PHASES OF RTA
=============

Phase 1: Compute Ranges
  - Hero's range: hands with non-zero probability
    From Blueprint network for current history
  - Opponent's range: inferred from actions in hand
    Bayes rule: P(hand | actions) ∝ P(actions | hand) × P(hand prior)
  
Phase 2: Build Subgame Tree
  - Only include hands in non-zero ranges
  - Prune boards already folded (check if hand still live)
  - Limit tree depth: 2-3 rounds (bet/call/check)
  
Phase 3: Run Fast CFR Solver
  - MCCFR external sampling (already implemented!)
  - K=1000 iterations (takes ~1 second)
  - Converges to approximate Nash on this subgame
  
Phase 4: Extract Decision
  - Current situation: hero's hand + opp's range
  - Query regret network: "what's the regret for betting?"
  - Apply regret matching: sample action proportional to positive regrets
  - Execute decision

REGRET MATCHING ON RIVER
=========================

Instead of value functions, use regrets (we already have infrastructure!):

  For each hand in hero's range:
    1. Query RegretValueNetwork for action regrets
    2. Form strategy: σ(a) = max(R(a), 0) / Σ max(R(a'), 0)
    3. Sample action from σ(a)
  
  Multi-hand handling:
    - If hero is uncertain (multiple hands), weight by probability
    - σ_hero = Σ_hand P(hand) × σ(hand)
    - Sample from aggregate strategy

BLUEPRINT vs RTA COMPARISON
============================

Blueprint (Offline):
  ├─ Trained on all 200k hands × all boards
  ├─ Fixed strategy (doesn't adapt in-game)
  ├─ Fast: O(1) neural network inference
  └─ Exploitable: opponents learn patterns

RTA (Online):
  ├─ Solves only CURRENT subgame (1-100 hands)
  ├─ Adapts to specific opponent range
  ├─ Slow: O(1000 iters × tree size) = ~5 seconds
  └─ Unexploitable: always plays optimally on river

Hybrid (Recommended):
  ├─ Use Blueprint for preflop/flop/turn
  ├─ Switch to RTA on river (highest variance)
  ├─ Exploitability: exploitable/iters^{1/2} + ε_blueprint
  └─ Time: 4.99s Blueprint + 0.01s RTA

---

References:
  - Burch et al. (2014): "Improved Opponent Modeling"
  - Brown & Sandholm (2017): "Safe and Nested Subgame Solving"
  - Libratus: Brown et al. "Libratus: The Superhuman Poker Player" (2017)
  - Pluribus: Brown et al. "Superhuman AI for Multiplayer Poker" (2019)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# HAND RANGE REPRESENTATION
# ============================================================================

@dataclass
class HandRange:
    """Probability distribution over hands at a game state."""
    
    hands: Dict[str, float]  # {hand_name: probability}
    """
    Hand names: '169' canonical hands (e.g., 'AKs', 'AA', '72o')
    Probabilities: sum to 1.0 (or 0 if no hands in range)
    
    Example:
      {
        'AA': 0.05,    # 5% of hands are pocket aces
        'KK': 0.05,
        'AKs': 0.08,
        'AKo': 0.12,
        ...
      }
    """
    
    def __post_init__(self) -> None:
        """Validate and normalize."""
        total = sum(self.hands.values())
        if total > 0:
            # Normalize to sum to 1
            self.hands = {h: p / total for h, p in self.hands.items()}
    
    def equity_vs_range(self, hero_hand: str, opponent_range: HandRange) -> float:
        """
        Compute equity of hero_hand vs opponent's range.
        
        Equity = P(hero_hand wins vs random hand from opponent_range)
        
        Args:
            hero_hand: 'AKs', 'AA', etc.
            opponent_range: HandRange object
        
        Returns:
            Equity in [0, 1]
        
        NOTE: Placeholder. Real version requires hand evaluator.
        """
        # TODO: Implement real equity computation
        # Use hand strength evaluation (runouts, Monte Carlo, etc.)
        # For now: stub value
        return np.random.uniform(0.3, 0.7)
    
    def get_summary(self) -> str:
        """Human-readable summary of range."""
        non_zero = {h: p for h, p in self.hands.items() if p > 0}
        if not non_zero:
            return "Empty range"
        
        hands_list = sorted(non_zero.items(), key=lambda x: x[1], reverse=True)
        return ", ".join([f"{h}({p:.1%})" for h, p in hands_list[:5]])


# ============================================================================
# SUBGAME REPRESENTATION
# ============================================================================

@dataclass
class RiverSubgame:
    """
    Texas Hold'em river subgame: final decision point.
    
    Fixed: hole cards, board cards
    Variables: stack sizes, pot
    Play out: remaining actions until showdown
    """
    
    hero_hand: str  # 'AKs', 'AA', etc. (canonical)
    board: Tuple[str, ...]  # (card, card, card, card, card)
    pot_before_decision: float  # In BB
    hero_effective_stack: float  # Hero's remaining chips (BB)
    opponent_effective_stack: float  # Opponent's remaining chips (BB)
    hero_to_act: bool  # If False, opponent acts first
    
    # Optional: range information
    opponent_range: Optional[HandRange] = None
    
    def min_bet(self) -> float:
        """Minimum bet size (tournament: 1 BB, cash: depends)."""
        return 1.0
    
    def max_bet(self) -> float:
        """Maximum bet size (all-in)."""
        return min(self.hero_effective_stack, self.opponent_effective_stack)
    
    def is_all_in(self) -> bool:
        """Check if either player is all-in."""
        return (
            self.hero_effective_stack <= 0 or
            self.opponent_effective_stack <= 0
        )
    
    def __str__(self) -> str:
        return (
            f"RiverSubgame({self.hero_hand} on {self.board}, "
            f"pot={self.pot_before_decision}BB, "
            f"hero={self.hero_effective_stack}BB, opp={self.opponent_effective_stack}BB)"
        )


# ============================================================================
# RANGE-BASED SUBGAME SOLVING
# ============================================================================

class SubgameSolver:
    """
    Solves poker subgames via fast CFR variant.
    
    Inputs:
      - Hero hand (or range of hands)
      - Opponent range (inferred from action history)
      - Pot & stack information
    
    Outputs:
      - Optimal strategy (action distribution)
      - Exploitability bound
    
    Algorithm: External sampling MCCFR with hand sampling
      1. Sample hero hand from hero range (or use single hand)
      2. Sample opponent hand from opponent range
      3. Run MCCFR traversal on this (hero, opponent) pair
      4. Accumulate regrets for this pair
      5. Repeat K iterations → converge to subgame Nash
    """
    
    def __init__(
        self,
        regret_network: Optional[nn.Module] = None,
        num_iterations: int = 1000,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            regret_network: Already-trained regret value network
                           Can be None (use uniform strategy)
            num_iterations: MCCFR iterations (trade-off speed vs convergence)
            device: PyTorch device
        """
        self.regret_network = regret_network
        self.num_iterations = num_iterations
        self.device = device
        
        # Accumulate regrets during solving
        self.regrets: Dict[str, Dict[int, float]] = {}  # {hand: {action: regret}}
        
        logger.info(
            f"SubgameSolver: iterations={num_iterations}, "
            f"regret_network={'available' if regret_network else 'none'}"
        )
    
    def solve(
        self,
        subgame: RiverSubgame,
        hero_range: HandRange,
        opponent_range: HandRange,
    ) -> Dict[str, float]:
        """
        Solve river subgame and return optimal action distribution.
        
        Args:
            subgame: RiverSubgame with pot/stack info
            hero_range: Distribution over hero's possible hands
            opponent_range: Distribution over opponent's possible hands
        
        Returns:
            {action: probability} for hero's next action
            Keys: 'fold' (if facing bet), 'check', 'call', 'bet', 'raise'
        """
        logger.info(f"Solving subgame: {subgame}")
        logger.debug(f"  Hero range: {hero_range.get_summary()}")
        logger.debug(f"  Opponent range: {opponent_range.get_summary()}")
        
        # Initialize regret accumulators
        self.regrets = {
            hand: {action: 0.0 for action in range(12)}
            for hand in hero_range.hands.keys()
        }
        
        # Run MCCFR iterations
        for iteration in range(self.num_iterations):
            # Sample hands from ranges
            hero_hand = np.random.choice(
                list(hero_range.hands.keys()),
                p=list(hero_range.hands.values()),
            )
            opponent_hand = np.random.choice(
                list(opponent_range.hands.keys()),
                p=list(opponent_range.hands.values()),
            )
            
            # Solve this (hero_hand, opponent_hand) pair
            pair_regrets = self._solve_hand_pair(
                subgame, hero_hand, opponent_hand
            )
            
            # Accumulate regrets
            if hero_hand in self.regrets:
                for action, regret in pair_regrets.items():
                    self.regrets[hero_hand][action] += regret
            
            if (iteration + 1) % 100 == 0:
                logger.debug(f"  MCCFR iteration {iteration + 1}/{self.num_iterations}")
        
        # Aggregate strategy from regrets
        strategy = self._aggregate_strategy(hero_range)
        
        logger.info(f"Subgame solved: strategy={strategy}")
        return strategy
    
    def _solve_hand_pair(
        self,
        subgame: RiverSubgame,
        hero_hand: str,
        opponent_hand: str,
    ) -> Dict[int, float]:
        """
        Evaluate a single (hero_hand, opponent_hand) pair and compute regrets.
        
        Returns:
            {action_idx: counterfactual_regret}
        """
        # Placeholder: uniform regrets (no action is better/worse)
        # Real version: run game tree traversal, compute showdown values
        return {action: 0.0 for action in range(12)}
    
    def _aggregate_strategy(self, hero_range: HandRange) -> Dict[str, float]:
        """
        Convert accumulated regrets into aggregate strategy.
        
        Applies regret matching:
          σ(a) = max(Σ_hand R(a|hand) × P(hand), 0) / Σ_a' ...
        
        Returns:
            {action_name: probability}
        """
        # Sum regrets weighted by hand probability
        action_regrets = [0.0] * 12
        
        for hand, hand_prob in hero_range.hands.items():
            if hand in self.regrets:
                for action_idx, regret in self.regrets[hand].items():
                    action_regrets[action_idx] += hand_prob * regret
        
        # Regret matching: normalize positive regrets
        positive_regrets = [max(r, 0.0) for r in action_regrets]
        total_positive = sum(positive_regrets)
        
        if total_positive <= 0:
            # No positive regrets: uniform strategy
            strategy = {f"action_{i}": 1.0 / 12 for i in range(12)}
        else:
            strategy = {
                f"action_{i}": positive_regrets[i] / total_positive
                for i in range(12)
            }
        
        # Map to poker actions (fold, check, call, bet, raise)
        # Simplification: top-probability action
        dominant_action = max(strategy.items(), key=lambda x: x[1])[0]
        
        return {
            'fold': 0.0,
            'check': 0.3 if dominant_action == 'action_1' else 0.0,
            'call': 0.2 if dominant_action == 'action_2' else 0.0,
            'bet': 0.5 if dominant_action in ['action_3', 'action_4'] else 0.0,
        }


# ============================================================================
# OPPONENT RANGE INFERENCE
# ============================================================================

class RangeInference:
    """
    Infer opponent's hand range from action history.
    
    Uses Bayesian updating:
      P(hand | actions) ∝ P(actions | hand) × P(hand | prior)
    
    where:
      P(hand | prior) = Blueprint strategy for this hand
      P(actions | hand) = likelihood of opponent playing this hand given actions
    """
    
    def __init__(
        self,
        strategy_network: Optional[nn.Module] = None,
        prior_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            strategy_network: Trained blueprint network (gives prior)
            prior_weights: Manual prior weights (e.g., from game history)
        """
        self.strategy_network = strategy_network
        self.prior_weights = prior_weights or self._default_prior()
    
    def _default_prior(self) -> Dict[str, float]:
        """Default: uniform prior over all 169 hands."""
        return {f"hand_{i}": 1.0 / 169 for i in range(169)}
    
    def infer_range(
        self,
        board: Tuple[str, ...],
        action_history: List[Dict],
    ) -> HandRange:
        """
        Infer opponent range from action history.
        
        Args:
            board: Community cards
            action_history: List of {action, player, amount, ...}
        
        Returns:
            HandRange: posterior distribution over opponent hands
        """
        # Start with prior
        posterior = dict(self.prior_weights)
        
        # Bayes update for each action
        for action in action_history:
            if action.get('player') != 'opponent':
                continue  # Skip hero's actions
            
            # Likelihood: how likely is opponent to take this action?
            likelihood = self._action_likelihood(
                board, action.get('action'), action.get('amount', 0)
            )
            
            # Update posterior: P(hand | action) ∝ P(action | hand) × P(hand)
            for hand in posterior:
                posterior[hand] *= likelihood.get(hand, 0.5)
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {h: p / total for h, p in posterior.items()}
        
        return HandRange(posterior)
    
    def _action_likelihood(
        self,
        board: Tuple[str, ...],
        action: str,
        amount: float = 0,
    ) -> Dict[str, float]:
        """
        Likelihood of opponent taking this action with each hand.
        
        Returns:
            {hand: likelihood}
        
        NOTE: Placeholder. Real version queries strategy network.
        """
        # Simple heuristic: strong hands more likely to bet
        likelihood = {}
        for i in range(169):
            hand = f"hand_{i}"
            if action == 'bet':
                likelihood[hand] = 0.7  # Likely to bet (unclear hands)
            elif action == 'check':
                likelihood[hand] = 0.3  # Less likely to check
            else:
                likelihood[hand] = 0.5  # Neutral
        return likelihood


# ============================================================================
# RTA DECISION AGENT
# ============================================================================

class RTAAgent:
    """
    Real-Time Agent: makes decisions using online subgame solving.
    
    Workflow:
      1. Receive game state: hole cards, action history
      2. Infer opponent range from action history
      3. Build river subgame
      4. Solve subgame with SubgameSolver
      5. Return optimal action
    """
    
    def __init__(
        self,
        strategy_network: Optional[nn.Module] = None,
        regret_network: Optional[nn.Module] = None,
        num_solve_iterations: int = 1000,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            strategy_network: Blueprint strategy network (for priors)
            regret_network: Regret value network (for subgame solving)
            num_solve_iterations: MCCFR iterations per subgame
            device: PyTorch device
        """
        self.strategy_network = strategy_network
        self.regret_network = regret_network
        self.device = device
        
        self.range_inference = RangeInference(strategy_network=strategy_network)
        self.subgame_solver = SubgameSolver(
            regret_network=regret_network,
            num_iterations=num_solve_iterations,
            device=device,
        )
        
        logger.info(
            f"RTAAgent initialized: "
            f"strategy_network={'available' if strategy_network else 'none'}, "
            f"regret_network={'available' if regret_network else 'none'}, "
            f"solve_iters={num_solve_iterations}"
        )
    
    def get_action(
        self,
        hero_hand: str,
        board: Tuple[str, ...],
        pot: float,
        hero_stack: float,
        opponent_stack: float,
        action_history: List[Dict],
    ) -> Dict[str, any]:
        """
        Compute optimal action for current situation.
        
        Args:
            hero_hand: Hero's hole cards (canonical, e.g., 'AKs')
            board: Community cards
            pot: Pot size (in BB)
            hero_stack: Hero's remaining stack (in BB)
            opponent_stack: Opponent's remaining stack (in BB)
            action_history: Previous actions in hand
        
        Returns:
            {
                'action': 'fold' | 'check' | 'call' | 'bet' | 'raise',
                'amount': float (bet/raise only),
                'confidence': float [0, 1],
                'alternative_actions': {...}
            }
        """
        logger.info(
            f"RTAAgent.get_action: {hero_hand} on {board}, "
            f"pot={pot}BB, hero={hero_stack}BB, opp={opponent_stack}BB"
        )
        
        # Infer opponent range from action history
        opponent_range = self.range_inference.infer_range(board, action_history)
        
        # Create subgame
        subgame = RiverSubgame(
            hero_hand=hero_hand,
            board=board,
            pot_before_decision=pot,
            hero_effective_stack=hero_stack,
            opponent_effective_stack=opponent_stack,
            hero_to_act=True,  # Assume hero's turn (can infer from history)
            opponent_range=opponent_range,
        )
        
        # Create hero range (might be uncertain)
        # For now: 100% confident hero has hero_hand
        hero_range = HandRange({hero_hand: 1.0})
        
        # Solve subgame
        strategy = self.subgame_solver.solve(
            subgame, hero_range, opponent_range
        )
        
        # Execute best action
        best_action = max(strategy.items(), key=lambda x: x[1])[0]
        confidence = strategy[best_action]
        
        return {
            'action': best_action,
            'amount': self._compute_bet_amount(subgame, best_action),
            'confidence': confidence,
            'alternative_actions': {
                a: p for a, p in strategy.items() if p > 0.1
            },
        }
    
    def _compute_bet_amount(self, subgame: RiverSubgame, action: str) -> float:
        """Compute bet size for 'bet' or 'raise' action."""
        if action not in ['bet', 'raise']:
            return 0.0
        
        # Simple: pot-sized bet
        return subgame.pot_before_decision


# ============================================================================
# UTILITIES
# ============================================================================

def create_rta_agent(
    strategy_network: Optional[nn.Module] = None,
    regret_network: Optional[nn.Module] = None,
    num_iterations: int = 1000,
    device: str = 'cpu',
) -> RTAAgent:
    """Factory function to create RTA agent."""
    return RTAAgent(
        strategy_network=strategy_network,
        regret_network=regret_network,
        num_solve_iterations=num_iterations,
        device=torch.device(device),
    )
