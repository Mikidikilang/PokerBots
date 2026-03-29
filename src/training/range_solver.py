"""
Range-Based Subgame Solver (Phase 4.2)

Integrates safe subgame solving with Bayesian range inference.
Solves poker subgames over the full range of hands consistent with history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .safe_subgame_solver import SafeSubgameSolver, SubgameTrunkValue, SafeSubgameSolution
from .bayesian_range import BayesianRangeInference, HandRange

logger = logging.getLogger(__name__)


@dataclass
class SubgameContext:
    """Full context for subgame solving."""
    
    board: Tuple[str, ...]
    """Community cards (3-5 cards)"""
    
    action_history: List[Dict]
    """Full action sequence to reach this subgame"""
    
    pot: float
    """Current pot size (in BB)"""
    
    hero_stack: float
    """Hero's remaining stack (in BB)"""
    
    opponent_stack: float
    """Opponent's remaining stack (in BB)"""
    
    hero_effective_stack: float
    """min(hero_stack, opponent_stack)"""
    
    hero_position: str
    """'button' or 'big_blind'"""
    
    def __post_init__(self):
        self.hero_effective_stack = min(self.hero_stack, self.opponent_stack)


@dataclass
class RangeBasedSubgameSolution:
    """
    Solution to range-based subgame.
    
    Instead of a single action for one hand,
    returns action distribution across the full range.
    """
    
    range_strategy: Dict[str, Dict[str, float]]
    """
    {hand: {action: probability}}
    
    Example:
      {
        'AKs': {'fold': 0.0, 'check': 0.1, 'bet': 0.9},
        'AA': {'fold': 0.0, 'check': 0.0, 'bet': 1.0},
        '72o': {'fold': 0.3, 'check': 0.5, 'bet': 0.2},
      }
    """
    
    recommended_action: str
    """Best action for hero's current hand (majority probability)"""
    
    trunk_value_constraint: float
    """Blueprint trunk value (constraint)"""
    
    trunk_value_achieved: float
    """Actually achieved trunk value with solution"""
    
    is_safe: bool
    """Constraint satisfied"""
    
    iterations: int
    """CFR iterations to convergence"""
    
    solve_time: float
    """Wall-clock time (seconds)"""


class RangeBasedSubgameSolver:
    """
    Solve subgames across full hand ranges.
    
    Key Features:
        - Infers opponent range from action history (Bayesian)
        - Solves safe subgame respecting trunk value constraint
        - Returns action distribution across all hands
        - Supports range-based (not single-hand) decisions
    
    Workflow:
        1. Infer opponent range from history
        2. Look up hero's range for this position/history
        3. Call safe subgame solver with both ranges
        4. Extract strategy for each hand
        5. Return recommendation for hero's actual hand
    """
    
    def __init__(
        self,
        strategy_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        num_iterations: int = 1000,
        time_limit: float = 10.0,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            strategy_network: Blueprint strategy network
            value_network: Blueprint value network (for trunk computation)
            num_iterations: MCCFR iterations
            time_limit: Wall-clock time limit (seconds)
            device: PyTorch device
        """
        self.strategy_network = strategy_network
        self.value_network = value_network
        self.device = device
        
        self.range_inference = BayesianRangeInference(
            strategy_network=strategy_network,
            device=device,
        )
        
        self.safe_solver = SafeSubgameSolver(
            strategy_network=strategy_network,
            num_iterations=num_iterations,
            time_limit=time_limit,
            device=device,
        )
        
        logger.info("RangeBasedSubgameSolver initialized")
    
    def solve(
        self,
        hero_hand: str,
        context: SubgameContext,
        initial_hero_range: Optional[Dict[str, float]] = None,
    ) -> RangeBasedSubgameSolution:
        """
        Solve subgame with full range-based reasoning.
        
        Args:
            hero_hand: Hero's actual hand ('AKs', 'AA', etc.)
            context: SubgameContext with board, history, stacks
            initial_hero_range: Optional starting distribution over hero's hands
                               (default: uniform or from strategy network)
        
        Returns:
            RangeBasedSubgameSolution with full range strategy
        """
        import time
        start_time = time.time()
        
        logger.info(f"Range-based solving: {hero_hand} on {context.board}")
        logger.info(f"  Pot: {context.pot}BB, hero_eff: {context.hero_effective_stack}BB")
        
        # Step 1: Infer opponent range
        opponent_range = self.range_inference.infer_range(
            board=context.board,
            action_history=context.action_history,
        )
        
        logger.info(f"  Opponent range: {opponent_range.get_summary()}")
        
        # Step 2: Get hero range
        if initial_hero_range is None:
            hero_range = self._get_hero_range(context)
        else:
            hero_range = HandRange(hands=initial_hero_range, board=context.board)
        
        logger.info(f"  Hero range: {hero_range.get_summary()}")
        
        # Step 3: Compute trunk value constraint
        trunk_value = self._compute_trunk_value(
            hero_range,
            opponent_range,
            context,
        )
        
        logger.info(f"  Trunk value constraint: {trunk_value.hero_value:.2f}BB")
        
        # Step 4: Solve with safe subgame solver
        # Use first hand in hero range for the solver
        # (ideally would solve across all, but time-limited)
        first_hand = hero_hand if hero_hand in hero_range.hands else next(iter(hero_range.hands))
        
        safe_solution = self.safe_solver.solve(
            hero_hand=first_hand,
            hero_range=hero_range.hands,
            opponent_range=opponent_range.hands,
            trunk_value=trunk_value,
            board=context.board,
            pot=context.pot,
            hero_stack=context.hero_stack,
            opponent_stack=context.opponent_stack,
        )
        
        # Step 5: Build range strategy (simplified: use same for all hands)
        # In full implementation: solve separately for each hand vs opponent range
        range_strategy = {
            hand: safe_solution.strategy.copy()
            for hand in hero_range.get_non_zero_hands()
        }
        
        # Recommended action: sample from distribution
        recommended = max(
            safe_solution.strategy.items(),
            key=lambda x: x[1]
        )[0]
        
        elapsed = time.time() - start_time
        
        result = RangeBasedSubgameSolution(
            range_strategy=range_strategy,
            recommended_action=recommended,
            trunk_value_constraint=trunk_value.hero_value,
            trunk_value_achieved=safe_solution.trunk_value_achieved,
            is_safe=safe_solution.is_constraint_satisfied,
            iterations=safe_solution.iterations,
            solve_time=elapsed,
        )
        
        logger.info(
            f"Range solution: action={result.recommended_action}, "
            f"safe={result.is_safe}, time={elapsed:.2f}s"
        )
        
        return result
    
    def get_action(
        self,
        hero_hand: str,
        context: SubgameContext,
    ) -> str:
        """
        Get single action recommendation for hero's hand.
        
        Args:
            hero_hand: Hero's actual hand
            context: Subgame context
        
        Returns:
            Action name: 'fold', 'check', 'call', 'bet', 'raise', 'all_in'
        """
        solution = self.solve(hero_hand, context)
        return solution.recommended_action
    
    def _get_hero_range(self, context: SubgameContext) -> HandRange:
        """
        Compute hero's range at this decision node.
        
        Uses:
        - Starting hand ranges (preflop position)
        - Actions taken (folding some hands, not others)
        - Network probabilities (which hands likely at this state)
        """
        # Placeholder: return uniform
        num_hands = 169
        hands = {f"hand_{i}": 1.0 / num_hands for i in range(num_hands)}
        return HandRange(hands=hands)
    
    def _compute_trunk_value(
        self,
        hero_range: HandRange,
        opponent_range: HandRange,
        context: SubgameContext,
    ) -> SubgameTrunkValue:
        """
        Compute blueprint's expected value in trunk (leading to this subgame).
        
        This is the constraint for safe subgame solving.
        Hero cannot guarantee less than this.
        """
        # Placeholder: would use value network to compute
        # hero_value = E[value | hero_range, opponent_range, history]
        
        hero_value = 0.0  # Stub
        opponent_value = -hero_value
        
        return SubgameTrunkValue(
            hero_value=hero_value,
            opponent_value=opponent_value,
            pot_size=context.pot,
            hero_position=context.hero_position,
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Range-Based Subgame Solver Testing ===")
    
    solver = RangeBasedSubgameSolver(num_iterations=100, time_limit=2.0)
    
    context = SubgameContext(
        board=('As', 'Ks', '2h', '3d', '5c'),
        action_history=[
            {'player': 'opponent', 'action': 'check'},
            {'player': 'hero', 'action': 'bet', 'amount': 25},
            {'player': 'opponent', 'action': 'call', 'amount': 25},
        ],
        pot=50.0,
        hero_stack=100.0,
        opponent_stack=100.0,
        hero_position='button',
    )
    
    solution = solver.solve(
        hero_hand='AKs',
        context=context,
    )
    
    print(f"\nFinal recommendation: {solution.recommended_action}")
    print(f"Safe: {solution.is_safe}")
    print(f"Trunk margin: {solution.trunk_value_achieved - solution.trunk_value_constraint:.3f}BB")
    print(f"Solve time: {solution.solve_time:.2f}s")
    
    # Test action recommendation
    print("\n=== Action Recommendation ===")
    action = solver.get_action('AKs', context)
    print(f"Action for AKs: {action}")
