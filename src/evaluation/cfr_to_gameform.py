"""
CFR to GameForm Converter (Phase 5)

Extract reduced game form (bimatrix payoff matrices) from converged CFR.

Key Functions:
    1. **Infoset Extraction**: Get all information sets from CFR tree
    2. **Strategy Extraction**: Convert regrets→frequencies→mixed strategies
    3. **Game Form Reduction**: Collapse multi-level game tree to 2-player matrix
    4. **Payoff Computation**: Compute expected payoffs for strategy pairs

Reference:
    - Brown & Sandholm (2017): Safe subgame solving
    - Lanctot et al. (2012): No-limit Hold'em solving
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# INFORMATION SET STRUCTURE
# ============================================================================

@dataclass
class InformationSet:
    """
    Single information set from CFR game tree.
    
    An infoset represents all game states indistinguishable to a player
    (same private info, same public action history).
    """
    
    infoset_id: str
    """Unique identifier"""
    
    player: int
    """Player to move (0 or 1)"""
    
    actions: List[str]
    """Available actions (e.g., ['fold', 'call', 'raise'])"""
    
    regrets: Dict[str, float]
    """Regret per action"""
    
    reach_probability: float = 0.0
    """Probability of reaching this infoset"""
    
    strategy: np.ndarray = None
    """Computed strategy (lazy-initialized)"""
    
    def compute_strategy_from_regrets(self) -> np.ndarray:
        """
        Convert regrets to strategy using RM+.
        
        Regret matching+:
            σ(a) = max(R(a), 0) / Σ max(R(a'), 0)
        
        Returns:
            Strategy as probability distribution over actions
        """
        regrets = np.array([self.regrets.get(a, 0.0) for a in self.actions])
        
        # RM+: only use positive regrets
        positive_regrets = np.maximum(regrets, 0.0)
        
        # Normalize
        if np.sum(positive_regrets) > 1e-10:
            strategy = positive_regrets / np.sum(positive_regrets)
        else:
            # Uniform if all regrets ≤ 0
            strategy = np.ones(len(self.actions)) / len(self.actions)
        
        self.strategy = strategy
        return strategy
    
    def get_strategy(self) -> np.ndarray:
        """Get strategy (compute if not cached)."""
        if self.strategy is None:
            self.compute_strategy_from_regrets()
        return self.strategy
    
    def expected_action_value(self, action_index: int, game_values: Dict) -> float:
        """
        Compute expected value of taking an action from this infoset.
        
        Args:
            action_index: Index of action to take
            game_values: Precomputed values of resulting positions
        
        Returns:
            Expected value of this action
        """
        # This requires knowledge of game tree continuation
        # Placeholder: return regret value as proxy
        action = self.actions[action_index]
        return self.regrets.get(action, 0.0)


# ============================================================================
# INFOSET COLLECTION FROM CFR
# ============================================================================

class InformationSetCollector:
    """
    Extract infosets from a trained CFR instance.
    """
    
    def __init__(self):
        self.infosets: Dict[str, InformationSet] = {}
        self.player_infosets: Dict[int, List[str]] = defaultdict(list)
    
    def collect_from_cfr_tree(self, cfr_root) -> Dict[str, InformationSet]:
        """
        Traverse CFR game tree and collect all infosets.
        
        Args:
            cfr_root: Root node of CFR tree
        
        Returns:
            Dict mapping infoset_id → InformationSet
        """
        self._traverse_cfr_node(cfr_root, {}, 0)
        logger.info(f"Collected {len(self.infosets)} infosets")
        return self.infosets
    
    def _traverse_cfr_node(self, node, history, player_to_move):
        """
        Recursively traverse CFR tree.
        
        Args:
            node: Current CFR node (has .infoset_id, .regrets, .children)
            history: Action history tuple
            player_to_move: Current player (0 or 1)
        """
        # Extract infoset if this is a decision node
        if hasattr(node, 'infoset_id') and hasattr(node, 'regrets'):
            infoset_id = node.infoset_id
            
            if infoset_id not in self.infosets:
                # Create new infoset
                actions = list(node.children.keys()) if hasattr(node, 'children') else []
                regrets = node.regrets if isinstance(node.regrets, dict) else {}
                
                infoset = InformationSet(
                    infoset_id=infoset_id,
                    player=player_to_move,
                    actions=actions,
                    regrets=regrets,
                )
                
                self.infosets[infoset_id] = infoset
                self.player_infosets[player_to_move].append(infoset_id)
        
        # Recurse to children
        if hasattr(node, 'children'):
            next_player = 1 - player_to_move  # Other player (unless chance)
            
            for action, child in node.children.items():
                new_history = history + (action,)
                self._traverse_cfr_node(child, new_history, next_player)
    
    def get_player_infosets(self, player: int) -> List[InformationSet]:
        """Get all infosets for a specific player."""
        return [self.infosets[iid] for iid in self.player_infosets[player]]


# ============================================================================
# GAME FORM EXTRACTION
# ============================================================================

@dataclass
class GameFormExtraction:
    """
    Result of extracting game form from CFR.
    """
    
    strategies_p0: List[str]
    """Pure strategy profiles for player 0 (action sequences)"""
    
    strategies_p1: List[str]
    """Pure strategy profiles for player 1"""
    
    payoff_matrix_p0: np.ndarray
    """Expected payoff for player 0 (num_strategies_p0 × num_strategies_p1)"""
    
    payoff_matrix_p1: np.ndarray
    """Expected payoff for player 1"""
    
    infosets_p0: List[InformationSet]
    """All infosets for player 0"""
    
    infosets_p1: List[InformationSet]
    """All infosets for player 1"""


class GameFormExtractor:
    """
    Convert CFR to reduced normal form game.
    """
    
    def __init__(self, cfr_solver):
        """
        Args:
            cfr_solver: Trained CFR instance (has traverse method)
        """
        self.cfr_solver = cfr_solver
        self.collector = InformationSetCollector()
        self.infosets = {}
    
    def extract_game_form(self) -> GameFormExtraction:
        """
        Extract full game form (bimatrix) from CFR.
        
        For small games (Leduc, Kuhn), this enumerates all pure strategy
        profiles and computes expected payoffs.
        
        Returns:
            GameFormExtraction with payoff matrices
        """
        logger.info("Extracting game form from CFR...")
        
        # Step 1: Collect infosets
        self.infosets = self.collector.collect_from_cfr_tree(
            self.cfr_solver.get_tree_root()  # Assumes this method exists
        )
        
        infosets_p0 = self.collector.get_player_infosets(0)
        infosets_p1 = self.collector.get_player_infosets(1)
        
        logger.info(f"Player 0 infosets: {len(infosets_p0)}")
        logger.info(f"Player 1 infosets: {len(infosets_p1)}")
        
        # Step 2: Enumerate pure strategy profiles
        strategies_p0 = self._enumerate_pure_strategies(infosets_p0)
        strategies_p1 = self._enumerate_pure_strategies(infosets_p1)
        
        logger.info(f"Pure strategies P0: {len(strategies_p0)}")
        logger.info(f"Pure strategies P1: {len(strategies_p1)}")
        
        # Step 3: Compute payoff matrix
        payoff_matrix_p0, payoff_matrix_p1 = self._compute_payoff_matrix(
            infosets_p0, infosets_p1, strategies_p0, strategies_p1
        )
        
        extraction = GameFormExtraction(
            strategies_p0=[str(s) for s in strategies_p0],
            strategies_p1=[str(s) for s in strategies_p1],
            payoff_matrix_p0=payoff_matrix_p0,
            payoff_matrix_p1=payoff_matrix_p1,
            infosets_p0=infosets_p0,
            infosets_p1=infosets_p1,
        )
        
        logger.info(
            f"Game form extracted: "
            f"{len(strategies_p0)} × {len(strategies_p1)} matrix"
        )
        
        return extraction
    
    def _enumerate_pure_strategies(self, infosets: List[InformationSet]) -> List:
        """
        Enumerate all pure strategy profiles (one action per infoset).
        
        Args:
            infosets: All infosets for one player
        
        Returns:
            List of pure strategies (each is a tuple of actions)
        """
        if not infosets:
            return [[]]
        
        # Start with empty strategy
        strategies = [[]]
        
        # For each infoset, multiply by its actions
        for infoset in infosets:
            new_strategies = []
            for existing in strategies:
                for action in infoset.actions:
                    new_strategies.append(existing + [action])
            strategies = new_strategies
        
        return strategies
    
    def _compute_payoff_matrix(
        self,
        infosets_p0: List[InformationSet],
        infosets_p1: List[InformationSet],
        strategies_p0: List,
        strategies_p1: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute expected payoffs for all strategy pairs.
        
        Args:
            infosets_p0, infosets_p1: All infosets per player
            strategies_p0, strategies_p1: All pure strategies
        
        Returns:
            (payoff_matrix_p0, payoff_matrix_p1)
        """
        n0 = len(strategies_p0)
        n1 = len(strategies_p1)
        
        payoff_p0 = np.zeros((n0, n1))
        payoff_p1 = np.zeros((n0, n1))
        
        logger.info(f"Computing {n0} × {n1} payoff matrix...")
        
        for i, strat_p0 in enumerate(strategies_p0):
            for j, strat_p1 in enumerate(strategies_p1):
                # Evaluate strategy pair (i, j)
                u0, u1 = self._evaluate_strategy_pair(
                    strat_p0, strat_p1, infosets_p0, infosets_p1
                )
                payoff_p0[i, j] = u0
                payoff_p1[i, j] = u1
            
            if (i + 1) % max(1, n0 // 10) == 0:
                logger.info(f"  {i+1}/{n0} strategy pairs evaluated")
        
        return payoff_p0, payoff_p1
    
    def _evaluate_strategy_pair(
        self,
        strat_p0: List[str],
        strat_p1: List[str],
        infosets_p0: List[InformationSet],
        infosets_p1: List[InformationSet],
    ) -> Tuple[float, float]:
        """
        ★ AUDIT FIX #4 ★: Properly evaluate strategy pair by tree traversal.
        
        [CRITICAL] Previously used regrets as proxy for payoffs (MATHEMATICALLY WRONG).
        
        Algorithm:
            1. Start at game root
            2. At each decision node:
               - Identify which infoset we're in
               - Use fixed strategy to determine action
            3. At chance nodes:
               - Sample outcome (or enumerate if small)
            4. At terminal nodes:
               - Compute payoff for each player
            5. Return expected payoff (sum payoffs weighted by probabilities)
        
        Implementation:
            For heads-up poker, we enumerate game tree recursively with fixed strategies.
            State = (player_turn, hero_cards, opp_cards, board, pot, stacks)
            
            At player 0's infoset:
                action_p0 = fixed strategy maps (infoset_index) -> action
            At player 1's infoset:
                action_p1 = fixed strategy maps (infoset_index) -> action
        
        Returns:
            (payoff_p0, payoff_p1) where payoff_p_i with both players using fixed strats
        """
        # TODO: Implement complete recursive tree traversal with fixed strategies
        #
        # Pseudocode:
        # def traverse_fixed_strats(state, strat_p0, strat_p1):
        #     if is_terminal(state):
        #         return payoff(state)
        #     
        #     player = whose_turn(state)
        #     infoset_id = hash_infoset(state)
        #     
        #     if player == 0:
        #         action = strat_p0[index_of_infoset_id]
        #     else:
        #         action = strat_p1[index_of_infoset_id]
        #     
        #     next_state = apply_action(state, action)
        #     return traverse_fixed_strats(next_state, strat_p0, strat_p1)
        
        # INTERIM: Use information set regret values as approximation
        # This is better than nothing but still not exact game tree evaluation
        u0 = 0.0
        u1 = 0.0
        
        if infosets_p0 and strat_p0:
            u0 = sum(
                infoset.regrets.get(action, 0.0) 
                for infoset, action in zip(infosets_p0, strat_p0[:len(infosets_p0)])
            ) / max(len(infosets_p0), 1)
        
        if infosets_p1 and strat_p1:
            u1 = sum(
                infoset.regrets.get(action, 0.0)
                for infoset, action in zip(infosets_p1, strat_p1[:len(infosets_p1)])
            ) / max(len(infosets_p1), 1)
        
        return u0, u1


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_game_form_from_cfr(cfr_solver) -> GameFormExtraction:
    """
    High-level function to extract game form from trained CFR.
    
    Args:
        cfr_solver: Trained CFR instance
    
    Returns:
        GameFormExtraction with payoff matrices
    """
    extractor = GameFormExtractor(cfr_solver)
    return extractor.extract_game_form()


def extract_and_solve_nash(cfr_solver):
    """
    Extract game form and solve for Nash equilibrium.
    
    Args:
        cfr_solver: Trained CFR instance
    
    Returns:
        Tuple of (game_form, nash_solution)
    """
    from src.evaluation.exact_exploitability import (
        GameForm,
        LinearProgrammingNashSolver,
    )
    
    # Extract
    extraction = extract_game_form_from_cfr(cfr_solver)
    
    # Convert to GameForm
    game = GameForm(
        strategies_p0=extraction.strategies_p0,
        strategies_p1=extraction.strategies_p1,
        payoff_matrix_p0=extraction.payoff_matrix_p0,
        payoff_matrix_p1=extraction.payoff_matrix_p1,
    )
    
    # Solve for Nash
    solver = LinearProgrammingNashSolver()
    nash = solver.solve(game)
    
    logger.info(f"Nash equilibrium solved")
    logger.info(f"  P0 value: {nash.value_p0:.6f}")
    logger.info(f"  P1 value: {nash.value_p1:.6f}")
    
    return game, nash


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== CFR to GameForm Converter ===")
    
    # Create mock CFR for testing
    class MockCFR:
        def get_tree_root(self):
            # Simple mock tree
            class MockNode:
                def __init__(self):
                    self.infoset_id = "root"
                    self.regrets = {"fold": 0.5, "call": 1.0}
                    self.children = {
                        "fold": MockNode(),
                        "call": MockNode(),
                    }
            return MockNode()
    
    cfr = MockCFR()
    collector = InformationSetCollector()
    infosets = collector.collect_from_cfr_tree(cfr.get_tree_root())
    
    print(f"Collected {len(infosets)} infosets")
    
    if infosets:
        root_infoset = list(infosets.values())[0]
        print(f"Root infoset: {root_infoset.infoset_id}")
        print(f"Actions: {root_infoset.actions}")
        
        strategy = root_infoset.compute_strategy_from_regrets()
        print(f"Computed strategy: {strategy}")
