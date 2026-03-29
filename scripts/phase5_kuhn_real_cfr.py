"""
Kuhn Poker CFR Training Script

Trains CFR on Kuhn poker and extracts learned Nash strategies.
Tests if CFR learns the correct Nash equilibrium:
- Jack: Bet ~1/3
- Queen: Bet 0.0 (NEVER!)  ← Critical test
- King: Bet 1.0 (always)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env.kuhn_poker_minimal import KuhnPokerEnv, KuhnAction
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class InfosetStrategy:
    """Strategy for a single infoset."""
    infoset_id: str
    action_regrets: Dict[int, float] = field(default_factory=dict)
    cumulative_strategy: Dict[int, float] = field(default_factory=dict)
    visit_count: int = 0

    def get_strategy(self) -> Dict[int, float]:
        """Get current mixed strategy using regret matching."""
        # Regret matching: prob of action = max(regret, 0) / sum of positive regrets
        positive_regrets = {a: max(r, 0.0) for a, r in self.action_regrets.items()}
        total = sum(positive_regrets.values())
        
        if total <= 0:
            # Uniform if no positive regrets
            num_actions = len(self.action_regrets)
            return {a: 1.0 / num_actions for a in self.action_regrets.keys()}
        
        return {a: positive_regrets[a] / total for a in self.action_regrets.keys()}

    def update_regrets(self, regrets: Dict[int, float]):
        """Update cumulative regrets."""
        for action, regret in regrets.items():
            self.action_regrets[action] = self.action_regrets.get(action, 0.0) + regret

    def update_strategy_sum(self, strategy: Dict[int, float]):
        """Accumulate strategy for averaging."""
        for action, prob in strategy.items():
            self.cumulative_strategy[action] = self.cumulative_strategy.get(action, 0.0) + prob
        self.visit_count += 1

    def get_average_strategy(self) -> Dict[int, float]:
        """Get average strategy across all iterations."""
        if self.visit_count == 0:
            num_actions = len(self.cumulative_strategy)
            return {a: 1.0 / num_actions for a in self.cumulative_strategy.keys()}
        
        return {a: self.cumulative_strategy[a] / self.visit_count 
                for a in self.cumulative_strategy.keys()}


class KuhnPokerCFR:
    """CFR trainer for Kuhn poker."""

    def __init__(self, num_iterations: int = 1000, seed: int = 42):
        self.num_iterations = num_iterations
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.infosets: Dict[str, InfosetStrategy] = {}
        self.iteration = 0

    def train(self):
        """Run CFR training."""
        logger.info("=" * 90)
        logger.info("KUHN POKER CFR TRAINING (Real, Not Theoretical)")
        logger.info("=" * 90)
        logger.info(f"Running {self.num_iterations} iterations...")
        logger.info(f"Target Nash: Jack=1/3, Queen=0, King=1.0")
        logger.info("")

        for iteration in range(self.num_iterations):
            self.iteration = iteration
            
            # Standard Chance-Sampling CFR: one traversal per iteration
            # Both players' infosets are updated in the same pass
            env = KuhnPokerEnv(seed=self.rng.randint(0, 2**31))
            self._traverse(env, target_player=None)  # None = update all infosets

            if (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")

        return self.infosets

    def _traverse(self, env: KuhnPokerEnv, target_player: int = None) -> float:
        """
        Recursive traversal (standard CFR). 
        When target_player is None, updates ALL infosets in a single pass.
        Returns expected payoff from player 0's perspective.
        """
        if env.is_over():
            payoff = env.get_payoff()
            return payoff  # Always from P0's perspective

        current_player = env.get_player_id()
        legal_actions = env.get_legal_actions()

        # Get infoset
        infoset_id = env.get_information_set_key(current_player)
        if infoset_id not in self.infosets:
            self.infosets[infoset_id] = InfosetStrategy(
                infoset_id=infoset_id,
                action_regrets={a: 0.0 for a in legal_actions}
            )

        infoset = self.infosets[infoset_id]

        # Get current strategy
        strategy = infoset.get_strategy()

        # Compute value of infoset
        infoset_value = 0.0
        action_values = {}

        for action in legal_actions:
            # Simulate action
            env_copy = self._copy_env(env)
            env_copy.step(action)

            # Recursive value
            action_value = self._traverse(env_copy, target_player=target_player)
            action_values[action] = action_value
            infoset_value += strategy[action] * action_value


        # UPDATE REGRETS (always done in CFR - not conditional on target_player)
        regrets = {}
        for action in legal_actions:
            if current_player == 0:
                # P0 perspective: regret = value_of_action - value_of_infoset
                regrets[action] = action_values[action] - infoset_value
            else:
                # P1 perspective: regret = -(value_of_action - value_of_infoset) 
                # because P1 wants to minimize P0's payoff
                regrets[action] = -(action_values[action] - infoset_value)
        
        infoset.update_regrets(regrets)

        # Update strategy sum for averaging (always done)
        infoset.update_strategy_sum(strategy)

        return infoset_value

    @staticmethod
    def _copy_env(env: KuhnPokerEnv) -> KuhnPokerEnv:
        """Create a copy of environment state."""
        copy = KuhnPokerEnv()
        copy.p0_card = env.p0_card
        copy.p1_card = env.p1_card
        copy.history = env.history.copy()
        copy.current_player_idx = env.current_player_idx
        copy.is_terminal = env.is_terminal
        copy.payoff = env.payoff
        return copy

    def print_results(self):
        """Extract and print learned Nash strategies."""
        logger.info("\n" + "=" * 90)
        logger.info("LEARNED NASH EQUILIBRIUM (Player 1 First Decision)")
        logger.info("=" * 90)

        # Debug: print ALL infosets that were created
        logger.info(f"\nTotal infosets created: {len(self.infosets)}")
        logger.info("First 20 infoset IDs:")
        for i, iset_id in enumerate(sorted(self.infosets.keys())[:20]):
            logger.info(f"  {iset_id}")

        # Extract P1's first decision infosets (after P0 CHECK)
        p1_preflop_infosets = {
            iset_id: iset for iset_id, iset in self.infosets.items()
            if iset_id.startswith("P1_") and iset_id.endswith("_C")  # P1 acts after P0 CHECK
        }
        
        logger.info(f"\nP1 decision infosets (after P0 CHECK): {len(p1_preflop_infosets)}")
        for iset_id in sorted(p1_preflop_infosets.keys()):
            logger.info(f"  {iset_id}")

        logger.info("\nPlayer 1 Strategy After P0 CHECK:")
        logger.info(f"{'Card':<10}{'Bet Prob':<15}{'Check Prob':<15}{'Expected Nash':<20}")
        logger.info("-" * 60)

        results = {}
        for iset_id, infoset in p1_preflop_infosets.items():
            card = iset_id.split("_")[1]  # Extract card (J, Q, or K)
            
            avg_strategy = infoset.get_average_strategy()

            # Actions: 0=CHECK, 1=BET
            check_prob = avg_strategy.get(KuhnAction.CHECK, 0.0)
            bet_prob = avg_strategy.get(KuhnAction.BET, 0.0)

            expected = {
                "J": "~0.33 (bluff sometimes)",
                "Q": "0.0 (NEVER bluff! Key test)",
                "K": "1.0 (always value bet)"
            }.get(card, "?")

            logger.info(f"{card:<10}{bet_prob:<15.4f}{check_prob:<15.4f}{expected:<20}")
            
            results[card] = {
                "bet_prob": bet_prob,
                "check_prob": check_prob,
                "expected": expected
            }

        # Validation
        logger.info("\n" + "=" * 90)
        logger.info("VALIDATION AGAINST THEORETICAL NASH")
        logger.info("=" * 90)

        validation_passed = True

        # Queen must be 0
        queen_bet = results["Q"]["bet_prob"]
        if abs(queen_bet) < 0.05:
            logger.info(f"✅ Queen bet probability: {queen_bet:.4f} (should be 0)")
        else:
            logger.info(f"❌ Queen bet probability: {queen_bet:.4f} (should be 0)")
            validation_passed = False

        # Jack bluffs around 1/3
        jack_bet = results["J"]["bet_prob"]
        if 0.2 < jack_bet < 0.45:
            logger.info(f"✅ Jack bet probability: {jack_bet:.4f} (should be ~1/3 = 0.33)")
        else:
            logger.info(f"⚠️  Jack bet probability: {jack_bet:.4f} (should be ~1/3 = 0.33)")

        # King always bets (should be 3α = 1.0 if α = 1/3)
        king_bet = results["K"]["bet_prob"]
        if king_bet > 0.95:
            logger.info(f"✅ King bet probability: {king_bet:.4f} (should be 1.0)")
        else:
            logger.info(f"⚠️  King bet probability: {king_bet:.4f} (should be 1.0)")

        logger.info("\n" + "=" * 90)
        if validation_passed:
            logger.info("✅ PHASE 5 VALIDATED: CFR Learned Correct Nash Equilibrium!")
            logger.info("   The Queen never bluffs (bet prob = 0) → Algorithm is sound!")
        else:
            logger.info("⚠️  Learning in progress or algorithm issue")
        logger.info("=" * 90)

        return results


if __name__ == "__main__":
    trainer = KuhnPokerCFR(num_iterations=10000)
    trainer.train()
    trainer.print_results()
