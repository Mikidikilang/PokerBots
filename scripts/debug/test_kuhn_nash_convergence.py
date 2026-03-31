#!/usr/bin/env python3
"""
PHASE 5: Kuhn Poker Nash Equilibrium Validation
========================================

Validates that CFR+ converges to theoretical Nash equilibrium on Kuhn (3-card) poker.

THEORETICAL NASH EQUILIBRIUM (Kuhn, 1950):
    Jack (J):  BET 33.33% ± 2% (CHECK 66.67%)
    Queen (Q): BET 0% (always CHECK)
    King (K):  BET 100% (always BET)
    
    Value: -1/18 for first player (P0), +1/18 for second player (P1)

CONVERGENCE TARGETS (10k iterations):
    Jack:  BET 28-38% (±5%)
    Queen: BET 0-5% (nearly never bet)
    King:  BET 95-100% (nearly always bet)

CRITICAL FIX (March 31, 2026):
    Sign flip in Player 1's terminal payoff.
    Before: Player 1 payoff inverted
    After:  Properly negated for zero-sum game
    
    Formula: return -payoff[0] when player_to_update == 1
"""

import logging
from typing import Dict, Tuple
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# KUHN POKER IMPLEMENTATION (Self-contained for validation)
# ============================================================================

class KuhnPokerEnv:
    """Minimal Kuhn (3-card) poker environment."""
    
    CARDS = ["Jack", "Queen", "King"]  # Indices 0, 1, 2
    ACTIONS = ["check", "bet"]
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Deal two cards from deck of 3."""
        self.deck = list(range(3))  # [0, 1, 2]
        np.random.shuffle(self.deck)
        
        self.p0_card = self.deck[0]
        self.p1_card = self.deck[1]
        self.history = []
        self.is_terminal = False
        self.payoffs = None
        
        return {
            "p0_card": self.p0_card,
            "p1_card": self.p1_card,
            "history": tuple(self.history),
        }
    
    def step(self, action: int, current_player: int) -> Tuple[Dict, float]:
        """
        Execute action.
        
        Args:
            action: 0=check, 1=bet
            current_player: 0 or 1
            
        Returns:
            (next_state, immediate_reward)
        """
        assert action in [0, 1], f"Invalid action: {action}"
        assert current_player in [0, 1], f"Invalid player: {current_player}"
        assert not self.is_terminal, "Game is over"
        
        action_name = self.ACTIONS[action]
        self.history.append(action_name)
        
        # Check if game is terminal
        if self._check_terminal():
            self.is_terminal = True
            self._compute_payoffs()
        
        next_state = {
            "p0_card": self.p0_card,
            "p1_card": self.p1_card,
            "history": tuple(self.history),
        }
        
        return next_state, 0.0  # Immediate reward not used in CFR
    
    def _check_terminal(self) -> bool:
        """
        Determine if game has reached terminal state.
        
        Kuhn Poker Game Tree:
            P0 acts:
                CHECK → P1 acts:
                    CHECK → Terminal (showdown)
                    BET → P0 acts:
                        CHECK (fold) → Terminal
                        BET (call) → Terminal (showdown)
                BET → P1 acts:
                    CHECK (fold) → Terminal
                    BET (raise) → Terminal (showdown)
        """
        h = self.history
        
        # After P1's first action (h length 2): game is terminal ONLY if:
        #   - P0 checked and P1 checked (showdown)
        #   - P0 bet and P1 took action (fold or raise → showdown)
        if len(h) == 2:
            # P0 BET, P1 responds → always terminal (P1 folds or calls)
            if h[0] == "bet":
                return True
            # P0 CHECK, P1 CHECK → terminal (showdown)
            if h[0] == "check" and h[1] == "check":
                return True
            # P0 CHECK, P1 BET → not terminal yet (P0 must respond)
            if h[0] == "check" and h[1] == "bet":
                return False
        
        # After P0's second action (h length 3): game is always terminal
        if len(h) == 3:
            return True
        
        # Games shouldn't exceed 3 actions in Kuhn poker
        if len(h) > 3:
            logger.warning(f"Kuhn poker exceeded maximum actions: {h}")
            return True
        
        return False
    
    def _compute_payoffs(self):
        """
        Compute payoffs after game ends.
        
        In Kuhn poker (zero-sum):
            - If P0 card > P1 card: P0 wins the pot
            - If P1 card > P0 card: P1 wins the pot
            - Pot size:
                - "check, check": 1 chip each → winner gets 1
                - "bet, check" (fold): P0 wins 1
                - "bet, bet" (showdown): winner gets 2
                - "check, bet, check" (fold): P1 wins 1
                - "check, bet, bet" (showdown): winner gets 2
        """
        h = self.history
        p0_card_rank = self.p0_card  # 0=Jack, 1=Queen, 2=King (ascending)
        p1_card_rank = self.p1_card
        p0_wins = p0_card_rank > p1_card_rank
        
        # Case 1: P0 CHECK, P1 CHECK → Showdown (1 chip pot)
        if h == ["check", "check"]:
            self.payoffs = [1, -1] if p0_wins else [-1, 1]
        
        # Case 2: P0 BET, P1 CHECK → P1 folds, P0 wins 1
        elif h == ["bet", "check"]:
            self.payoffs = [1, -1]
        
        # Case 3: P0 BET, P1 BET → Showdown (2 chips pot)
        elif h == ["bet", "bet"]:
            self.payoffs = [2, -2] if p0_wins else [-2, 2]
        
        # Case 4: P0 CHECK, P1 BET, P0 CHECK → P0 folds, P1 wins 1
        elif h == ["check", "bet", "check"]:
            self.payoffs = [-1, 1]
        
        # Case 5: P0 CHECK, P1 BET, P0 BET → Showdown (2 chips pot)
        elif h == ["check", "bet", "bet"]:
            self.payoffs = [2, -2] if p0_wins else [-2, 2]
        
        else:
            logger.warning(f"Unknown history for payoff: {h}")
            self.payoffs = [0, 0]
    
    def get_legal_actions(self) -> list:
        """Get legal actions for the current player."""
        # In Kuhn poker, both check and bet are always legal
        return [0, 1]  # 0=check, 1=bet
    
    def get_payoff(self, player: int) -> float:
        """Get payoff for a player (from their perspective)."""
        assert player in [0, 1]
        if self.payoffs is None:
            return 0.0
        return float(self.payoffs[player])


# ============================================================================
# CFR+ SOLVER FOR KUHN POKER
# ============================================================================

class KuhnCFRSolver:
    """Chance-sampling CFR+ for Kuhn poker."""
    
    CARDS = ["Jack", "Queen", "King"]
    ACTIONS = ["check", "bet"]
    
    def __init__(self):
        self.infosets: Dict[str, Dict] = {}  # infoset_id -> {actions, regrets, ...}
        self.iteration = 0
        self.cfr_plus = False  # Disable CFR+ clamping to see if that helps
    
    def get_infoset_id(self, player: int, card: int, history: tuple) -> str:
        """Create canonical infoset identifier."""
        return f"P{player}_{self.CARDS[card]}_{','.join(history) if history else 'root'}"
    
    def get_infoset(self, player: int, card: int, history: tuple):
        """Get or create infoset."""
        iid = self.get_infoset_id(player, card, history)
        
        if iid not in self.infosets:
            self.infosets[iid] = {
                "player": player,
                "card": self.CARDS[card],
                "history": history,
                "cumulative_regrets": {0: 0.0, 1: 0.0},  # 0=check, 1=bet
                "cumulative_strategy": {0: 0.0, 1: 0.0},  # For averaging
                "visit_count": 0,
            }
        
        return self.infosets[iid]
    
    def external_sampling_cfr(
        self,
        p0_card: int,
        p1_card: int,
        history: tuple,
        player_to_update: int,
        reach_p0: float = 1.0,
        reach_p1: float = 1.0,
    ) -> float:
        """
        External sampling CFR traversal using immutable game state.
        
        Args:
            p0_card: Player 0's card (0=Jack, 1=Queen, 2=King)
            p1_card: Player 1's card
            history: Tuple of actions taken so far (each "check" or "bet")
            player_to_update: Which player's regrets to update (0 or 1)
            reach_p0: Reach probability for Player 0
            reach_p1: Reach probability for Player 1
            
        Returns:
            Value of the node from player_to_update's perspective.
        """
        
        # ─────────────────────────────────────────────────────────────
        # TERMINAL NODE: Extract payoff with zero-sum correction
        # ─────────────────────────────────────────────────────────────
        
        # Check if game is terminal
        is_terminal = False
        if len(history) == 0:
            is_terminal = False
        elif len(history) == 1:
            is_terminal = False  # P1 must respond to P0's action
        elif len(history) == 2:
            # Terminal if: P0 bet, P1 responded OR P0 check && P1 check
            if history[0] == "bet":
                is_terminal = True
            elif history[0] == "check" and history[1] == "check":
                is_terminal = True
            else:  # "check", "bet"
                is_terminal = False  # P0 must respond
        elif len(history) == 3:
            is_terminal = True  # P0 responded to P1's bet
        else:
            is_terminal = True  # Safety val
        
        if is_terminal:
            # Compute payoff from game outcome
            p0_card_rank = p0_card
            p1_card_rank = p1_card
            p0_wins = p0_card_rank > p1_card_rank
            
            # Determine payoff based on history
            if history == ("check", "check"):
                payoff_p0 = 1 if p0_wins else -1
            elif history == ("bet", "check"):
                payoff_p0 = 1  # P1 folds
            elif history == ("bet", "bet"):
                payoff_p0 = 2 if p0_wins else -2
            elif history == ("check", "bet", "check"):
                payoff_p0 = -1  # P0 folds
            elif history == ("check", "bet", "bet"):
                payoff_p0 = 2 if p0_wins else -2
            else:
                payoff_p0 = 0
           
            # ★★★ FIX A: ZERO-SUM PAYOFF PERSPECTIVE ★★★
            # CRITICAL: In zero-sum games, Player 1's payoff = -Player 0's payoff
            # CFR requires tracking counterfactual values from EACH PLAYER'S perspective.
            # When player_to_update=1, we must return value from Player 1's perspective.
            #
            # CORRECT FORMULA (For zero-sum games):
            #   payoff_for_p1 = -payoff_for_p0
            #
            # This ensures regrets are computed correctly for both players:
            #   For P0: regret(a) = value_from_p0's_perspective(a) - avg_value
            #   For P1: regret(a) = value_from_p1's_perspective(a) - avg_value
            if player_to_update == 0:
                return float(payoff_p0)
            else:  # player_to_update == 1
                # Return from Player 1's perspective (negated for zero-sum)
                return float(-payoff_p0)
        
        # ─────────────────────────────────────────────────────────────
        # NON-TERMINAL: Get current player and infoset
        # ─────────────────────────────────────────────────────────────
        
        current_player = len(history) % 2  # P0 if even, P1 if odd
        infoset = self.get_infoset(current_player, [p0_card, p1_card][current_player], history)
        strategy = self._compute_strategy(infoset)
        
        # ─────────────────────────────────────────────────────────────
        # CASE A: Updating this player → Evaluate ALL actions
        # ─────────────────────────────────────────────────────────────
        
        if current_player == player_to_update:
            action_values = {}
            
            for action in [0, 1]:  # 0=check, 1=bet
                action_name = self.ACTIONS[action]
                new_history = history + (action_name,)
                
                new_reach_p0 = reach_p0 * (strategy[action] if current_player == 0 else 1.0)
                new_reach_p1 = reach_p1 * (strategy[action] if current_player == 1 else 1.0)
                
                value = self.external_sampling_cfr(
                    p0_card, p1_card, new_history,
                    player_to_update,
                    new_reach_p0, new_reach_p1
                )
                action_values[action] = value
            
            # Compute value and regrets
            avg_value = sum(strategy[a] * action_values[a] for a in [0, 1])
            
            # Opposing reach probability
            opposing_reach = reach_p1 if current_player == 0 else reach_p0
            
            # Accumulate regrets
            for action in [0, 1]:
                regret = action_values[action] - avg_value
                scaled_regret = regret * opposing_reach
                
                # Update cumulative regret
                if self.cfr_plus:
                    # CFR+: clamp negative regrets to 0 each iteration
                    infoset["cumulative_regrets"][action] = max(
                        infoset["cumulative_regrets"][action] + scaled_regret,
                        0.0
                    )
                else:
                    # Standard CFR
                    infoset["cumulative_regrets"][action] += scaled_regret
            
            return avg_value
        
        # ─────────────────────────────────────────────────────────────
        # CASE B: Opponent's turn → Sample ONE action
        # ─────────────────────────────────────────────────────────────
        
        else:
            # Sample from opponent's strategy
            action = np.random.choice([0, 1], p=[strategy[0], strategy[1]])
            action_name = self.ACTIONS[action]
            new_history = history + (action_name,)
            
            action_prob = strategy[action]
            new_reach_p0 = reach_p0 * (action_prob if current_player == 0 else 1.0)
            new_reach_p1 = reach_p1 * (action_prob if current_player == 1 else 1.0)
            
            value = self.external_sampling_cfr(
                p0_card, p1_card, new_history,
                player_to_update,
                new_reach_p0, new_reach_p1
            )
            
            return value
    
    def _compute_strategy(self, infoset: Dict) -> Dict[int, float]:
        """Compute regret-matched strategy: σ(a) = max(R(a), 0) / Σ max(R(a'), 0)"""
        cumulative_regrets = infoset["cumulative_regrets"]
        
        positive_regrets = {
            a: max(cumulative_regrets[a], 0.0) for a in [0, 1]
        }
        
        total = sum(positive_regrets.values())
        
        if total <= 1e-8:
            return {0: 0.5, 1: 0.5}  # Uniform if no regrets yet
        
        return {
            a: positive_regrets[a] / total for a in [0, 1]
        }
    
    def run_iteration(self):
        """Run one full MCCFR iteration (P0 traversal + P1 traversal)."""
        
        # Deal two random cards
        deck = list(range(3))
        np.random.shuffle(deck)
        p0_card = deck[0]
        p1_card = deck[1]
        
        # Update Player 0's regrets
        self.external_sampling_cfr(p0_card, p1_card, (), player_to_update=0)
        
        # Reshuffle and update Player 1's regrets
        np.random.shuffle(deck)
        p0_card = deck[0]
        p1_card = deck[1]
        self.external_sampling_cfr(p0_card, p1_card, (), player_to_update=1)
        
        # Increment iteration and accumulate strategies for averaging
        self.iteration += 1
        for infoset in self.infosets.values():
            strategy = self._compute_strategy(infoset)
            infoset["cumulative_strategy"][0] += strategy[0]
            infoset["cumulative_strategy"][1] += strategy[1]
            infoset["visit_count"] += 1
    
    def get_average_strategy(self, infoset_id: str) -> Dict[int, float]:
        """Get average strategy for an infoset (guaranteed to converge to Nash)."""
        infoset = self.infosets.get(infoset_id)
        if not infoset or infoset["visit_count"] == 0:
            return {0: 0.5, 1: 0.5}
        
        total = infoset["cumulative_strategy"][0] + infoset["cumulative_strategy"][1]
        if total <= 1e-8:
            return {0: 0.5, 1: 0.5}
        
        return {
            0: infoset["cumulative_strategy"][0] / total,
            1: infoset["cumulative_strategy"][1] / total,
        }
    
    def get_current_strategies(self) -> Dict[str, Dict[int, float]]:
        """Get current regret-matched strategies for all infosets."""
        return {iid: self._compute_strategy(infoset) for iid, infoset in self.infosets.items()}


# ============================================================================
# VALIDATION & TESTING
# ============================================================================

def test_kuhn_convergence(num_iterations: int = 10000):
    """Run CFR+ on Kuhn poker and validate convergence to Nash equilibrium."""
    
    logger.info("=" * 70)
    logger.info("PHASE 5: KUHN POKER NASH EQUILIBRIUM VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Running {num_iterations} CFR iterations...")
    logger.info("")
    
    solver = KuhnCFRSolver()
    
    # Run iterations
    for i in range(num_iterations):
        solver.run_iteration()
        
        if (i + 1) % 2000 == 0:
            logger.info(f"Completed {i + 1} / {num_iterations} iterations")
    
    # Extract final strategies
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL LEARNED STRATEGIES (Average over all iterations)")
    logger.info("=" * 70)
    
    nash_thresholds = {
        "Jack": (0.28, 0.38),         # ±5% of 33.33%
        "Queen": (0.0, 0.05),         # Should be ~0%
        "King": (0.95, 1.0),          # Should be ~100%
    }
    
    results = {}
    all_pass = True
    
    for card_idx, card_name in enumerate(["Jack", "Queen", "King"]):
        infoset_id = f"P0_{card_name}_"  # P0's initial infoset
        avg_strategy = solver.get_average_strategy(infoset_id)
        
        bet_prob = avg_strategy[1]  # Action 1 = bet
        results[card_name] = bet_prob
        
        (low, high) = nash_thresholds[card_name]
        in_range = low <= bet_prob <= high
        status = "✅ PASS" if in_range else "❌ FAIL"
        
        logger.info(f"{card_name:8s}: BET {bet_prob*100:6.2f}%    Range [{low*100:5.1f}%, {high*100:5.1f}%]   {status}")
        
        if not in_range:
            all_pass = False
    
    # Validate ALL cards converge correctly
    logger.info("")
    logger.info("=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)
    
    assertions_passed = 0
    assertions_total = 3
    
    # Assertion 1: Queen should NOT bet (< 5%)
    try:
        assert results["Queen"] < 0.05, f"Queen BET {results['Queen']*100:.2f}% but should be <5%"
        logger.info("✅ ASSERTION 1: Queen converged to NO BET")
        assertions_passed += 1
    except AssertionError as e:
        logger.info(f"❌ ASSERTION 1 FAILED: {e}")
    
    # Assertion 2: King should ALWAYS bet (> 95%)
    try:
        assert results["King"] > 0.95, f"King BET {results['King']*100:.2f}% but should be >95%"
        logger.info("✅ ASSERTION 2: King converged to ALWAYS BET")
        assertions_passed += 1
    except AssertionError as e:
        logger.info(f"❌ ASSERTION 2 FAILED: {e}")
    
    # Assertion 3: Jack should be in mixed strategy (28-38%)
    try:
        assert 0.28 <= results["Jack"] <= 0.38, f"Jack BET {results['Jack']*100:.2f}% but should be 28-38%"
        logger.info("✅ ASSERTION 3: Jack converged to mixed strategy (28-38%)")
        assertions_passed += 1
    except AssertionError as e:
        logger.info(f"❌ ASSERTION 3 FAILED: {e}")
    
    logger.info("")
    logger.info(f"SUMMARY: {assertions_passed}/{assertions_total} assertions passed")
    logger.info("=" * 70)
    
    if assertions_passed == assertions_total:
        logger.info("✅ CONVERGENCE TEST PASSED - Sign flip fixed successfully!")
        return True
    else:
        logger.info("❌ CONVERGENCE TEST FAILED")
        return False


if __name__ == "__main__":
    success = test_kuhn_convergence(num_iterations=10000)
    exit(0 if success else 1)
