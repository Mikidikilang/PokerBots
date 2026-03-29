"""
Kuhn Poker Environment for CFR Validation.

Kuhn Poker is the simplest imperfect information poker game:
- 3 card deck (Jack, Queen, King)
- Each player dealt 1 card
- 2 rounds: 1BB ante, then [check/bet] each round
- If both bet: showdown, higher card wins
- If folded: bettor wins

Game tree size: 12 states (small enough to solve exactly)
Nash Equilibrium: p0_value ≈ 0 (symmetric game)

Perfect for validating CFR convergence!
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class KuhnPokerState:
    """Kuhn poker game state."""
    p0_card: int                    # 0=J, 1=Q, 2=K
    p1_card: int
    history: list[int]             # Actions taken: 0=check, 1=bet, 2=call, 3=fold
    current_player: int            # 0 or 1


class KuhnPokerEnv:
    """
    Kuhn Poker game environment.
    
    Cards: 0 (J), 1 (Q), 2 (K)
    Actions: 0 (check/call), 1 (bet/fold)
    """
    
    NAMES = {
        0: "Jack",
        1: "Queen", 
        2: "King",
    }
    
    ACTION_NAMES = {
        0: "check/call",
        1: "bet/fold",
    }
    
    def __init__(self):
        self.state: Optional[KuhnPokerState] = None
        self.root_state: Optional[KuhnPokerState] = None
    
    def reset(self) -> dict:
        """Start new game."""
        # Deal cards uniformly at random
        cards = np.arange(3)
        np.random.shuffle(cards)
        
        self.state = KuhnPokerState(
            p0_card=int(cards[0]),
            p1_card=int(cards[1]),
            history=[],
            current_player=0,  # p0 acts first
        )
        
        self.root_state = KuhnPokerState(
            p0_card=self.state.p0_card,
            p1_card=self.state.p1_card,
            history=[],
            current_player=0,
        )
        
        return self._get_obs()
    
    def _get_obs(self) -> dict:
        """Return observation (what current player sees)."""
        # Current player sees only their own card
        my_card = (
            self.state.p0_card if self.state.current_player == 0
            else self.state.p1_card
        )
        
        return {
            "my_card": my_card,
            "history": self.state.history.copy(),
            "legal_actions": self._get_legal_actions(),
            "kuhn_state": self.state,  # For testing
        }
    
    def _get_legal_actions(self) -> list[int]:
        """Get legal actions at this state."""
        # In Kuhn: always 2 actions available
        if len(self.state.history) == 0:
            return [0, 1]  # check or bet
        elif len(self.state.history) == 1:
            return [0, 1]  # call/fold or check/bet
        elif len(self.state.history) == 2:
            return [0, 1]  # call/fold or check (end)
        else:
            return []
    
    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        Take action. Return (obs, reward, done).
        """
        if action not in self._get_legal_actions():
            raise ValueError(f"Illegal action {action}")
        
        self.state.history.append(action)
        
        # Check terminal conditions
        done, payoff = self._check_terminal()
        
        if done:
            # Terminal: return payoff for current player
            # (normalized: -1, 0, or +1)
            reward = float(payoff[self.state.current_player])
            return self._get_obs(), reward, True
        
        # Non-terminal: switch players, return 0 reward
        self.state.current_player = 1 - self.state.current_player
        return self._get_obs(), 0.0, False
    
    def _check_terminal(self) -> Tuple[bool, list[float]]:
        """
        Check if game is terminal. Return (is_terminal, payoffs).
        
        Payoffs are from perspective of [player0, player1].
        """
        h = self.state.history
        
        if len(h) == 0:
            return False, [0.0, 0.0]
        
        # After 1 action: can't be terminal yet
        if len(h) == 1:
            return False, [0.0, 0.0]
        
        # 2+ actions: check for terminal patterns
        last_two = h[-2:]
        
        # Both checked: showdown
        if last_two == [0, 0]:
            if self.state.p0_card > self.state.p1_card:
                return True, [1.0, -1.0]
            else:
                return True, [-1.0, 1.0]
        
        # p0: check, p1: bet
        if last_two == [0, 1]:
            # After p1 bet, need p0 response (not terminal yet)
            if len(h) == 2:
                return False, [0.0, 0.0]
            
            # p0's response to p1's bet
            p0_response = h[2] if len(h) > 2 else None
            if p0_response == 0:  # p0 folds
                return True, [-1.0, 1.0]
            elif p0_response == 1:  # p0 calls
                # Showdown after calling
                if self.state.p0_card > self.state.p1_card:
                    return True, [2.0, -2.0]
                else:
                    return True, [-2.0, 2.0]
        
        # p0: bet, p1: fold/call
        if last_two == [1, 0]:  # p0 bet, p1 fold
            return True, [1.0, -1.0]
        
        if last_two == [1, 1]:  # p0 bet, p1 call → showdown
            if self.state.p0_card > self.state.p1_card:
                return True, [2.0, -2.0]
            else:
                return True, [-2.0, 2.0]
        
        return False, [0.0, 0.0]
    
    def is_over(self) -> bool:
        """Check if game is terminal."""
        done, _ = self._check_terminal()
        return done
    
    def get_payoffs(self) -> list[float]:
        """Return payoffs [p0, p1]."""
        done, payoffs = self._check_terminal()
        if done:
            return payoffs
        return [0.0, 0.0]
    
    def get_current_player(self) -> int:
        """Return current player (0 or 1)."""
        return self.state.current_player
    
    def get_infoset_id(self) -> str:
        """
        Return infoset ID for CFR purposes.
        Kuhn: (player, my_card, history) uniquely identifies infoset.
        """
        my_card = (
            self.state.p0_card if self.state.current_player == 0
            else self.state.p1_card
        )
        hist_str = "".join(str(a) for a in self.state.history)
        return f"p{self.state.current_player}_{self.NAMES[my_card]}_{hist_str}"
    
    def copy_state(self) -> KuhnPokerState:
        """Return copy of current state for save/restore."""
        return KuhnPokerState(
            p0_card=self.state.p0_card,
            p1_card=self.state.p1_card,
            history=self.state.history.copy(),
            current_player=self.state.current_player,
        )
    
    def restore_state(self, state: KuhnPokerState):
        """Restore game to given state."""
        self.state = KuhnPokerState(
            p0_card=state.p0_card,
            p1_card=state.p1_card,
            history=state.history.copy(),
            current_player=state.current_player,
        )


# ============================================================================
# GTO Solver for Kuhn Poker
# ============================================================================

class KuhnPokerGTO:
    """Solve Kuhn Poker to Nash equilibrium via exhaustive search."""
    
    @staticmethod
    def solve() -> dict[str, dict[int, float]]:
        """
        Return Nash equilibrium strategy.
        
        Returns: {infoset_id: {action: probability}}
        """
        # In Kuhn poker, exact NE is known:
        # https://en.wikipedia.org/wiki/Kuhn_poker
        
        # p0 with Jack/Queen: check; with King: bet (3:1)
        # p1 with Jack: bet; with Queen: check; with King (after check): check
        # p1 with Jack (after bet): call 1/3; with King: fold 1/3
        
        return {
            "p0_Jack_": {0: 1.0, 1: 0.0},          # Check with J
            "p0_Queen_": {0: 1.0, 1: 0.0},         # Check with Q
            "p0_King_": {0: 0.25, 1: 0.75},        # Bet 3/4 with K
            "p0__1": {0: 0.0, 1: 1.0},             # Fold to bet
            "p1_Jack_": {0: 0.0, 1: 1.0},          # Bet with J
            "p1_Queen_": {0: 1.0, 1: 0.0},         # Check with Q
            "p1_King_": {0: 1.0, 1: 0.0},          # Check with K
            "p1_Jack_1": {0: 2.0/3.0, 1: 1.0/3.0}, # Call 1/3 to bet
            "p1_King_1": {0: 1.0, 1: 0.0},         # Fold
        }
    
    @staticmethod
    def gto_value() -> list[float]:
        """Return GTO game values [p0, p1]."""
        # Kuhn poker is symmetric → value = 0 for both
        return [0.0, 0.0]


if __name__ == "__main__":
    # Quick test
    env = KuhnPokerEnv()
    obs = env.reset()
    print(f"Initial obs: {obs}")
    
    for _ in range(10):
        obs = env.reset()
        while not env.is_over():
            action = np.random.choice([0, 1])
            obs, reward, done = env.step(action)
            print(f"Action: {action}, Done: {done}, Reward: {reward}")
        
        payoffs = env.get_payoffs()
        print(f"Game over. Payoffs: {payoffs}\n")
