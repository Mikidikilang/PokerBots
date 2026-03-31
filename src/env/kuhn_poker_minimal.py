"""
Minimal Kuhn Poker Environment (duck-types RLCard interface)

Only implements what CFREngine, MCCFRTraversal, and EnvStateManager actually call.

Nash Equilibrium (correct version - Player 1 preflop):
- Jack:   Bet with probability α (e.g., 1/3)
- Queen:  Bet with probability 0 (NEVER!)  ← Key test
- King:   Bet with probability 3α (e.g., 1.0)

Player 0 (dealer): Can check/bet preflop
Player 1 (BB): Responds to check or can fold/call vs bet
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from enum import IntEnum


class KuhnAction(IntEnum):
    """Action enum for Kuhn poker."""
    CHECK = 0
    BET = 1
    FOLD = 2
    CALL = 3


class KuhnPokerEnv:
    """Minimal Kuhn poker environment compatible with CFR engine."""

    CARDS = ("J", "Q", "K")  # Jack=0, Queen=1, King=2
    PAYOFF_UNIT = 1  # Ante = 1 chip

    def __init__(self, num_players: int = 2, seed: int = None):
        """Initialize Kuhn poker game."""
        self.num_players = num_players
        self.cards = [0, 1, 2]  # J, Q, K
        self.rng = np.random.RandomState(seed)
        
        # Game state
        self.p0_card: int = None
        self.p1_card: int = None
        self.history: List[str] = []
        self.current_player_idx: int = 0
        self.is_terminal: bool = False
        self.payoff: float = 0.0
        
        self.reset()

    def reset(self):
        """Reset game to initial state."""
        # Shuffle deck and deal
        self.rng.shuffle(self.cards)
        self.p0_card = self.cards[0]
        self.p1_card = self.cards[1]
        self.history = []
        self.current_player_idx = 0  # Player 0 acts first (dealer)
        self.is_terminal = False
        self.payoff = 0.0
        return self.get_state()

    def get_player_id(self) -> int:
        """Get current player (0 or 1)."""
        return self.current_player_idx

    def get_legal_actions(self) -> List[int]:
        """Return legal actions for current player."""
        if self.is_terminal:
            return []

        history_str = "".join(self.history)
        
        # No cards dealt - shouldn't happen
        if self.p0_card is None:
            return []

        # Kuhn poker action rules:
        if len(history_str) == 0:  # P0's turn (preflop)
            return [KuhnAction.CHECK, KuhnAction.BET]
        
        elif len(history_str) == 1:  # P1's turn
            if history_str[0] == "C":  # P0 checked
                return [KuhnAction.CHECK, KuhnAction.BET]
            elif history_str[0] == "B":  # P0 bet
                return [KuhnAction.FOLD, KuhnAction.CALL]
        
        elif len(history_str) == 2:  # P0's turn (postflop, only happens after CC or BC)
            if history_str == "CC":
                # Showdown - not relevant
                return []
            elif history_str == "BC":  # P0 bet, P1 called
                # Game ends in showdown
                return []
        
        return []

    def step(self, action: int) -> Tuple[bool, Dict]:
        """
        Execute action. Returns (is_terminal, next_state_dict).
        
        action: KuhnAction enum value
        """
        if self.is_terminal:
            raise RuntimeError("Game is already terminal")

        action_str = {KuhnAction.CHECK: "C", KuhnAction.BET: "B", 
                      KuhnAction.FOLD: "F", KuhnAction.CALL: "K"}[action]
        
        self.history.append(action_str)
        history_str = "".join(self.history)

        # Check for terminal states
        self.is_terminal, self.payoff = self._check_terminal(history_str)

        if not self.is_terminal:
            # Switch to next player
            self.current_player_idx = 1 - self.current_player_idx

        return self.is_terminal, self.get_state()

    def _check_terminal(self, history: str) -> Tuple[bool, float]:
        """Check if game is terminal and compute payoff (from P0's perspective)."""
        
        # Valid terminal histories:
        # CC: P0 check, P1 check -> showdown
        # CBF: P0 check, P1 bet, P0 fold -> P1 wins +1  
        # CBK: P0 check, P1 bet, P0 call -> showdown
        # BF: P0 bet, P1 fold -> P0 wins +1
        # BK: P0 bet, P1 call -> showdown
        
        if history == "CC":  # Both checked -> showdown
            if self.p0_card > self.p1_card:
                return True, self.PAYOFF_UNIT  # +1
            elif self.p0_card < self.p1_card:
                return True, -self.PAYOFF_UNIT  # -1
            else:
                return True, 0.0  # Tie
        
        elif history == "CBF":  # P0 checked, P1 bet, P0 folded -> P1 wins
            return True, -self.PAYOFF_UNIT  # P0 loses 1 to P1
        
        elif history == "CBK":  # P0 checked, P1 bet, P0 called -> showdown
            if self.p0_card > self.p1_card:
                return True, 2 * self.PAYOFF_UNIT  # Win 2
            elif self.p0_card < self.p1_card:
                return True, -2 * self.PAYOFF_UNIT  # Lose 2
            else:
                return True, 0.0  # Tie
        
        elif history == "BF":  # P0 bet, P1 folded -> P0 wins
            return True, self.PAYOFF_UNIT  # P0 wins 1
        
        elif history == "BK":  # P0 bet, P1 called -> showdown
            if self.p0_card > self.p1_card:
                return True, 2 * self.PAYOFF_UNIT  # Win 2
            elif self.p0_card < self.p1_card:
                return True, -2 * self.PAYOFF_UNIT  # Lose 2
            else:
                return True, 0.0  # Tie
        
        # Non-terminal states
        elif history == "C":  # P0 checked, waiting for P1
            return False, 0.0
        elif history == "CB":  # P0 checked, P1 bet, waiting for P0 fold/call
            return False, 0.0
        elif history == "B":  # P0 bet, waiting for P1 fold/call
            return False, 0.0
        
        return False, 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return game state dict (minimal)."""
        return {
            "p0_card": self.p0_card,
            "p1_card": self.p1_card,
            "p0_can_see_p1_card": False,  # Imperfect info
            "p1_can_see_p0_card": False,
            "history": "".join(self.history),
            "current_player": self.current_player_idx,
            "is_terminal": self.is_terminal,
            "payoff": self.payoff,
        }

    def get_payoff(self) -> float:
        """Get payoff (from P0's perspective)."""
        return self.payoff

    def is_over(self) -> bool:
        """Check if game is over."""
        return self.is_terminal

    def get_information_set_key(self, player: int) -> str:
        """
        Return infoset key: (player, card, history).
        This is what CFR uses to track strategies.
        """
        card = [self.p0_card, self.p1_card][player]
        card_str = self.CARDS[card]
        history_str = "".join(self.history)
        return f"P{player}_{card_str}_{history_str}"

    def __str__(self) -> str:
        """String representation."""
        history = " ".join(
            [{"C": "Check", "B": "Bet", "F": "Fold", "K": "Call"}.get(a, a) 
             for a in self.history]
        )
        return (f"KuhnPoker | P0: {self.CARDS[self.p0_card]} | "
                f"P1: {self.CARDS[self.p1_card]} | "
                f"History: {history} | "
                f"Terminal: {self.is_terminal}")
