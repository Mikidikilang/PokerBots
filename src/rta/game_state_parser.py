"""
Live Game State Parser (game_state_parser.py).

Converts real-time poker events (from screen-scraping, APIs, or log files)
into the normalized state dictionary expected by ObservationBuilder.build().

Key differences from training:
1. Must track state incrementally (events arrive one at a time).
2. Must handle missing/incomplete events gracefully (for robust scraping).
3. Must compute pot_before and spr_before for betting history (critical for RTA).
4. Must infer legal actions from rlcard-style raw observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GameStateTracker:
    """Maintains accumulated game state across events."""
    
    num_players:       int = 6
    big_blind:         float = 2.0
    initial_stack_bb:  float = 200.0
    
    # Current hand state
    current_street:    int = 0  # 0=preflop, 1=flop, 2=turn, 3=river
    pot:               float = 0.0
    my_position:       int = 0
    my_hand:           list[str] = field(default_factory=list)  # ["AS", "KS"]
    public_cards:      list[str] = field(default_factory=list)  # ["2H", "3D", "4C"]
    
    # Chip stacks (one per player, indexed by seat)
    stacks:            list[float] = field(default_factory=list)
    stakes:            list[float] = field(default_factory=list)  # amount each player has in pot
    
    # Betting history (accumulated actions this hand)
    betting_history:   list[dict[str, Any]] = field(default_factory=list)
    
    # State machine: ensure "action" events only occur after "deal"
    hand_started:      bool = False  # True after "hand_start" or "deal" event
    cards_dealt:       bool = False  # True after "deal" event
    
    # Cached computed values
    _legal_actions_cache: list[int] | None = None
    _amount_to_call_cache: float = 0.0
    
    def __post_init__(self) -> None:
        """Ensure arrays are properly sized."""
        if not self.stacks:
            self.stacks = [self.initial_stack_bb * self.big_blind] * self.num_players
        if not self.stakes:
            self.stakes = [0.0] * self.num_players
        if len(self.stacks) < self.num_players:
            self.stacks.extend([self.initial_stack_bb * self.big_blind] * (self.num_players - len(self.stacks)))
        if len(self.stakes) < self.num_players:
            self.stakes.extend([0.0] * (self.num_players - len(self.stakes)))
    
    def reset_hand(self) -> None:
        """Reset for a new hand."""
        self.current_street = 0
        self.pot = 0.0
        self.my_hand = []
        self.public_cards = []
        self.stakes = [0.0] * self.num_players
        self.betting_history = []
        self._legal_actions_cache = None
        self.hand_started = False
        self.cards_dealt = False
    
    def to_raw_state(self) -> dict[str, Any]:
        """Export current state as raw observation dict."""
        my_chips = self.stacks[self.my_position]
        opponent_chips = [
            self.stacks[i] for i in range(self.num_players)
            if i != self.my_position
        ]
        max_stake = max(self.stakes) if self.stakes else 0.0
        my_stake = self.stakes[self.my_position]
        amount_to_call = max(0.0, max_stake - my_stake)
        
        return {
            "hand": self.my_hand,
            "public_cards": self.public_cards,
            "pot": self.pot,
            "my_chips": my_chips,
            "big_blind": self.big_blind,
            "amount_to_call": amount_to_call,
            "position": self.my_position,
            "legal_actions": self._compute_legal_actions(),
            "opponent_chips": opponent_chips,
            "betting_history": list(self.betting_history),  # copy
            "min_raise": self.big_blind,  # fallback; should be computed from rlcard context
        }


class LiveGameStateBuilder:
    """Parses real-time poker events and maintains coherent game state.
    
    This class is designed to work with:
    - Screen-scraping APIs (PokerStars, 888poker, etc.)
    - RLCard game logs (for testing)
    - Custom event streams (JSON, protobuf, etc.)
    
    Key responsibility:
    - Accumulate `pot_before` and `spr_before` in betting_history
      (required for proper RTA inference with betting history features).
    
    Example usage:
        builder = LiveGameStateBuilder(num_players=6, big_blind=2.0)
        builder.process_event({"type": "deal", "hand": ["AS", "KS"]})
        builder.process_event({"type": "action", "player": 2, "action": "raise", "amount": 100})
        
        state = builder.get_state()
        # state now has pot_before and spr_before in betting_history
    """
    
    def __init__(
        self,
        num_players: int = 6,
        big_blind: float = 2.0,
        initial_stack_bb: float = 200.0,
    ) -> None:
        """Initialize the parser.
        
        Args:
            num_players: Number of players in the game.
            big_blind: Big blind amount in chips.
            initial_stack_bb: Starting stack in big blinds.
        """
        self.tracker = GameStateTracker(
            num_players=num_players,
            big_blind=big_blind,
            initial_stack_bb=initial_stack_bb,
        )
        self.logger = logging.getLogger(__name__)
    
    def process_event(self, event: dict[str, Any]) -> None:
        """Process a single poker event and update internal state.
        
        Supported event types:
            "deal":         Deal hole cards to hero.
            "board":        Update public cards (flop/turn/river).
            "action":       Hero or oppo action (fold/check/raise/etc).
            "pot_update":   Explicit pot size update (if scrape includes it).
            "stack_update":  Update opponent stack sizes.
        
        Args:
            event: Dictionary with at least a "type" key.
        """
        event_type = event.get("type", "unknown")
        
        if event_type == "hand_start":
            self.tracker.reset_hand()
            self.tracker.hand_started = True
            logger.debug("Hand reset and started")
        
        elif event_type == "deal":
            self.tracker.my_hand = event.get("hand", [])
            self.tracker.my_position = event.get("position", 0)
            self.tracker.cards_dealt = True
            self.tracker.hand_started = True
            logger.debug(
                "Deal: hand=%s, position=%d",
                self.tracker.my_hand, self.tracker.my_position
            )
        
        elif event_type == "board":
            # Flop, turn, or river
            public = event.get("public_cards", [])
            self.tracker.public_cards = public
            self.tracker.current_street = self._infer_street(len(public))
            logger.debug(
                "Board update: cards=%s, street=%d",
                public, self.tracker.current_street
            )
        
        elif event_type == "action":
            # Defensive check: ensure cards are dealt before processing actions
            if not self.tracker.cards_dealt:
                logger.warning(
                    "Action event received before deal. Skipping action (state error)."
                )
                return
            self._process_action(event)
        
        elif event_type == "pot_update":
            self.tracker.pot = float(event.get("pot", 0.0))
            logger.debug("Pot update: %.2f", self.tracker.pot)
        
        elif event_type == "stack_update":
            stacks = event.get("stacks", [])
            if stacks:
                self.tracker.stacks = [float(s) for s in stacks]
                logger.debug("Stack update: %s", [f"{s:.0f}" for s in self.tracker.stacks])
    
    def get_state(self) -> dict[str, Any]:
        """Export current game state as raw observation dict.
        
        This dict is directly compatible with ObservationBuilder.build(raw_state_dict).
        """
        return self.tracker.to_raw_state()
    
    def _process_action(self, event: dict[str, Any]) -> None:
        """Process an action event (fold/check/call/raise/all-in).
        
        Critical: Compute pot_before and spr_before before the action,
        then update pot and stacks after.
        """
        player_id = event.get("player", -1)
        action_str = event.get("action", "unknown")
        amount = float(event.get("amount", 0.0))
        
        # Validate player ID (bounds check)
        if not (0 <= player_id < self.tracker.num_players):
            logger.warning(
                "Invalid player ID %d (expected [0, %d)). Skipping action.",
                player_id, self.tracker.num_players
            )
            return
        
        # Compute state BEFORE this action
        pot_before = self.tracker.pot
        
        # [FIX] Compute effective stack with empty generator guard
        # In all-in situations, some stacks may be 0. Use max() as fallback.
        active_stacks = [
            self.tracker.stacks[i]
            for i in range(self.tracker.num_players)
            if self.tracker.stacks[i] > 0
        ]
        
        if active_stacks:
            effective_stack = min(active_stacks)
        else:
            # All players all-in or busted — use max remaining or fallback
            effective_stack = max(self.tracker.stacks) if self.tracker.stacks else 0.0
            if effective_stack > 0:
                logger.warning(
                    "All active stacks are 0: using max remaining stack (%.0f) for SPR calculation",
                    effective_stack
                )
        
        spr_before = effective_stack / pot_before if pot_before > 0.0 else 0.0
        
        # Map action string to action index (PokerAction enum equivalent)
        action_idx = self._map_action_name_to_index(action_str, amount)
        
        # Append to betting history with pot_before and spr_before
        self.tracker.betting_history.append({
            "action": action_idx,
            "amount": amount,
            "player": player_id,
            "street": self.tracker.current_street,
            "pot_before": pot_before,
            "spr_before": spr_before,
        })
        
        # Update state AFTER action
        if action_idx == 0:  # Fold
            pass  # Don't update pot or stacks
        else:
            self.tracker.stakes[player_id] += amount
            self.tracker.stacks[player_id] -= amount
            self.tracker.pot = sum(self.tracker.stakes)
        
        logger.debug(
            "Action: player=%d, %s (amount=%.0f), spr_before=%.2f, pot_before=%.0f",
            player_id, action_str, amount, spr_before, pot_before
        )
    
    def _infer_street(self, num_public_cards: int) -> int:
        """Map number of public cards to street index."""
        if num_public_cards == 0:
            return 0  # preflop
        elif num_public_cards == 3:
            return 1  # flop
        elif num_public_cards == 4:
            return 2  # turn
        elif num_public_cards == 5:
            return 3  # river
        else:
            return 0
    
    def _map_action_name_to_index(self, action_name: str, amount: float) -> int:
        """Convert action string to PokerAction index.
        
        Examples:
            "fold" → 0
            "check" → 1
            "call" → 1
            "raise_0.25x_pot" → 3
            "all_in" → 10
        """
        action_name_lower = action_name.lower().strip()
        
        if action_name_lower in ("fold",):
            return 0
        elif action_name_lower in ("check", "call"):
            return 1
        elif action_name_lower in ("min_raise", "min-raise"):
            return 2
        elif action_name_lower in ("raise_0.25x_pot", "raise_quarter", "0.25x"):
            return 3
        elif action_name_lower in ("raise_0.33x_pot", "raise_third", "0.33x"):
            return 4
        elif action_name_lower in ("raise_0.5x_pot", "raise_half", "0.5x"):
            return 5
        elif action_name_lower in ("raise_0.75x_pot", "raise_75", "0.75x"):
            return 6
        elif action_name_lower in ("raise_1.0x_pot", "raise_pot", "1x"):
            return 7
        elif action_name_lower in ("raise_1.5x_pot", "raise_150", "1.5x"):
            return 8
        elif action_name_lower in ("raise_2.0x_pot", "raise_2x", "2x"):
            return 9
        elif action_name_lower in ("all_in", "allin"):
            return 10
        else:
            logger.warning("Unknown action: %s (defaulting to fold)", action_name)
            return 0
    
    def _compute_legal_actions(self) -> list[int]:
        """Compute legal actions for current game state.
        
        Adapted from benchmark_runner.py logic:
        - Fold always legal
        - Check/call legal if amount_to_call >= 0
        - Raises legal based on stack size and prior raises
        """
        my_chips = self.tracker.stacks[self.tracker.my_position]
        max_stake = max(self.tracker.stakes) if self.tracker.stakes else 0.0
        my_stake = self.tracker.stakes[self.tracker.my_position]
        amount_to_call = max(0.0, max_stake - my_stake)
        
        if my_chips <= 0:
            return [0]  # Can only fold (shouldnt happen in real game)
        
        legal = [0]  # Fold always legal
        
        if amount_to_call <= my_chips:
            legal.append(1)  # Can check/call
        
        if amount_to_call < my_chips:
            # Can raise
            num_raises = len([h for h in self.tracker.betting_history if h["action"] >= 2])
            
            # Heuristic: enable raises based on number of prior raises
            if num_raises >= 1:
                legal.append(2)  # Min-raise
            if num_raises >= 2:
                legal.extend([3, 4])  # 25%, 33% pot
            if num_raises >= 3:
                legal.extend([5, 6, 7])  # 50%, 75%, 1x
            if num_raises >= 4:
                legal.extend([8, 9])  # 1.5x, 2x
            
            if my_chips > 0:
                legal.append(10)  # All-in always available if stack left
        
        return sorted(list(set(legal)))  # Remove duplicates and sort
