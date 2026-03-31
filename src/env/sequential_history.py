"""
Phase 1: Game State Representation & Sequential History for VR-DeepPDCFR+

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

The VR-DeepPDCFR+ algorithm requires precise tracking of game history to enable:
  1. Counterfactual tree traversal with reach probability weighting
  2. Fast information set lookups (hashing) during CFR updates
  3. Clear separation of public vs private information
  4. Immutable state objects to prevent side-effects during tree walks

DESIGN PRINCIPLES
=================

1. SOLID Principles Adherence:
   - Single Responsibility: Each class handles one concern
   - Open/Closed: Extensible without modification
   - Composition over Inheritance: Use dataclass/struct patterns
   
2. Immutability:
   - State objects are frozen dataclasses (immutable by default)
   - State transitions create NEW instances (functional style)
   - Eliminates subtle bugs from shared mutable state
   
3. Type Safety:
   - Strict type hints for all parameters and returns
   - Use Literal types for constrained string enums
   - Protocol-based duck typing where appropriate

4. Efficiency:
   - Cached hashing for O(1) infoset lookups
   - Batch serialization to tensors for neural networks
   - No deep copying; structural sharing via immutability

INFORMATION HIERARCHY
=====================

                     GameState
                    /          \\
             PublicState    PrivateState
            /      |     \\         |
          Pot  Cards  History   HoleCards
          |
     ActionSequence

PublicState (everything visible to all players):
  - Community cards (revealed progressively: flop, turn, river)
  - Betting history (action sequence: check, bet, call, fold)
  - Pot size, stack sizes
  - Current street (preflop, flop, turn, river)

PrivateState (unique per player):
  - Hole cards (only known to that player)
  - In CFR, we track the belief distribution over opponent cards

InformationSet (for CFR):
  - Defined by: hole cards + public history
  - Two situations are in the same infoset IFF:
    - Same player to act
    - Same hole cards
    - Same PUBLIC history (this player doesn't see opponent's hole cards)
  - Used for strategy averaging and regret accumulation

SERIALIZATION FOR NEURAL NETWORKS (Phase 2-3)
==============================================

GameState can be vectorized into tensor format:
  - Card features: one-hot encoding of board + hole cards
  - Betting history: action sequence features (action type, size, street)
  - Position/player info: current player index, stack sizes

This is handled by separate encoder classes (not core state repr).

---

References:
  - Koulis, Schvartzman et al. (2022): "VR-DeepPDCFR+"
  - Hart & Mas-Colell (2000): "A Simple Adaptive Procedure Leading to CFR"
  - Bowling et al. (2015): "Heads-up Limit Hold'em Poker is Solved"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & TYPE DEFINITIONS
# ============================================================================

class ActionType(Enum):
    """Enumeration of all possible poker actions.
    
    These are game-theoretic actions; not implementation details.
    """
    FOLD = auto()     # Pass the action to opponent (forfeits rights)
    CHECK = auto()    # Pass without betting (only when to_you=0)
    CALL = auto()     # Match opponent's bet
    BET = auto()      # Initiate aggression (arbitrary size, >= min bet)
    RAISE = auto()    # Re-aggress (size >= min raise)


class Street(Enum):
    """Enumeration of betting streets.
    
    Defines when cards are revealed and when new betting rounds occur.
    """
    PREFLOP = auto()  # Before any community cards revealed
    FLOP = auto()     # After 3 community cards
    TURN = auto()     # After 4th community card
    RIVER = auto()    # After 5th community card


# ============================================================================
# ACTION SEQUENCE: Immutable History of Actions
# ============================================================================

@dataclass(frozen=True)
class HoleCards:
    """Represents a player's private hole cards.
    
    Attributes:
        card1: First card (0-51 using Treys encoding)
        card2: Second card (0-51 using Treys encoding)
    
    Example:
        A♠ K♦ would be represented as HoleCards(51, 36)
        [Treys: A♠=51, K♦=36 under standard indexing]
    """
    card1: int  # 0-51 (Treys encoding)
    card2: int  # 0-51
    
    def __post_init__(self):
        """Validate card indices."""
        for card in (self.card1, self.card2):
            if not 0 <= card <= 51:
                raise ValueError(f"Invalid card index: {card}, must be 0-51")
            if self.card1 == self.card2:
                raise ValueError("Cannot have duplicate cards")
    
    @property
    def cards(self) -> Tuple[int, int]:
        """Return cards as tuple for easy unpacking."""
        return (self.card1, self.card2)


@dataclass(frozen=True)
class Action:
    """Represents a single action taken by a player during the game.
    
    Attributes:
        action_type: The type of action (fold, check, call, bet, raise)
        player_idx: Which player took the action (0 or 1)
        street: Which betting street this occurred on
        bet_size: How much was wagered (0 for check/fold/call with no bet)
        
    Note:
        All attributes are immutable. Actions form an immutable sequence.
        
    Example:
        Player 0 bets 2 units on preflop:
        >>> a = Action(ActionType.BET, player_idx=0, street=Street.PREFLOP, bet_size=2.0)
    """
    action_type: ActionType
    player_idx: int
    street: Street
    bet_size: float = 0.0  # In big blinds or chips
    
    def __post_init__(self):
        """Validate action constraints."""
        if not 0 <= self.player_idx <= 1:
            raise ValueError(f"player_idx must be 0 or 1, got {self.player_idx}")
        if self.action_type == ActionType.BET and self.bet_size <= 0:
            raise ValueError(f"BET action must have positive bet_size, got {self.bet_size}")
        if self.bet_size < 0:
            raise ValueError(f"bet_size cannot be negative, got {self.bet_size}")
    
    def __repr__(self) -> str:
        """Human-readable action description."""
        if self.action_type in (ActionType.FOLD, ActionType.CHECK):
            return f"{self.action_type.name}"
        else:
            return f"{self.action_type.name}({self.bet_size:.2f})"


@dataclass(frozen=True)
class ActionSequence:
    """Immutable sequence of actions taken in the game.
    
    Represents the public betting history. Two information sets with the same
    ActionSequence are indistinguishable from the perspective of public
    information.
    
    Attributes:
        actions: Tuple of Action objects (immutable)
        
    Note:
        Use the builder methods to construct sequences (never mutate directly).
        
    Example:
        >>> seq = ActionSequence.empty()
        >>> a1 = Action(ActionType.BET, player_idx=0, street=Street.PREFLOP, bet_size=1.0)
        >>> seq = seq.append(a1)
        >>> a2 = Action(ActionType.CALL, player_idx=1, street=Street.PREFLOP, bet_size=1.0)
        >>> seq = seq.append(a2)
    """
    actions: Tuple[Action, ...] = field(default_factory=tuple)
    
    @staticmethod
    def empty() -> ActionSequence:
        """Create an empty (initial) action sequence."""
        return ActionSequence(actions=())
    
    def append(self, action: Action) -> ActionSequence:
        """Return a new sequence with the action appended.
        
        Args:
            action: The action to append
            
        Returns:
            New ActionSequence with action added
            
        Note:
            Original sequence is unchanged (immutable).
        """
        return ActionSequence(actions=self.actions + (action,))
    
    def __len__(self) -> int:
        """Number of actions in the sequence."""
        return len(self.actions)
    
    def __getitem__(self, index: int) -> Action:
        """Access action by index."""
        return self.actions[index]
    
    def __iter__(self):
        """Iterate over actions in order."""
        return iter(self.actions)
    
    def to_string(self) -> str:
        """Compact string representation of action sequence.
        
        Returns:
            String like "C-B-C" where C=check, B=bet, etc.
        """
        action_chars = {
            ActionType.FOLD: "F",
            ActionType.CHECK: "C",
            ActionType.CALL: "L",
            ActionType.BET: "B",
            ActionType.RAISE: "R",
        }
        return "-".join(action_chars[a.action_type] for a in self.actions)


# ============================================================================
# PUBLIC STATE: Information Visible to All Players
# ============================================================================

@dataclass(frozen=True)
class PublicState:
    """Represents the public game state (visible to all players).
    
    Contains all information that is not opponent hole cards.
    
    Attributes:
        community_cards: Tuple of revealed community cards (0-51, empty if preflop)
        action_history: Sequence of all actions taken so far on this street
        current_street: Which street the game is on (preflop, flop, turn, river)
        pot_size: Total chips in the pot (starting from antes/blinds)
        player_stacks: Tuple of (p0_remaining_chips, p1_remaining_chips)
        button_idx: Which player is on the dealer button (0 or 1)
        small_blind: Blind posting amount
        big_blind: Big blind amount
        
    Note:
        All attributes are immutable. State transitions produce new instances.
        
    Invariants:
        - community_cards: () on preflop, 3 cards on flop, 4 on turn, 5 on river
        - All card values must be 0-51
        - Stacks cannot be negative
        - Pot must be non-negative
    """
    community_cards: Tuple[int, ...] = field(default_factory=tuple)
    action_history: ActionSequence = field(default_factory=ActionSequence.empty)
    current_street: Street = Street.PREFLOP
    pot_size: float = 0.0
    player_stacks: Tuple[float, float] = (0.0, 0.0)  # Remaining chips
    button_idx: int = 0
    small_blind: float = 0.01
    big_blind: float = 0.02
    
    def __post_init__(self):
        """Validate state invariants."""
        if not 0 <= self.button_idx <= 1:
            raise ValueError(f"button_idx must be 0 or 1, got {self.button_idx}")
        
        expected_cards = {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
        }
        if len(self.community_cards) != expected_cards[self.current_street]:
            raise ValueError(
                f"street={self.current_street.name} expects "
                f"{expected_cards[self.current_street]} community cards, "
                f"got {len(self.community_cards)}"
            )
        
        # Validate card indices
        for card in self.community_cards:
            if not 0 <= card <= 51:
                raise ValueError(f"Invalid community card: {card}")
        
        # Validate stacks
        for i, stack in enumerate(self.player_stacks):
            if stack < 0:
                raise ValueError(f"player_stacks[{i}] cannot be negative: {stack}")
        
        if self.pot_size < 0:
            raise ValueError(f"pot_size cannot be negative: {self.pot_size}")
    
    def append_action(self, action: Action) -> PublicState:
        """Create a new PublicState with an appended action.
        
        Args:
            action: The action to append
            
        Returns:
            New PublicState with updated history
            
        Note:
            Does NOT transition streets or update pot. Use advance_to_*.
            Action history is cumulative throughout all streets (preflop->flop->turn->river).
        """
        new_history = self.action_history.append(action)
        return PublicState(
            community_cards=self.community_cards,
            action_history=new_history,
            current_street=self.current_street,
            pot_size=self.pot_size,
            player_stacks=self.player_stacks,
            button_idx=self.button_idx,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
    
    def advance_to_flop(self, flop_cards: Tuple[int, int, int]) -> PublicState:
        """Transition to flop with the revealed cards.
        
        Args:
            flop_cards: 3-tuple of community cards (0-51 each)
            
        Returns:
            New PublicState on flop street
            
        Note:
            action_history is NOT reset; it accumulates throughout the entire hand.
            This is critical for CFR: information sets are keyed by complete action sequence.
        """
        if len(flop_cards) != 3:
            raise ValueError(f"Flop requires exactly 3 cards, got {len(flop_cards)}")
        return PublicState(
            community_cards=flop_cards,
            action_history=self.action_history,  # PRESERVE: cumulative history
            current_street=Street.FLOP,
            pot_size=self.pot_size,
            player_stacks=self.player_stacks,
            button_idx=self.button_idx,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
    
    def advance_to_turn(self, turn_card: int) -> PublicState:
        """Transition to turn with the 4th community card.
        
        Args:
            turn_card: The 4th community card (0-51)
            
        Returns:
            New PublicState on turn street
            
        Note:
            action_history is NOT reset; it accumulates throughout the entire hand.
            This is critical for CFR: information sets are keyed by complete action sequence.
        """
        if self.current_street != Street.FLOP:
            raise ValueError(f"Can only advance to turn from flop, current={self.current_street}")
        new_cards = self.community_cards + (turn_card,)
        return PublicState(
            community_cards=new_cards,
            action_history=self.action_history,  # PRESERVE: cumulative history
            current_street=Street.TURN,
            pot_size=self.pot_size,
            player_stacks=self.player_stacks,
            button_idx=self.button_idx,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
    
    def advance_to_river(self, river_card: int) -> PublicState:
        """Transition to river with the 5th community card.
        
        Args:
            river_card: The 5th community card (0-51)
            
        Returns:
            New PublicState on river street
            
        Note:
            action_history is NOT reset; it accumulates throughout the entire hand.
            This is critical for CFR: information sets are keyed by complete action sequence.
        """
        if self.current_street != Street.TURN:
            raise ValueError(f"Can only advance to river from turn, current={self.current_street}")
        new_cards = self.community_cards + (river_card,)
        return PublicState(
            community_cards=new_cards,
            action_history=self.action_history,  # PRESERVE: cumulative history
            current_street=Street.RIVER,
            pot_size=self.pot_size,
            player_stacks=self.player_stacks,
            button_idx=self.button_idx,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
    

    
    def __repr__(self) -> str:
        """Human-readable representation."""
        street = self.current_street.name
        action_str = self.action_history.to_string() if len(self.action_history) > 0 else "(empty)"
        return f"PublicState(street={street}, actions={action_str}, pot={self.pot_size:.2f})"


# ============================================================================
# PRIVATE STATE: Information Unique to a Player
# ============================================================================

@dataclass(frozen=True)
class PrivateState:
    """Represents the private game state for a specific player.
    
    This is the information known only to that player (their hole cards).
    
    Attributes:
        player_idx: Which player this private state belongs to (0 or 1)
        hole_cards: The player's private hole cards
        
    Note:
        In CFR, we don't directly track opponent hole cards. Instead,
        we traverse the game tree over all possible opponent cards weighted
        by reach probabilities.
    """
    player_idx: int
    hole_cards: HoleCards
    
    def __post_init__(self):
        """Validate player index."""
        if not 0 <= self.player_idx <= 1:
            raise ValueError(f"player_idx must be 0 or 1, got {self.player_idx}")


# ============================================================================
# INFORMATION SET: Grouping for CFR Strategy Management
# ============================================================================

@dataclass(frozen=True)
class InformationSet:
    """Represents an information set in the game tree (CFR terminology).
    
    An information set groups all game states that are indistinguishable
    to the player making a decision. In poker:
    
      InfoSet(hole_cards, public_history, current_street, to_act_player)
    
    Key property: If two states have the same infoset, they must have:
      - Same player to act
      - Same hole cards for that player
      - Same PUBLIC history (opponent's cards are unknown to this player)
    
    Used in CFR for:
      1. Strategy averaging: accumulate strategy over all states in same infoset
      2. Regret accumulation: average regrets across infoset
      3. Lookups: fast table of {InfoSet -> Strategy}
    
    Attributes:
        player_idx: Which player is deciding (0 or 1)
        hole_cards: That player's private hole cards
        public_hash: Hash of (community_cards, action_history, street)
            - This is the public information (visible to deciding player)
        public_str: String representation of public state (for debugging)
        
    Note:
        We use hashing rather than storing full PublicState for efficiency.
        The public_str is optional (for logging only).
    """
    player_idx: int
    hole_cards: HoleCards
    public_hash: int
    public_str: str = ""  # For human debugging only
    
    @staticmethod
    def from_states(
        public_state: PublicState,
        private_state: PrivateState,
    ) -> InformationSet:
        """Construct an InformationSet from public and private states.
        
        Args:
            public_state: The public game state
            private_state: The player's private state
            
        Returns:
            InformationSet representing this decision point
        """
        return InformationSet(
            player_idx=private_state.player_idx,
            hole_cards=private_state.hole_cards,
            public_hash=hash(public_state),
            public_str=repr(public_state),
        )
    
    def __repr__(self) -> str:
        """Human-readable information set."""
        return (f"InfoSet(P{self.player_idx}, "
                f"cards=({self.hole_cards.card1},{self.hole_cards.card2}), "
                f"hash={self.public_hash})")


# ============================================================================
# GAME STATE: Complete State at a Decision Point
# ============================================================================

@dataclass(frozen=True)
class GameState:
    """Complete immutable game state at a decision point or terminal node.
    
    Combines public and private information. Used in game tree traversal for:
      1. Simulating actions and transitioning to next state
      2. Evaluating terminal payoffs
      3. Tracking reach probabilities in counterfactual traversal
    
    Attributes:
        public_state: All information visible to all players
        private_states: Tuple of (PrivateState_p0, PrivateState_p1) - index by player_idx
        terminal: Whether this is a terminal state (game over)
        payoffs: If terminal, payoffs for each player (0-indexed dict)
        
    Invariants:
        - If terminal: payoffs dict must exist and have entries
        - If not terminal: payoffs should be empty
        - private_states must have exactly 2 elements (p0, p1)
    
    Example:
        >>> # Initial state
        >>> pub = PublicState(pot_size=0.03, player_stacks=(100.0, 100.0))
        >>> p0_cards = HoleCards(card1=51, card2=36)  # A♠ K♦
        >>> p1_cards = HoleCards(card1=50, card2=35)  # K♠ Q♦
        >>> 
        >>> priv = (
        ...     PrivateState(player_idx=0, hole_cards=p0_cards),
        ...     PrivateState(player_idx=1, hole_cards=p1_cards),
        ... )
        >>> 
        >>> state = GameState(
        ...     public_state=pub,
        ...     private_states=priv,
        ...     terminal=False,
        ...     payoffs={}
        ... )
    """
    public_state: PublicState
    private_states: Tuple[PrivateState, PrivateState]
    terminal: bool = False
    payoffs: Dict[int, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate state consistency."""
        if self.terminal and len(self.payoffs) == 0:
            raise ValueError("Terminal state must have payoffs")
        if not self.terminal and len(self.payoffs) > 0:
            logger.warning("Non-terminal state has payoffs; they will be ignored")
        if len(self.private_states) != 2:
            raise ValueError(f"private_states must have exactly 2 elements, got {len(self.private_states)}")
    
    def get_infoset(self, player_idx: int) -> InformationSet:
        """Get the information set for a given player at this state.
        
        Args:
            player_idx: Which player (0 or 1)
            
        Returns:
            InformationSet representing player's view of the game
        """
        if not 0 <= player_idx <= 1:
            raise ValueError(f"player_idx must be 0 or 1, got {player_idx}")
        return InformationSet.from_states(
            public_state=self.public_state,
            private_state=self.private_states[player_idx],
        )
    
    def append_action(self, action: Action) -> GameState:
        """Create a new GameState with the action applied.
        
        Args:
            action: The action to apply
            
        Returns:
            New GameState with updated public history
            
        Note:
            Does NOT update pot or stacks. Game logic handles that separately.
        """
        new_public = self.public_state.append_action(action)
        return GameState(
            public_state=new_public,
            private_states=self.private_states,
            terminal=False,
            payoffs={},
        )
    
    def to_terminal(self, payoffs: Dict[int, float]) -> GameState:
        """Create a terminal state with the given payoffs.
        
        Args:
            payoffs: Dict mapping player_idx -> chip payoff
            
        Returns:
            New GameState marked as terminal with payoffs
        """
        return GameState(
            public_state=self.public_state,
            private_states=self.private_states,
            terminal=True,
            payoffs=payoffs,
        )
    
    def __repr__(self) -> str:
        """Human-readable state."""
        status = "TERMINAL" if self.terminal else "ACTIVE"
        p0_cards = self.private_states[0].hole_cards
        p1_cards = self.private_states[1].hole_cards
        player0_cards = f"({p0_cards.card1},{p0_cards.card2})"
        player1_cards = f"({p1_cards.card1},{p1_cards.card2})"
        return (f"GameState({status}, P0{player0_cards}, P1{player1_cards}, "
                f"{self.public_state})")


# ============================================================================
# GAME HISTORY: Full Trajectory for Counterfactual Traversal
# ============================================================================

@dataclass(frozen=True)
class GameHistory:
    """Immutable record of a complete game trajectory.
    
    Used in counterfactual tree traversal to:
      1. Track reach probabilities (product of all actions along path)
      2. Compute regrets and strategy updates
      3. Implement CFR updates efficiently
    
    Attributes:
        states: Tuple of GameState objects in chronological order
            - states[0] is the initial state (preflop, cards dealt)
            - states[-1] is the terminal state
        reach_probs: Reach probability for each state
            - reach_probs[i] = prob of reaching states[i]
            - Used to weight counterfactual contributions in CFR
    """
    states: Tuple[GameState, ...]
    reach_probs: Tuple[float, ...] = field(default_factory=tuple)
    
    @staticmethod
    def empty() -> GameHistory:
        """Create an empty game history."""
        return GameHistory(states=(), reach_probs=())
    
    def append(
        self,
        state: GameState,
        reach_prob: float = 1.0,
    ) -> GameHistory:
        """Append a state to the history.
        
        Args:
            state: The state to append
            reach_prob: Reach probability to this state
            
        Returns:
            New GameHistory with state appended
        """
        new_reach_probs = self.reach_probs + (reach_prob,)
        return GameHistory(
            states=self.states + (state,),
            reach_probs=new_reach_probs,
        )
    
    def final_state(self) -> Optional[GameState]:
        """Get the terminal state (end of game)."""
        return self.states[-1] if len(self.states) > 0 else None
    
    def __len__(self) -> int:
        """Number of states in the trajectory."""
        return len(self.states)
    
    def __getitem__(self, index: int) -> GameState:
        """Access state by index."""
        return self.states[index]
    
    def __iter__(self):
        """Iterate over states in chronological order."""
        return iter(self.states)


# ============================================================================
# SERIALIZATION HELPERS: Converting to Tensor Format (Phase 2-3)
# ============================================================================

class GameStateEncoder:
    """Encode GameState objects into tensor format for neural networks.
    
    This is a skeletal interface; full implementation comes in Phase 2-3
    when integrating with neural network architectures.
    
    Responsibility: Convert game state to fixed-size vector.
    Non-responsibility: Architecture/learning (separate module).
    """
    
    @staticmethod
    def encode_hole_cards(hole_cards: HoleCards) -> Tuple[int, int]:
        """Encode hole cards as card indices.
        
        Args:
            hole_cards: The HoleCards object
            
        Returns:
            Tuple of (card1_idx, card2_idx)
        """
        return hole_cards.card1, hole_cards.card2
    
    @staticmethod
    def encode_community_cards(community_cards: Tuple[int, ...]) -> Tuple[int, ...]:
        """Encode community cards.
        
        Args:
            community_cards: Tuple of card indices
            
        Returns:
            Same tuple (minimal processing for Phase 1)
        """
        return community_cards
    
    @staticmethod
    def encode_action_sequence(action_seq: ActionSequence) -> Tuple[str, ...]:
        """Encode action sequence as strings.
        
        Args:
            action_seq: The ActionSequence
            
        Returns:
            Tuple of action string representations
        """
        return tuple(repr(a) for a in action_seq.actions)


# ============================================================================
# FACTORY & UTILITY FUNCTIONS
# ============================================================================

def create_initial_state(
    p0_hole_cards: HoleCards,
    p1_hole_cards: HoleCards,
    initial_stacks: Tuple[float, float] = (100.0, 100.0),
    button_idx: int = 0,
    small_blind: float = 0.01,
    big_blind: float = 0.02,
) -> GameState:
    """Create an initial game state (cards dealt, preflop).
    
    Args:
        p0_hole_cards: Player 0's hole cards
        p1_hole_cards: Player 1's hole cards
        initial_stacks: Starting stack sizes for (p0, p1)
        button_idx: Which player is on the button (0 or 1)
        small_blind: Small blind amount
        big_blind: Big blind amount
        
    Returns:
        GameState ready for first action
        
    Example:
        >>> cards_p0 = HoleCards(51, 36)  # A♠ K♦
        >>> cards_p1 = HoleCards(50, 49)  # K♠ Q♠
        >>> state = create_initial_state(cards_p0, cards_p1)
    """
    initial_pot = small_blind + big_blind  # From blinds
    public_state = PublicState(
        community_cards=(),  # No cards yet
        action_history=ActionSequence.empty(),
        current_street=Street.PREFLOP,
        pot_size=initial_pot,
        player_stacks=(initial_stacks[0] - small_blind, 
                       initial_stacks[1] - big_blind),
        button_idx=button_idx,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    
    private_states = (
        PrivateState(player_idx=0, hole_cards=p0_hole_cards),
        PrivateState(player_idx=1, hole_cards=p1_hole_cards),
    )
    
    return GameState(
        public_state=public_state,
        private_states=private_states,
        terminal=False,
        payoffs={},
    )
