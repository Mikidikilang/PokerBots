# Phase 1: Game State Representation Architecture for VR-DeepPDCFR+

## Overview

This document describes the foundational data structures in `sequential_history.py` that enable efficient counterfactual tree traversal for the VR-DeepPDCFR+ algorithm.

**Key Design Goals:**
- ✅ Strict separation of public vs private information
- ✅ Immutable state objects (prevent bugs during tree walks)
- ✅ Fast information set hashing for O(1) strategy lookups
- ✅ Clean SOLID architecture (single responsibility)
- ✅ Comprehensive type safety with Python hints

---

## Class Hierarchy

### 1. **ActionType** (Enum)

Defines the game-theoretic actions a player can take.

```python
class ActionType(Enum):
    FOLD       # Forfeit hand (only in response to aggression)
    CHECK      # Pass without betting (only when no bet to face)
    CALL       # Match current bet
    BET        # Initiate aggression (blind min bet / arbitrary size)
    RAISE      # Re-aggress (increase amount from opponent's bet)
```

**Note:** These are game-level actions, not GUI buttons. `CALL` and `CHECK` are separate because they represent different decision types:
- `CHECK`: "I'm passing my option"
- `CALL`: "I'm matching a bet"

### 2. **Street** (Enum)

Betting street phases.

```python
class Street(Enum):
    PREFLOP    # Before board is revealed
    FLOP       # 3-card community cards
    TURN       # 4th community card
    RIVER      # 5th community card (final)
```

Streets determine when new cards are revealed and when action resets.

### 3. **HoleCards** (Frozen Dataclass)

A player's private hole cards.

```python
@dataclass(frozen=True)
class HoleCards:
    card1: int  # 0-51 (Treys encoding)
    card2: int  # 0-51
```

**Key Properties:**
- **Immutable**: frozen=True prevents accidental mutations
- **Card Encoding**: Uses Treys library (0-51 indexing)
  - `0-12`: Deuce-Ace of spades
  - `13-25`: Deuce-Ace of hearts
  - `26-38`: Deuce-Ace of diamonds
  - `39-51`: Deuce-Ace of clubs
  - Highest 2 bits identify suit, lower bits identify rank
- **Fast Hashing**: `__hash__` uses min/max normalization (order-independent)

**Example:**
```python
# A♠ K♦
cards = HoleCards(51, 36)

# K♦ A♠ (same hand, same hash)
cards2 = HoleCards(36, 51)
assert hash(cards) == hash(cards2)
```

### 4. **Action** (Frozen Dataclass)

A single action taken in the game.

```python
@dataclass(frozen=True)
class Action:
    action_type: ActionType
    player_idx: int           # 0 or 1
    street: Street
    bet_size: float = 0.0     # In big blinds
```

**Validation:**
- `action_type=BET` requires `bet_size > 0`
- `player_idx` must be 0 or 1
- All attributes immutable (frozen)

**Example:**
```python
# Player 0 bets 2 BB preflop
bet_action = Action(
    action_type=ActionType.BET,
    player_idx=0,
    street=Street.PREFLOP,
    bet_size=2.0
)

# Player 1 checks flop
check_action = Action(
    action_type=ActionType.CHECK,
    player_idx=1,
    street=Street.FLOP,
    bet_size=0.0
)
```

### 5. **ActionSequence** (Frozen Dataclass)

Immutable sequence of actions (public betting history).

```python
@dataclass(frozen=True)
class ActionSequence:
    actions: Tuple[Action, ...] = ()
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `empty()` | Create initial empty sequence |
| `append(action)` | Return NEW sequence with action added |
| `to_string()` | Compact repr: `"C-B-C-L"` (check-bet-check-call) |
| `__hash__()` | Cached hash for fast lookups |

**Example - Building a Sequence:**
```python
# Start empty
seq = ActionSequence.empty()

# Append actions (functional style - returns new objects)
action1 = Action(ActionType.BET, player_idx=0, street=Street.PREFLOP, bet_size=1.0)
seq = seq.append(action1)

action2 = Action(ActionType.CALL, player_idx=1, street=Street.PREFLOP, bet_size=1.0)
seq = seq.append(action2)

print(seq.to_string())  # "B-L"
print(len(seq))         # 2
```

### 6. **PublicState** (Frozen Dataclass)

All information visible to all players (no hole cards).

```python
@dataclass(frozen=True)
class PublicState:
    community_cards: Tuple[int, ...] = ()
    action_history: ActionSequence
    current_street: Street
    pot_size: float
    player_stacks: Tuple[float, float]
    button_idx: int  # 0 or 1
    small_blind: float
    big_blind: float
```

**Invariants (Validated in `__post_init__`):**
- PREFLOP: 0 community cards
- FLOP: 3 community cards
- TURN: 4 community cards
- RIVER: 5 community cards
- All card indices: 0-51
- Stack sizes ≥ 0
- Pot size ≥ 0

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `append_action(action)` | Add action, return new PublicState |
| `advance_to_flop(cards)` | Move to flop, reset action history |
| `advance_to_turn(card)` | Move to turn |
| `advance_to_river(card)` | Move to river |
| `__hash__()` | Cached hash (cards + history + street only) |

**Example:**
```python
# Initial state
pub = PublicState(
    community_cards=(),
    action_history=ActionSequence.empty(),
    current_street=Street.PREFLOP,
    pot_size=0.03,  # SB + BB
    player_stacks=(100.0, 100.0),
    button_idx=0,
    small_blind=0.01,
    big_blind=0.02
)

# Player 0 bets
action = Action(ActionType.BET, player_idx=0, street=Street.PREFLOP, bet_size=1.0)
pub = pub.append_action(action)

# Flop is dealt: 2♥ 7♦ K♠
flop_cards = (12, 25, 39)  # Treys indices
pub = pub.advance_to_flop(flop_cards)

# NOW action_history is reset (new street)
assert len(pub.action_history) == 0
assert pub.current_street == Street.FLOP
```

### 7. **PrivateState** (Frozen Dataclass)

Player-specific private information (their hole cards).

```python
@dataclass(frozen=True)
class PrivateState:
    player_idx: int
    hole_cards: HoleCards
```

**Note:** In CFR, we don't directly track opponent's hole cards. Instead, the algorithm traverses the game tree over all possible opponent hole cards, weighted by reach probabilities.

### 8. **InformationSet** (Frozen Dataclass)

Core CFR abstraction: groups decision points that are indistinguishable to a player.

```python
@dataclass(frozen=True)
class InformationSet:
    player_idx: int
    hole_cards: HoleCards
    public_hash: int      # Hash of PublicState
    public_str: str = ""  # For debugging
```

**What is an Information Set?**

Two game states are in the **same information set** if:
1. Same player to move (player_idx)
2. Same hole cards for that player
3. Identical PUBLIC history (cards + actions seen by all)

All states in the same infoset must be treated identically from the player's perspective.

**Example:**
```
Infoset 1: Player 1 with K♥ 5♦, Public=["P0 bet 1BB preflop"]
Infoset 2: Player 1 with K♥ 5♦, Public=["P0 checked preflop"]

→ Different public histories → DIFFERENT infosets

Infoset 1: Player 0 with A♠ A♦, Public=["flop: 2 4 6", "P0 checked, P1 bet"]
Infoset 2: Player 0 with A♠ A♦, Public=["flop: 2 4 6", "P0 checked, P1 bet"]

→ Identical → SAME infoset (strategy averaged across these)
```

**Key Methods:**

```python
# Construct from PublicState + PrivateState
infoset = InformationSet.from_states(public_state, private_state)

# Hash automatically combines private + public info
# Used as dictionary key for strategy lookups
strategy_table[infoset] = {'BET': 0.4, 'CHECK': 0.6}
```

### 9. **GameState** (Frozen Dataclass)

Complete game state: combines public + private information.

```python
@dataclass(frozen=True)
class GameState:
    public_state: PublicState
    private_states: Dict[int, PrivateState]  # {0: PrivateState, 1: PrivateState}
    terminal: bool = False
    payoffs: Dict[int, float] = {}  # {0: payoff0, 1: payoff1} if terminal
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `get_infoset(player_idx)` | Get InformationSet for specified player |
| `append_action(action)` | Apply action, return new GameState |
| `to_terminal(payoffs)` | Mark as terminal with payoff dict |

**Example:**
```python
# Create initial state
state = GameState(
    public_state=pub,
    private_states={
        0: PrivateState(0, HoleCards(51, 36)),  # A♠ K♦
        1: PrivateState(1, HoleCards(50, 49)),  # K♠ Q♠
    },
    terminal=False,
    payoffs={}
)

# Get Player 0's view
infoset_p0 = state.get_infoset(0)

# Apply action
action = Action(ActionType.BET, 0, Street.PREFLOP, 1.0)
next_state = state.append_action(action)

# Terminal state
payoffs = {0: 0.5, 1: -0.5}  # P0 wins 0.5, P1 loses 0.5
terminal_state = state.to_terminal(payoffs)
```

### 10. **GameHistory** (Frozen Dataclass)

Complete trajectory from initial state to showdown.

```python
@dataclass(frozen=True)
class GameHistory:
    states: Tuple[GameState, ...]
    reach_probs: Tuple[float, ...]  # Probability of reaching each state
```

**Key Methods:**

```python
# Build incrementally
history = GameHistory.empty()
history = history.append(state1, reach_prob=1.0)
history = history.append(state2, reach_prob=0.5)

# Access
final = history.final_state()
prob_to_state_5 = history.reach_probs[5]
```

**Why Reach Probabilities?**

In counterfactual tree traversal (CFR), we weight regret updates by reach probability:

$$\text{regret}(a) = \text{reach\_prob} \times \text{counterfactual\_value}(a)$$

This allows efficient importance sampling without replaying entire histories.

### 11. **GameStateEncoder** (Helper Class)

Skeleton for encoding GameState to tensor format (Phase 2-3).

```python
class GameStateEncoder:
    @staticmethod
    def encode_hole_cards(hole_cards: HoleCards) -> Tuple[int, int]:
        return hole_cards.card1, hole_cards.card2
    
    @staticmethod
    def encode_community_cards(cards: Tuple[int, ...]) -> Tuple[int, ...]:
        return cards
    
    @staticmethod
    def encode_action_sequence(seq: ActionSequence) -> Tuple[str, ...]:
        return tuple(repr(a) for a in seq.actions)
```

This is a placeholder; full neural network integration comes in Phase 2-3.

---

## Design Patterns & Best Practices

### 1. Immutability via Frozen Dataclasses

All state objects use `@dataclass(frozen=True)` to prevent mutations:

```python
# ❌ WRONG: Trying to mutate
state.pot_size = 100.0  # FrozenInstanceError!

# ✅ RIGHT: Create new instance
new_state = GameState(
    public_state=PublicState(..., pot_size=100.0),
    ...
)
```

**Benefits:**
- **No side effects**: Concurrent CFR traversals don't interfere
- **Structural sharing**: Old objects reused; only new parts copy
- **Easier debugging**: Can always inspect historical states

### 2. Functional State Transitions

Always return NEW objects:

```python
# ❌ WRONG (mutating style)
history = []
history.append(state)  # Mutable list

# ✅ RIGHT (functional style)
history = GameHistory.empty()
history = history.append(state)  # Returns new tuple
```

### 3. Fast Hashing with `@cached_property`

Information sets are frequently hashed for dictionary lookups:

```python
@cached_property
def __hash__(self) -> int:
    return hash((self.community_cards, self.action_history, self.current_street))
```

The `@cached_property` decorator ensures hash is computed once and reused, enabling O(1) lookups in `{InfoSet -> Strategy}` tables.

### 4. Separation of Concerns

- `PublicState`: What all players see
- `PrivateState`: Player-specific information
- `InformationSet`: CFR abstraction (for strategy storage)
- `GameState`: Full state at a decision point
- `GameHistory`: Complete trajectory (for counterfactual weight tracking)

### 5. Validation in `__post_init__`

All invariants are checked at construction time:

```python
def __post_init__(self):
    if len(self.community_cards) != expected_cards[self.current_street]:
        raise ValueError("Card count mismatch for street")
```

This prevents invalid states from ever being created.

---

## Usage Patterns for CFR

### Building a Game Tree

```python
from src.env.sequential_history import (
    ActionType, Street, Action, HoleCards, ActionSequence,
    PublicState, PrivateState, GameState, GameHistory,
    InformationSet, create_initial_state
)

# Step 1: Create initial state
p0_cards = HoleCards(51, 36)  # A♠ K♦
p1_cards = HoleCards(50, 49)  # K♠ Q♠
state = create_initial_state(p0_cards, p1_cards)

# Step 2: Apply actions in game tree
action = Action(ActionType.BET, player_idx=0, street=Street.PREFLOP, bet_size=1.0)
state = state.append_action(action)

# Step 3: Get information set for strategy lookup
infoset = state.get_infoset(1)
strategy = cfr_strategy_table[infoset]  # Dict of action probs

# Step 4: Sample action from strategy
action_prob = strategy.get(ActionType.CALL, 0.0)
action = Action(ActionType.CALL, player_idx=1, street=Street.PREFLOP, bet_size=1.0)

# Step 5: Recursive traversal (in counterfactual traversal loop)
next_reach_prob = reach_prob * action_prob
# ... continue tree traversal
```

### Storing Strategy During CFR

```python
# Information set serves as unique key
infoset = InformationSet.from_states(public_state, private_state)

# Store cumulative strategy (averaged over all iterations)
cumulative_strategy[infoset] = {
    ActionType.BET: 100.0,   # Sum of all π(a) in iterations
    ActionType.CHECK: 150.0,
}

# Normalize to probability
num_iterations = 1000
final_strategy = {
    ActionType.BET: 100.0 / 250.0,    # ~0.4
    ActionType.CHECK: 150.0 / 250.0,  # ~0.6
}
```

---

## Performance Characteristics

| Operation | Time | Space |
|-----------|------|-------|
| Hash InformationSet | O(1)* | O(1) |
| Create GameState | O(1) | O(cards + history) |
| Append action | O(1) | O(1) new tuple element |
| Check state invariants | O(cards + history) | O(1) |
| Lookups in strategy table | O(1) | O(num_infosets) |

*O(1) due to `@cached_property` memoization

---

## Integration with CFR Algorithm

This state representation **directly enables:**

1. **Counterfactual Traversal**
   - GameState combines all info needed for decisions
   - InformationSet enables O(1) strategy lookups
   - GameHistory tracks reach probabilities

2. **Value Computation**
   - Public history + reach probability → immediate counterfactual value calculation
   - No need to replay entire history

3. **Regret Accumulation**
   - InformationSet groups by decision point
   - Regrets averaged across infoset

4. **Strategy Averaging**
   - One entry per InformationSet
   - Unified view of all equivalent game states

---

## Next Steps (Phase 2-3)

Once this state representation is stable:

1. **Game Environment** (`env.py`)
   - Integrate with poker game simulator (RLCard, custom)
   - Implement action validation (legal moves)
   - Compute payoffs at terminal nodes

2. **Feature Engineering** (`features.py`)
   - Convert GameState → fixed-size tensor
   - Combine card features + betting history encoding
   - Normalization for neural network input

3. **Neural Network Architecture** (`model/`)
   - Policy head: predict π(a | infoset)
   - Value head: predict V(state)
   - Training curriculum

4. **CFR Engine** (`training/cfr_engine.py`)
   - Counterfactual tree traversal
   - Regret matching + strategy averaging
   - Integration with value network

---

## References

- Hart & Mas-Colell (2000): *A Simple Adaptive Procedure Leading to CFR*
- Bowling et al. (2015): *Heads-up Limit Hold'em Poker is Solved*
- Koulis, Schvartzman et al. (2022): *VR-DeepPDCFR+*
