# Phase 1 Quick Start Guide

## 5-Minute Setup

### Import the Core Classes

```python
from src.env.sequential_history import (
    ActionType,      # FOLD, CHECK, CALL, BET, RAISE
    Street,          # PREFLOP, FLOP, TURN, RIVER
    HoleCards,       # Player's private cards
    Action,          # Single game action
    ActionSequence,  # Sequence of actions (public history)
    PublicState,     # What all players see
    PrivateState,    # Player-specific data (hole cards)
    GameState,       # Complete state (pub + priv)
    InformationSet,  # CFR abstraction for strategies
    GameHistory,     # Full game trajectory
    create_initial_state,  # Convenience factory
)
```

### Create an Initial Game State

```python
# Deal hole cards
p0_cards = HoleCards(51, 36)  # A♠ K♦
p1_cards = HoleCards(50, 49)  # K♠ Q♠

# Create initial state (preflop, blinds posted)
state = create_initial_state(
    p0_hole_cards=p0_cards,
    p1_hole_cards=p1_cards,
    initial_stacks=(100.0, 100.0),
)

print(state)
# Output: GameState(ACTIVE, P0(51,36), P1(50,49), ...)
```

### Apply Actions

```python
# Player 0 bets 1 BB
action = Action(
    action_type=ActionType.BET,
    player_idx=0,
    street=Street.PREFLOP,
    bet_size=1.0,
)

# State transitions are immutable (returns new GameState)
state = state.append_action(action)
```

### Get Information Set (for CFR Strategy)

```python
# Get Player 1's view of the game
infoset = state.get_infoset(player_idx=1)

# Use as dictionary key for strategy storage
strategy_table = {}
strategy_table[infoset] = {
    ActionType.FOLD: 0.2,
    ActionType.CALL: 0.8,
}

# Later: fast O(1) lookup
action_probs = strategy_table[infoset]
```

---

## Common Patterns

### Build Action Sequence

```python
seq = ActionSequence.empty()

# Functional style (returns new sequence)
seq = seq.append(Action(...))
seq = seq.append(Action(...))

# String representation
print(seq.to_string())  # "B-C-C"
```

### Transition Between Streets

```python
# Apply flop cards
flop_cards = (12, 25, 39)  # 2♥ 7♦ K♠
new_public = state.public_state.advance_to_flop(flop_cards)

# Reconstruct GameState (action history auto-reset)
new_state = GameState(
    public_state=new_public,
    private_states=state.private_states,
)

# Flop action starts fresh
print(len(new_state.public_state.action_history))  # 0
```

### Track Game History with Reach Probabilities

```python
history = GameHistory.empty()

# Add states with reach probabilities
history = history.append(state1, reach_prob=1.0)      # Initial
history = history.append(state2, reach_prob=1.0)      # Both players acted surely
history = history.append(state3, reach_prob=0.5)      # Sampled 50% def probability

# Later: weight regret updates
for i, (state, reach_prob) in enumerate(zip(history.states, history.reach_probs)):
    # regret_update[i] *= reach_prob  (importance sampling)
    pass
```

### Terminal State with Payoffs

```python
# Someone folded or showdown reached
payoffs = {
    0: 0.5,    # Player 0 wins 0.5 chips
    1: -0.5,   # Player 1 loses 0.5 chips
}

terminal_state = state.to_terminal(payoffs)
assert terminal_state.terminal == True
```

---

## Card Encoding Reference

The project uses **Treys library card encoding** (0-51):

```
Spades:   0-12   (2♠ to A♠)
Hearts:  13-25   (2♥ to A♥)
Diamonds:26-38   (2♦ to A♦)
Clubs:   39-51   (2♣ to A♣)

Examples:
  A♠ = 51
  K♦ = 36
  2♣ = 39
```

To convert: `cards[i] // 13 = suit`, `cards[i] % 13 = rank`

---

## Architecture at a Glance

```
PublicState                 PrivateState
├─ community_cards          ├─ player_idx
├─ action_history           └─ hole_cards
├─ current_street
├─ pot_size
└─ player_stacks
        │                           │
        └───────────┬───────────────┘
                    │
               GameState
                    ├─ public_state
                    ├─ private_states {0, 1}
                    ├─ terminal: bool
                    └─ payoffs: dict
                    
                    │
                    └──→ InformationSet
                         ├─ player_idx
                         ├─ hole_cards
                         ├─ public_hash (fast lookup key)
                         └─ public_str (for debugging)

GameHistory
├─ states: tuple of GameState
└─ reach_probs: tuple of floats
```

---

## Key Properties

| Class | Immutable? | Hashable? | Use For |
|-------|-----------|-----------|---------|
| `HoleCards` | ✅ | ✅ | Player's private cards |
| `Action` | ✅ | ✅ | Single action in game |
| `ActionSequence` | ✅ | ✅ | Public betting history |
| `PublicState` | ✅ | ✅ | Public game info |
| `PrivateState` | ✅ | ✅ | Private hole cards |
| `GameState` | ✅ | ❌ | Complete decision state |
| `InformationSet` | ✅ | ✅ | CFR strategy key |
| `GameHistory` | ✅ | ❌ | Full game trajectory |

---

## Validation

All invariants are checked at construction:

```python
# ❌ Invalid: duplicate cards
cards = HoleCards(51, 51)  # ValueError!

# ✌ Invalid: wrong street for card count
state = PublicState(
    community_cards=(1, 2, 3, 4),  # 4 cards
    current_street=Street.FLOP,     # expects 3
)  # ValueError!

# ✅ Valid: proper street/cards
state = PublicState(
    community_cards=(1, 2, 3),
    current_street=Street.FLOP,
)  # OK
```

---

## Performance Tips

1. **Information set hashing is O(1)**: Uses `@cached_property` memoization
2. **State transitions are O(1)**: Only concatenates tuples
3. **Reuse GameState objects**: They're immutable, so safe to share
4. **Dictionary lookups are O(1)**: Use `InformationSet` as keys directly

---

## Next: Phase 2 Integration

When building the game environment:

1. **Validate actions** in a separate `action_validator.py` module
2. **Compute payoffs** in a separate `payoff_calculator.py` module
3. **Store strategies** using `InformationSet` as dictionary keys
4. **Track reach probabilities** using `GameHistory`

Don't modify this module; it's designed to be stable and composable.

---

## Troubleshooting

**Q: Why Tuples instead of Lists?**  
A: Immutability prevents bugs. Use `tuple()` for structure sharing and hashability.

**Q: How do I update pot_size?**  
A: Create a new `PublicState` with the updated pot. Game logic (not this module) computes updates.

**Q: Where's the CFR algorithm?**  
A: Phase 2. This module is just state representation. Tree traversal comes next.

**Q: Can I use this with RLCard?**  
A: Yes, wrap RLCard's state → `GameState` via a converter function (Phase 2).

---

## Examples

See `tests/test_state_representation.py` for:
- 7 complete runnable examples
- CFR integration patterns
- History tracking patterns
- Street transitions

Run with:
```bash
python tests/test_state_representation.py
```

---

## Further Reading

- `src/env/STATE_ARCHITECTURE.md` - Detailed design document
- `PHASE_1_IMPLEMENTATION.md` - Complete implementation summary
- References in docstrings for CFR theory

---

**Status: ✅ Phase 1 Complete**  
**Ready for: Phase 2 (Environment Integration)**
