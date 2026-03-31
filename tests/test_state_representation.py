"""
Examples and tests for Phase 1 state representation.

Demonstrates how to:
  1. Create game states
  2. Build action sequences
  3. Construct information sets
  4. Use for CFR tree traversal
"""

from src.env.sequential_history import (
    ActionType,
    Street,
    Action,
    HoleCards,
    ActionSequence,
    PublicState,
    PrivateState,
    GameState,
    GameHistory,
    InformationSet,
    create_initial_state,
    GameStateEncoder,
)


def example_1_basic_state_creation():
    """Example 1: Creating an initial game state."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Creating Initial Game State")
    print("="*70)
    
    # Define hole cards for both players
    p0_cards = HoleCards(51, 36)  # A♠ K♦
    p1_cards = HoleCards(50, 49)  # K♠ Q♠
    
    # Create initial state (preflop, cards dealt, blinds posted)
    state = create_initial_state(
        p0_hole_cards=p0_cards,
        p1_hole_cards=p1_cards,
        initial_stacks=(100.0, 100.0),
        button_idx=0,
        small_blind=0.01,
        big_blind=0.02,
    )
    
    print(f"Initial state:\n{state}\n")
    print(f"Public state:\n{state.public_state}\n")
    print(f"Player 0 cards: {state.private_states[0].hole_cards.cards}")
    print(f"Player 1 cards: {state.private_states[1].hole_cards.cards}")
    print(f"Pot: {state.public_state.pot_size}")
    print(f"Stacks: P0={state.public_state.player_stacks[0]:.2f}, "
          f"P1={state.public_state.player_stacks[1]:.2f}")


def example_2_action_sequence():
    """Example 2: Building an immutable action sequence."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Building Immutable Action Sequence")
    print("="*70)
    
    # Start with empty sequence
    seq = ActionSequence.empty()
    print(f"Empty sequence: {len(seq)} actions")
    
    # Action 1: P0 bets 1 BB preflop
    action1 = Action(
        action_type=ActionType.BET,
        player_idx=0,
        street=Street.PREFLOP,
        bet_size=1.0,
    )
    seq = seq.append(action1)
    print(f"\nAfter P0 bets 1 BB: {len(seq)} action(s)")
    print(f"  Sequence string: {seq.to_string()}")
    
    # Action 2: P1 calls
    action2 = Action(
        action_type=ActionType.CALL,
        player_idx=1,
        street=Street.PREFLOP,
        bet_size=1.0,
    )
    seq = seq.append(action2)
    print(f"\nAfter P1 calls 1 BB: {len(seq)} action(s)")
    print(f"  Sequence string: {seq.to_string()}")
    
    # Action 3: P0 checks flop
    action3 = Action(
        action_type=ActionType.CHECK,
        player_idx=0,
        street=Street.FLOP,
        bet_size=0.0,
    )
    seq = seq.append(action3)
    print(f"\nAfter P0 checks flop: {len(seq)} action(s)")
    print(f"  Sequence string: {seq.to_string()}")
    
    # Demonstrate immutability (old sequence unchanged)
    seq2 = ActionSequence.empty().append(action1)
    assert len(seq2) == 1
    assert len(seq) == 3
    print(f"\nOriginal first sequence still has {len(seq2)} action(s) ✓")


def example_3_game_state_transitions():
    """Example 3: Transitioning game state through streets."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Game State Transitions Through Streets")
    print("="*70)
    
    # Initial state
    state = create_initial_state(HoleCards(51, 36), HoleCards(50, 49))
    print(f"Initial street: {state.public_state.current_street}")
    print(f"Community cards: {len(state.public_state.community_cards)}")
    
    # Preflop action: P0 bets
    action1 = Action(ActionType.BET, 0, Street.PREFLOP, 1.0)
    state = state.append_action(action1)
    print(f"\nAfter P0 bets: {state.public_state.action_history.to_string()}")
    
    # Preflop action: P1 calls
    action2 = Action(ActionType.CALL, 1, Street.PREFLOP, 1.0)
    state = state.append_action(action2)
    print(f"After P1 calls: {state.public_state.action_history.to_string()}")
    
    # Move to flop
    flop_cards = (12, 25, 39)  # 2♥ 7♦ K♠
    new_public = state.public_state.advance_to_flop(flop_cards)
    state = GameState(
        public_state=new_public,
        private_states=state.private_states,
        terminal=False,
        payoffs={},
    )
    print(f"\nAfter advancing to flop:")
    print(f"  Street: {state.public_state.current_street}")
    print(f"  Community cards: {len(state.public_state.community_cards)}")
    print(f"  Action history reset: {len(state.public_state.action_history)} actions")
    
    # Flop action
    action3 = Action(ActionType.CHECK, 0, Street.FLOP, 0.0)
    state = state.append_action(action3)
    print(f"  After P0 checks: {state.public_state.action_history.to_string()}")


def example_4_information_sets():
    """Example 4: Information sets for CFR strategy storage."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Information Sets (CFR Strategy Keys)")
    print("="*70)
    
    # Create two different game states
    state1 = create_initial_state(HoleCards(51, 36), HoleCards(50, 49))
    state2 = create_initial_state(HoleCards(51, 36), HoleCards(50, 49))
    
    # Get player 0's information set
    infoset1 = state1.get_infoset(0)
    infoset2 = state2.get_infoset(0)
    
    print(f"StateSet 1 infoset:\n  {infoset1}")
    print(f"State 2 infoset:\n  {infoset2}")
    
    same_infoset = (
        infoset1.player_idx == infoset2.player_idx and
        infoset1.hole_cards == infoset2.hole_cards and
        infoset1.public_hash == infoset2.public_hash
    )
    print(f"\nSame infoset? {same_infoset}")
    
    # Demonstrate infoset hashing for O(1) lookups
    strategy_table = {}
    strategy_table[infoset1] = {
        ActionType.BET: 0.4,
        ActionType.CHECK: 0.6,
    }
    
    # Same infoset → can look up immediately
    strategy = strategy_table.get(infoset2)
    if strategy:
        print(f"\nStrategy for same infoset:\n  {strategy}")
    
    # Different public history → different infoset
    action = Action(ActionType.BET, 0, Street.PREFLOP, 1.0)
    state3 = state1.append_action(action)
    infoset3 = state3.get_infoset(1)
    
    print(f"\nInfoset after P0 action:\n  {infoset3}")
    print(f"Hash changed from {infoset2.public_hash} → {infoset3.public_hash}")


def example_5_game_history():
    """Example 5: Tracking full game history with reach probabilities."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Game History and Reach Probabilities")
    print("="*70)
    
    # Build a simple game trajectory
    state = create_initial_state(HoleCards(51, 36), HoleCards(50, 49))
    history = GameHistory.empty()
    
    # State 0: Initial
    history = history.append(state, reach_prob=1.0)
    print(f"State 0 (initial): reach_prob=1.0")
    
    # State 1: P0 bets
    action = Action(ActionType.BET, 0, Street.PREFLOP, 1.0)
    state = state.append_action(action)
    history = history.append(state, reach_prob=1.0)
    print(f"State 1 (P0 bets): reach_prob=1.0")
    
    # State 2: P1 calls with prob 0.7
    action = Action(ActionType.CALL, 1, Street.PREFLOP, 1.0)
    state = state.append_action(action)
    history = history.append(state, reach_prob=0.7)
    print(f"State 2 (P1 calls): reach_prob=0.7")
    
    # State 3: Terminal (showdown)
    payoffs = {0: -0.03, 1: 0.03}  # P1 wins ~3 chips
    terminal_state = state.to_terminal(payoffs)
    history = history.append(terminal_state, reach_prob=0.7)
    print(f"State 3 (terminal): reach_prob=0.7, payoffs={payoffs}")
    
    print(f"\nTotal trajectory length: {len(history)}")
    print(f"Final state is terminal: {history.final_state().terminal}")


def example_6_encoding_for_neural_nets():
    """Example 6: Preparing state for neural network (Phase 2-3)."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Encoding GameState for Neural Networks")
    print("="*70)
    
    state = create_initial_state(HoleCards(51, 36), HoleCards(50, 49))
    
    # Encode hole cards
    cards = GameStateEncoder.encode_hole_cards(
        state.private_states[0].hole_cards
    )
    print(f"Player 0 hole cards (indices): {cards}")
    
    # Encode community cards
    board = GameStateEncoder.encode_community_cards(
        state.public_state.community_cards
    )
    print(f"Community cards: {board} (empty preflop)")
    
    # Add actions, encode sequence
    action1 = Action(ActionType.BET, 0, Street.PREFLOP, 1.0)
    state = state.append_action(action1)
    
    action2 = Action(ActionType.CALL, 1, Street.PREFLOP, 1.0)
    state = state.append_action(action2)
    
    action_strs = GameStateEncoder.encode_action_sequence(
        state.public_state.action_history
    )
    print(f"Action sequence strings: {action_strs}")
    print(f"\nNote: Full neural network integration comes in Phase 2-3")


def example_7_cfr_workflow():
    """Example 7: Simplified CFR tree traversal workflow."""
    print("\n" + "="*70)
    print("EXAMPLE 7: CFR Counterfactual Tree Traversal Pattern")
    print("="*70)
    
    # Set up strategy table and regret buffers
    infoset_strategy = {}  # {InformationSet: {ActionType: probability}}
    infoset_regrets = {}   # {InformationSet: {ActionType: cumulative_regret}}
    
    # Initial state
    state = create_initial_state(HoleCards(51, 36), HoleCards(50, 49))
    reach_prob_p0 = 1.0
    reach_prob_p1 = 1.0
    
    print("Iteration 1 of CFR traversal:")
    print(f"  Initial reach_prob(P0)={reach_prob_p0}, reach_prob(P1)={reach_prob_p1}")
    
    # Get player 0's infoset
    infoset_p0 = state.get_infoset(0)
    
    # Initialize strategy if first time seeing this infoset
    if infoset_p0 not in infoset_strategy:
        infoset_strategy[infoset_p0] = {
            ActionType.BET: 0.5,
            ActionType.CHECK: 0.5,
        }
        print(f"  First time seeing infoset, initialized uniform strategy")
    
    # Get strategy at this infoset
    sigma = infoset_strategy[infoset_p0]
    print(f"  Strategy at infoset: {sigma}")
    
    # Simulate: P0 bets (action prob = 0.5)
    action_p0 = ActionType.BET
    action_prob_p0 = sigma[action_p0]
    reach_prob_p1 = reach_prob_p1 * action_prob_p0  # Counterfactual reach
    
    action = Action(action_p0, 0, Street.PREFLOP, 1.0)
    state = state.append_action(action)
    print(f"\n  P0 chooses {action_p0}, counterfactual reach=({reach_prob_p0}, {reach_prob_p1})")
    
    # Get player 1's infoset
    infoset_p1 = state.get_infoset(1)
    
    if infoset_p1 not in infoset_strategy:
        infoset_strategy[infoset_p1] = {
            ActionType.FOLD: 0.3,
            ActionType.CALL: 0.7,
        }
        print(f"  First time seeing infoset, initialized strategy")
    
    sigma_p1 = infoset_strategy[infoset_p1]
    print(f"  Strategy at infoset: {sigma_p1}")
    
    # Simulate: P1 calls (action prob = 0.7)
    action_p1 = ActionType.CALL
    action_prob_p1 = sigma_p1[action_p1]
    reach_prob_p0 = reach_prob_p0 * action_prob_p1  # Counterfactual reach
    
    action = Action(action_p1, 1, Street.PREFLOP, 1.0)
    state = state.append_action(action)
    print(f"  P1 chooses {action_p1}, counterfactual reach=({reach_prob_p0}, {reach_prob_p1})")
    
    # Move to flop
    new_public = state.public_state.advance_to_flop((12, 25, 39))  # 2♥ 7♦ K♠
    state = GameState(
        public_state=new_public,
        private_states=state.private_states,
        terminal=False,
        payoffs={},
    )
    
    print(f"\nFlop: {new_public.community_cards}")
    print(f"Reach probabilities at flop: P0={reach_prob_p0:.3f}, P1={reach_prob_p1:.3f}")
    print("\nNote: Full CFR would continue tree traversal with value computation")


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PHASE 1: Game State Representation Examples".center(68) + "█")
    print("█" + "  VR-DeepPDCFR+ Foundation".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    example_1_basic_state_creation()
    example_2_action_sequence()
    example_3_game_state_transitions()
    example_4_information_sets()
    example_5_game_history()
    example_6_encoding_for_neural_nets()
    example_7_cfr_workflow()
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  All examples completed successfully! ✓".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
