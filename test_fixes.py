#!/usr/bin/env python
"""Test the three critical bug fixes."""

from src.env.sequential_history import *

def test_bug_fix_1_history_accumulation():
    """Test that action history accumulates across streets (CFR bug fix)."""
    ps = PublicState(action_history=ActionSequence.empty(), current_street=Street.PREFLOP)
    action = Action(ActionType.BET, 0, Street.PREFLOP, 1.0)
    ps = ps.append_action(action)
    actions_before = len(ps.action_history)
    
    ps_flop = ps.advance_to_flop((0, 1, 2))
    actions_after = len(ps_flop.action_history)
    
    assert actions_after == actions_before, \
        f"BUG: History reset! Before={actions_before}, After={actions_after}"
    assert actions_after == 1, \
        f"BUG: Expected 1 action, got {actions_after}"
    print("✓ BUG FIX 1: History accumulates across streets (not reset)")
    return True


def test_bug_fix_2_frozen_dataclass_hash():
    """Test that frozen dataclass __hash__ works without FrozenInstanceError."""
    try:
        # This should work now without FrozenInstanceError
        hs = HoleCards(51, 36)
        h = hash(hs)
        print(f"✓ BUG FIX 2: HoleCards hash works: {h}")
        
        ps = PublicState()
        h2 = hash(ps)
        print(f"✓ BUG FIX 2: PublicState hash works: {h2}")
        
        return True
    except Exception as e:
        print(f"✗ FAIL BUG FIX 2: {e}")
        return False


def test_bug_fix_3_immutable_private_states():
    """Test that GameState.private_states is a tuple, not dict."""
    gs = create_initial_state(HoleCards(51, 36), HoleCards(50, 49))
    
    # Check type
    assert isinstance(gs.private_states, tuple), \
        f"BUG: private_states should be tuple, got {type(gs.private_states)}"
    assert len(gs.private_states) == 2, \
        f"BUG: private_states should have 2 elements, got {len(gs.private_states)}"
    
    # Check immutability
    try:
        gs.private_states[0] = None
        print("✗ FAIL BUG FIX 3: private_states is not immutable!")
        return False
    except TypeError:
        print("✓ BUG FIX 3: GameState.private_states is immutable tuple (not dict)")
        return True


def test_information_set_hashing():
    """Test that InformationSet hashing works correctly."""
    ps = PublicState(action_history=ActionSequence.empty(), current_street=Street.PREFLOP)
    priv = PrivateState(player_idx=0, hole_cards=HoleCards(51, 36))
    infoset = InformationSet.from_states(ps, priv)
    
    # Should be hashable and work as dict key
    strategy_table = {infoset: 0.5}
    assert strategy_table[infoset] == 0.5
    print("✓ BONUS: InformationSet hashing works for CFR strategy lookups")
    return True


if __name__ == "__main__":
    print("Testing 3 critical bug fixes...\n")
    
    results = [
        test_bug_fix_1_history_accumulation(),
        test_bug_fix_2_frozen_dataclass_hash(),
        test_bug_fix_3_immutable_private_states(),
        test_information_set_hashing(),
    ]
    
    print(f"\n{'='*60}")
    if all(results):
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
