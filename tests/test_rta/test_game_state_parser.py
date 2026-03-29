"""
Comprehensive unit tests for LiveGameStateBuilder (src/rta/game_state_parser.py).

Tests cover:
1. Multi-way all-in SPR calculation (CRITICAL BUG FIX)
2. Invalid player ID handling (bounds checking)
3. Event ordering validation (state machine)
"""

import pytest
from src.rta.game_state_parser import LiveGameStateBuilder, GameStateTracker


class TestLiveGameStateBuilderAllIn:
    """Test suite for all-in scenarios and SPR calculation."""

    def test_multi_way_all_in_spr_calculation(self) -> None:
        """
        CRITICAL: Test that _process_action does NOT crash with 
        ValueError: min() arg is an empty sequence during multi-way all-ins.
        
        Scenario:
        - 6 players, 200 BB initial stack each
        - Preflop: Players 0–4 go all-in, stacks become 0
        - Action on Player 5
        - SPR calculation must handle empty active_stacks list gracefully
        
        Expected:
        - No ValueError raised
        - spr_before calculated safely
        - Logs warning about all-in situation
        """
        # Setup
        builder = LiveGameStateBuilder(num_players=6, big_blind=2.0, initial_stack_bb=200.0)
        
        # Deal hole cards
        builder.process_event({
            "type": "deal",
            "hand": ["AS", "KS"],
            "position": 5
        })
        
        assert builder.tracker.cards_dealt
        assert builder.tracker.my_hand == ["AS", "KS"]
        assert builder.tracker.my_position == 5
        
        # Preflop actions: Players 0–4 all go all-in
        # Simulate:
        # Player 0: raises to 100 (stack: 200 -> 100)
        # Player 1: raises to 300 (all-in, stack: 200 -> 0)
        # Player 2: raises to 500 (all-in, stack: 200 -> 0)
        # Player 3: raises to 700 (all-in, stack: 200 -> 0)
        # Player 4: goes all-in 200 (stack: 200 -> 0)
        
        # Player 0 raises to 100 BBs
        builder.process_event({
            "type": "action",
            "player": 0,
            "action": "raise_pot",
            "amount": 100.0
        })
        assert builder.tracker.stacks[0] == 100.0  # 200 - 100
        assert builder.tracker.pot == 100.0
        
        # Player 1 goes all-in
        builder.process_event({
            "type": "action",
            "player": 1,
            "action": "all_in",
            "amount": 200.0
        })
        assert builder.tracker.stacks[1] == 0.0  # 200 - 200 (all-in)
        assert builder.tracker.pot == 300.0
        
        # Player 2 goes all-in
        builder.process_event({
            "type": "action",
            "player": 2,
            "action": "all_in",
            "amount": 200.0
        })
        assert builder.tracker.stacks[2] == 0.0
        assert builder.tracker.pot == 500.0
        
        # Player 3 goes all-in
        builder.process_event({
            "type": "action",
            "player": 3,
            "action": "all_in",
            "amount": 200.0
        })
        assert builder.tracker.stacks[3] == 0.0
        assert builder.tracker.pot == 700.0
        
        # Player 4 goes all-in
        builder.process_event({
            "type": "action",
            "player": 4,
            "action": "all_in",
            "amount": 200.0
        })
        assert builder.tracker.stacks[4] == 0.0
        assert builder.tracker.pot == 900.0
        
        # At this point: players 0-4 have stacks [100, 0, 0, 0, 0], player 5 has 200
        # Next action on player 0 should NOT crash with "min() arg is an empty sequence"
        # This is the CRITICAL TEST: can we safely compute effective_stack?
        
        # Player 0 passes (would be street -> assumption is we are properly tracking)
        # Now critical: Player 5 action when stacks are [100, 0, 0, 0, 0, Y]
        pot_before_p5 = builder.tracker.pot
        
        # CRITICAL: This action should NOT raise ValueError: min() arg is an empty sequence
        builder.process_event({
            "type": "action",
            "player": 5,
            "action": "check",
            "amount": 0.0
        })
        
        # Verify betting history was recorded
        assert len(builder.tracker.betting_history) > 0
        
        # Check last action (Player 5's check)
        last_action = builder.tracker.betting_history[-1]
        assert last_action["player"] == 5
        assert last_action["action"] == 1  # CHECK_CALL = 1
        assert last_action["pot_before"] == pot_before_p5
        
        # CRITICAL: spr_before must be calculated without crash
        assert "spr_before" in last_action
        assert isinstance(last_action["spr_before"], float)
        assert last_action["spr_before"] >= 0.0
        
        # Verify the state is still consistent
        state = builder.get_state()
        assert state is not None
        assert state["pot"] > 0
        assert state["my_chips"] >= 0

    def test_all_stacks_zero_fallback(self) -> None:
        """
        Test fallback behavior when ALL stacks are exactly 0.
        This is an edge case where even max() returns 0.
        """
        builder = LiveGameStateBuilder(num_players=3, big_blind=1.0, initial_stack_bb=100.0)
        
        # Deal
        builder.process_event({
            "type": "deal",
            "hand": ["AS", "KS"],
            "position": 2
        })
        
        # Manually zero all stacks (representing all-in)
        builder.tracker.stacks = [0.0, 0.0, 0.0]
        builder.tracker.stakes = [100.0, 100.0, 0.0]  # Everyone committed
        builder.tracker.pot = 200.0
        
        # Action with all stacks = 0 should NOT crash
        builder.process_event({
            "type": "action",
            "player": 0,
            "action": "fold",
            "amount": 0.0
        })
        
        # Verify fold was recorded
        assert builder.tracker.betting_history[-1]["action"] == 0  # FOLD = 0
        assert builder.tracker.betting_history[-1]["spr_before"] == 0.0  # All stacks = 0

    def test_spr_with_zero_pot_preflop(self) -> None:
        """
        Test SPR calculation on preflop (pot_before = 0, no antes).
        Should not divide by zero.
        """
        builder = LiveGameStateBuilder(num_players=2, big_blind=2.0)
        
        # Fresh hand, no antes, pot = 0
        builder.process_event({
            "type": "deal",
            "hand": ["AH", "AD"],
            "position": 0
        })
        
        assert builder.tracker.pot == 0.0
        
        # Action with pot_before = 0
        builder.process_event({
            "type": "action",
            "player": 1,
            "action": "raise_pot",
            "amount": 4.0
        })
        
        # spr_before should be 0 (pot_before = 0)
        last_action = builder.tracker.betting_history[-1]
        assert last_action["spr_before"] == 0.0  # 0 / 0 handled as 0

    def test_invalid_player_id_bounds_check(self) -> None:
        """
        Test that invalid player IDs are rejected gracefully.
        
        Cases:
        - player_id = -1 (negative)
        - player_id = num_players (out of range)
        - player_id >> num_players (way out)
        """
        builder = LiveGameStateBuilder(num_players=6, big_blind=2.0)
        
        builder.process_event({
            "type": "deal",
            "hand": ["AS", "KS"],
            "position": 2
        })
        
        initial_history_len = len(builder.tracker.betting_history)
        
        # Invalid: player_id = -1
        builder.process_event({
            "type": "action",
            "player": -1,
            "action": "fold",
            "amount": 0.0
        })
        
        # Action should be skipped, no new entry in betting_history
        assert len(builder.tracker.betting_history) == initial_history_len
        
        # Invalid: player_id = 6 (out of range for 6-player game)
        builder.process_event({
            "type": "action",
            "player": 6,
            "action": "check",
            "amount": 0.0
        })
        
        assert len(builder.tracker.betting_history) == initial_history_len
        
        # Invalid: player_id = 100
        builder.process_event({
            "type": "action",
            "player": 100,
            "action": "raise_pot",
            "amount": 10.0
        })
        
        assert len(builder.tracker.betting_history) == initial_history_len

    def test_state_machine_action_before_deal(self) -> None:
        """
        Test that "action" events are ignored before "deal" event.
        Ensures state machine prevents malformed event sequences.
        """
        builder = LiveGameStateBuilder(num_players=6, big_blind=2.0)
        
        # WRONG: action before deal
        builder.process_event({
            "type": "action",
            "player": 0,
            "action": "fold",
            "amount": 0.0
        })
        
        # Action should be skipped
        assert len(builder.tracker.betting_history) == 0
        assert not builder.tracker.cards_dealt
        
        # NOW: proper deal
        builder.process_event({
            "type": "deal",
            "hand": ["AS", "KS"],
            "position": 2
        })
        
        assert builder.tracker.cards_dealt
        
        # CORRECT: action after deal should be processed
        builder.process_event({
            "type": "action",
            "player": 0,
            "action": "fold",
            "amount": 0.0
        })
        
        assert len(builder.tracker.betting_history) == 1

    def test_hand_start_resets_state(self) -> None:
        """Test that "hand_start" event properly resets state machine."""
        builder = LiveGameStateBuilder(num_players=6, big_blind=2.0)
        
        # Setup initial state
        builder.process_event({
            "type": "deal",
            "hand": ["AS", "KS"],
            "position": 1
        })
        
        builder.process_event({
            "type": "action",
            "player": 0,
            "action": "fold",
            "amount": 0.0
        })
        
        assert builder.tracker.cards_dealt
        assert len(builder.tracker.betting_history) == 1
        
        # New hand
        builder.process_event({
            "type": "hand_start"
        })
        
        # Verify reset
        assert not builder.tracker.cards_dealt
        assert not builder.tracker.hand_started  # Actually this gets set to False, then True on deal
        assert len(builder.tracker.betting_history) == 0
        assert builder.tracker.my_hand == []


class TestGameStateTracker:
    """Unit tests for GameStateTracker dataclass."""

    def test_initialization(self) -> None:
        """Test default initialization of GameStateTracker."""
        tracker = GameStateTracker()
        
        assert tracker.num_players == 6
        assert tracker.big_blind == 2.0
        assert tracker.pot == 0.0
        assert tracker.cards_dealt is False
        assert tracker.hand_started is False
        assert len(tracker.stacks) == 6
        assert all(s == 200.0 * 2.0 for s in tracker.stacks)

    def test_to_raw_state(self) -> None:
        """Test export to raw state dict."""
        tracker = GameStateTracker(num_players=4)
        tracker.my_position = 1
        tracker.my_hand = ["AH", "KH"]
        tracker.pot = 100.0
        tracker.current_street = 0
        
        state = tracker.to_raw_state()
        
        assert state["hand"] == ["AH", "KH"]
        assert state["pot"] == 100.0
        assert state["my_chips"] == tracker.stacks[1]
        assert state["position"] == 1
        assert "legal_actions" in state
        assert "opponent_chips" in state
        assert len(state["opponent_chips"]) == 3  # 4 players - 1 (self)


class TestActionMapping:
    """Test action string to index mapping."""

    def test_action_name_mapping(self) -> None:
        """Test that all action names map to correct indices."""
        builder = LiveGameStateBuilder(num_players=6, big_blind=2.0)
        
        test_cases = [
            ("fold", 0),
            ("check", 1),
            ("call", 1),
            ("min_raise", 2),
            ("raise_0.25x_pot", 3),
            ("raise_0.33x_pot", 4),
            ("raise_0.5x_pot", 5),
            ("raise_0.75x_pot", 6),
            ("raise_1.0x_pot", 7),
            ("raise_pot", 7),
            ("raise_1.5x_pot", 8),
            ("raise_150", 8),
            ("raise_2.0x_pot", 9),
            ("raise_2x", 9),
            ("all_in", 10),
            ("allin", 10),
        ]
        
        for action_name, expected_idx in test_cases:
            actual_idx = builder._map_action_name_to_index(action_name, 100.0)
            assert actual_idx == expected_idx, f"Action '{action_name}' mapped to {actual_idx}, expected {expected_idx}"

    def test_unknown_action_defaults_to_fold(self) -> None:
        """Test that unknown action names default to fold (index 0)."""
        builder = LiveGameStateBuilder(num_players=6, big_blind=2.0)
        
        unknown_actions = ["bet_50", "weird_action", "foobar", ""]
        
        for action_name in unknown_actions:
            idx = builder._map_action_name_to_index(action_name, 100.0)
            assert idx == 0, f"Unknown action '{action_name}' should default to Fold (0), got {idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
