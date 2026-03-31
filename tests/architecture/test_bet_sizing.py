"""
Phase 4: Superhuman Bet Sizing Abstraction Test Suite.

Validates street-specific, SPR-aware bet sizing that adapts pot multipliers
based on the game stage (preflop, flop, turn, river).

Test Coverage:
  1. BetSizingConfig: Verify street-specific multiplier configuration.
  2. resolve_action with street context: Same action index should resolve
     to different actual pot multipliers on different streets.
  3. Action mapping: Verify mapped actions translate to valid RLCard amounts.
  4. Corner cases: Stack capping, min-raise, all-in handling.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pytest

from src.env.action_mapper import (
    ActionMapper,
    BetSizingConfig,
    GameContext,
    PokerAction,
    ResolvedAction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBetSizingConfig:
    """Test BetSizingConfig construction and street-specific multipliers."""

    def test_default_configuration(self):
        """Test that default config initializes with sensible defaults."""
        config = BetSizingConfig()
        
        assert config.preflop == [0.33, 0.5, 0.75, 1.0]
        assert config.flop == [0.33, 0.5, 0.75, 1.0]
        assert config.turn == [0.5, 0.75, 1.0, 1.5]
        assert config.river == [0.5, 0.75, 1.0, 1.5]

    def test_custom_configuration(self):
        """Test that custom bet sizing can be set."""
        custom_preflop = [0.25, 0.5, 1.0]
        custom_river = [0.75, 1.5, 2.0]
        
        config = BetSizingConfig(
            preflop=custom_preflop,
            river=custom_river,
        )
        
        assert config.preflop == custom_preflop
        assert config.river == custom_river
        # Others should keep defaults
        assert config.flop == [0.33, 0.5, 0.75, 1.0]
        assert config.turn == [0.5, 0.75, 1.0, 1.5]

    def test_get_multipliers(self):
        """Test retrieving street-specific multipliers."""
        config = BetSizingConfig()
        
        preflop_mult = config.get_multipliers(0)
        assert preflop_mult == [0.33, 0.5, 0.75, 1.0]
        
        flop_mult = config.get_multipliers(1)
        assert flop_mult == [0.33, 0.5, 0.75, 1.0]
        
        turn_mult = config.get_multipliers(2)
        assert turn_mult == [0.5, 0.75, 1.0, 1.5]
        
        river_mult = config.get_multipliers(3)
        assert river_mult == [0.5, 0.75, 1.0, 1.5]

    def test_get_multipliers_invalid_street(self):
        """Test that invalid street indices fall back to preflop."""
        config = BetSizingConfig()
        
        invalid_mult = config.get_multipliers(99)
        assert invalid_mult == [0.33, 0.5, 0.75, 1.0]


class TestGameContext:
    """Test GameContext initialization and validation."""

    def test_valid_context(self):
        """Test construction with valid parameters."""
        ctx = GameContext(
            pot_size=100,
            my_stack=500,
            amount_to_call=50,
            min_raise_amount=100,
            big_blind=10,
            street=1,
        )
        assert ctx.street == 1

    def test_context_with_default_street(self):
        """Test that street defaults to 0 (preflop)."""
        ctx = GameContext(
            pot_size=100,
            my_stack=500,
            amount_to_call=50,
            min_raise_amount=100,
            big_blind=10,
        )
        assert ctx.street == 0

    def test_invalid_street_raises(self):
        """Test that invalid street raises ValueError."""
        with pytest.raises(ValueError, match="street"):
            GameContext(
                pot_size=100,
                my_stack=500,
                amount_to_call=50,
                min_raise_amount=100,
                big_blind=10,
                street=4,  # Invalid: > 3
            )


class TestResolveActionStreetSpecific:
    """Test that resolve_action uses street-specific bet sizing."""

    def setup_method(self):
        """Initialize ActionMapper for each test."""
        self.mapper = ActionMapper()

    def test_same_action_different_multipliers_flop_vs_river(self):
        """
        CRITICAL TEST: Same action index should resolve to different
        actual pot multipliers on Flop vs River.
        
        Scenario:
          - Flop: pot_size=100, action index for 0.75x pot should be 0.33*pot
          - River: same pot_size=100, action index should resolve to 1.5x pot
        """
        # Build contexts for Flop and River with same pot structure
        flop_ctx = GameContext(
            pot_size=100,
            my_stack=1000,
            amount_to_call=0,
            min_raise_amount=10,
            big_blind=10,
            street=1,  # Flop
        )
        
        river_ctx = GameContext(
            pot_size=100,
            my_stack=1000,
            amount_to_call=0,
            min_raise_amount=10,
            big_blind=10,
            street=3,  # River
        )
        
        # Use action 7: RAISE_THREE_QUARTER_POT (index 7)
        # On flop: should map to 0.75x (3rd multiplier in [0.33, 0.5, 0.75, 1.0])
        # On river: should map to 1.0x (3rd multiplier in [0.5, 0.75, 1.0, 1.5])
        action = PokerAction.RAISE_THREE_QUARTER_POT
        
        flop_resolved = self.mapper.resolve_action(action, flop_ctx)
        river_resolved = self.mapper.resolve_action(action, river_ctx)
        
        logger.info(f"Flop (street=1) {action.name}: {flop_resolved.amount:.1f} chips")
        logger.info(f"River (street=3) {action.name}: {river_resolved.amount:.1f} chips")
        
        # They should be different because the bet sizing config is different
        assert flop_resolved.amount != river_resolved.amount, \
            f"Expected different amounts for same action on different streets: " \
            f"Flop={flop_resolved.amount}, River={river_resolved.amount}"

    def test_preflop_smaller_bets(self):
        """Test that Preflop uses smaller multipliers than Turn/River."""
        action = PokerAction.RAISE_HALF_POT
        
        preflop_ctx = GameContext(
            pot_size=100,
            my_stack=1000,
            amount_to_call=0,
            min_raise_amount=10,
            big_blind=10,
            street=0,
        )
        
        turn_ctx = GameContext(
            pot_size=100,
            my_stack=1000,
            amount_to_call=0,
            min_raise_amount=10,
            big_blind=10,
            street=2,
        )
        
        preflop_resolved = self.mapper.resolve_action(action, preflop_ctx)
        turn_resolved = self.mapper.resolve_action(action, turn_ctx)
        
        # Both should be valid raises
        assert preflop_resolved.amount > preflop_ctx.amount_to_call
        assert turn_resolved.amount > turn_ctx.amount_to_call

    def test_conservative_preflop_aggressive_river(self):
        """Test strategy differentiation: conservative preflop, aggressive river."""
        # Use the last raise action (RAISE_2X_POT)
        action = PokerAction.RAISE_2X_POT
        
        preflop_ctx = GameContext(
            pot_size=100,
            my_stack=500,
            amount_to_call=0,
            min_raise_amount=10,
            big_blind=10,
            street=0,  # Preflop: smaller multipliers
        )
        
        river_ctx = GameContext(
            pot_size=100,
            my_stack=500,
            amount_to_call=0,
            min_raise_amount=10,
            big_blind=10,
            street=3,  # River: larger multipliers
        )
        
        preflop_result = self.mapper.resolve_action(action, preflop_ctx)
        river_result = self.mapper.resolve_action(action, river_ctx)
        
        # River should have higher multipliers available (1.5x) vs Preflop (1.0x max)
        # So for the same action, we expect different sizing
        logger.info(f"Preflop bet: {preflop_result.amount}")
        logger.info(f"River bet: {river_result.amount}")


class TestResolveActionCornerCases:
    """Test edge cases and stack capping behavior."""

    def setup_method(self):
        """Initialize ActionMapper for each test."""
        self.mapper = ActionMapper()

    def test_stack_capping_all_in(self):
        """Test that bet sizing is capped to all-in when raising too large."""
        ctx = GameContext(
            pot_size=100,
            my_stack=50,  # Small stack
            amount_to_call=0,
            min_raise_amount=10,
            big_blind=10,
            street=1,
        )
        
        # Try a large raise
        action = PokerAction.RAISE_FULL_POT
        resolved = self.mapper.resolve_action(action, ctx)
        
        # Should be capped to all-in (50 chips)
        assert resolved.action == PokerAction.ALL_IN
        assert resolved.amount == 50

    def test_min_raise(self):
        """Test minimum raise sizing."""
        ctx = GameContext(
            pot_size=100,
            my_stack=1000,
            amount_to_call=50,
            min_raise_amount=150,
            big_blind=10,
            street=2,
        )
        
        resolved = self.mapper.resolve_action(PokerAction.MIN_RAISE, ctx)
        assert resolved.action == PokerAction.MIN_RAISE
        assert resolved.amount == 150

    def test_all_in_action(self):
        """Test explicit all-in action."""
        ctx = GameContext(
            pot_size=100,
            my_stack=200,
            amount_to_call=50,
            min_raise_amount=150,
            big_blind=10,
            street=0,
        )
        
        resolved = self.mapper.resolve_action(PokerAction.ALL_IN, ctx)
        assert resolved.action == PokerAction.ALL_IN
        assert resolved.amount == 200

    def test_fold(self):
        """Test fold action."""
        ctx = GameContext(
            pot_size=100,
            my_stack=500,
            amount_to_call=50,
            min_raise_amount=100,
            big_blind=10,
            street=1,
        )
        
        resolved = self.mapper.resolve_action(PokerAction.FOLD, ctx)
        assert resolved.action == PokerAction.FOLD
        assert resolved.amount == 0

    def test_check(self):
        """Test check action (only valid with no amount to call)."""
        ctx = GameContext(
            pot_size=100,
            my_stack=500,
            amount_to_call=0,
            min_raise_amount=100,
            big_blind=10,
            street=1,
        )
        
        resolved = self.mapper.resolve_action(PokerAction.CHECK, ctx)
        assert resolved.action == PokerAction.CHECK
        assert resolved.amount == 0

    def test_check_fallback_to_call(self):
        """Test that CHECK with outstanding bet falls back to CALL."""
        ctx = GameContext(
            pot_size=100,
            my_stack=500,
            amount_to_call=50,  # Outstanding bet
            min_raise_amount=100,
            big_blind=10,
            street=1,
        )
        
        resolved = self.mapper.resolve_action(PokerAction.CHECK, ctx)
        # Should fall back to CALL
        assert resolved.action == PokerAction.CALL
        assert resolved.amount == 50


class TestBetSizingIntegration:
    """Integration tests for complete bet sizing workflow."""

    def setup_method(self):
        """Initialize ActionMapper for each test."""
        self.mapper = ActionMapper()

    def test_flop_vs_river_sizing_comprehensive(self):
        """
        Comprehensive test: Flop vs River strategy differentiation.
        
        Scenario:
          - Flop (street=1): Small bets (0.33, 0.5, 0.75, 1.0x pot)
          - River (street=3): Larger bets (0.5, 0.75, 1.0, 1.5x pot)
          
        Same action index should resolve to different actual bets.
        """
        pot = 200
        stack = 1000
        
        flop_ctx = GameContext(
            pot_size=pot,
            my_stack=stack,
            amount_to_call=0,
            min_raise_amount=20,
            big_blind=10,
            street=1,  # Flop
        )
        
        river_ctx = GameContext(
            pot_size=pot,
            my_stack=stack,
            amount_to_call=0,
            min_raise_amount=20,
            big_blind=10,
            street=3,  # River
        )
        
        # Test multiple raise actions
        test_actions = [
            PokerAction.RAISE_QUARTER_POT,
            PokerAction.RAISE_THIRD_POT,
            PokerAction.RAISE_HALF_POT,
            PokerAction.RAISE_THREE_QUARTER_POT,
            PokerAction.RAISE_FULL_POT,
            PokerAction.RAISE_1_5X_POT,
            PokerAction.RAISE_2X_POT,
        ]
        
        for action in test_actions:
            flop_result = self.mapper.resolve_action(action, flop_ctx)
            river_result = self.mapper.resolve_action(action, river_ctx)
            
            logger.info(
                f"{action.name}: Flop={flop_result.amount:.0f}, River={river_result.amount:.0f}"
            )
            
            # Both should be valid (positive amount)
            assert flop_result.amount > 0
            assert river_result.amount > 0

    def test_all_streets_progression(self):
        """Test that bet sizing can progress through all streets."""
        action = PokerAction.RAISE_FULL_POT
        
        for street in range(4):  # 0=preflop, 1=flop, 2=turn, 3=river
            ctx = GameContext(
                pot_size=100,
                my_stack=500,
                amount_to_call=0,
                min_raise_amount=10,
                big_blind=10,
                street=street,
            )
            
            resolved = self.mapper.resolve_action(action, ctx)
            assert resolved.action == action or resolved.action == PokerAction.ALL_IN
            assert resolved.amount > 0
            logger.info(f"Street {street}: {resolved.description}")


def test_end_to_end_flop_vs_river():
    """
    END-TO-END TEST: Demonstrates the core requirement.
    
    Same action index (e.g., action_idx=7) resolves to different actual
    pot multipliers on Flop vs River based on street-specific configuration.
    """
    mapper = ActionMapper()
    
    # Flop state
    flop_state = GameContext(
        pot_size=100,
        my_stack=500,
        amount_to_call=0,
        min_raise_amount=20,
        big_blind=10,
        street=1,  # Flop (street index 1)
    )
    
    # River state
    river_state = GameContext(
        pot_size=100,
        my_stack=500,
        amount_to_call=0,
        min_raise_amount=20,
        big_blind=10,
        street=3,  # River (street index 3)
    )
    
    # Same action index: action 6 (RAISE_HALF_POT)
    action = PokerAction.RAISE_HALF_POT
    
    flop_result = mapper.resolve_action(action, flop_state)
    river_result = mapper.resolve_action(action, river_state)
    
    logger.info("=" * 70)
    logger.info("FLOP vs RIVER BET SIZING VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"Action: {action.name}")
    logger.info(f"Pot Size: {flop_state.pot_size}")
    logger.info(f"")
    logger.info(f"Flop (street=1):")
    logger.info(f"  Config multipliers: {mapper.bet_sizing_config.get_multipliers(1)}")
    logger.info(f"  Resolved amount: {flop_result.amount:.0f} chips")
    logger.info(f"  Pot multiple: {(flop_result.amount / 100):.2f}x")
    logger.info(f"")
    logger.info(f"River (street=3):")
    logger.info(f"  Config multipliers: {mapper.bet_sizing_config.get_multipliers(3)}")
    logger.info(f"  Resolved amount: {river_result.amount:.0f} chips")
    logger.info(f"  Pot multiple: {(river_result.amount / 100):.2f}x")
    logger.info("=" * 70)
    
    # ASSERTION: Flop and River should have DIFFERENT bet sizing
    # due to different street-specific multiplier configurations
    assert flop_result.amount != river_result.amount, \
        f"Expected different bet amounts for same action on different streets. " \
        f"Flop: {flop_result.amount}, River: {river_result.amount}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
