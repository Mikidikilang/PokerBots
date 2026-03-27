"""
Wrapper smoke tests for environment integration (test_wrappers.py).

Phase 4-25: Basic functionality tests for RLCardWrapper and make_env().
Tests the poker environment wrapper against RLCard backend.
"""

from __future__ import annotations

import pytest
from src.env.wrappers import RLCardWrapper, WrapperConfig, make_env
from src.env.features import ObservationBuilder, ObservationConfig


class TestRLCardWrapper:
    """Smoke tests for the RLCardWrapper poker environment."""

    def test_wrapper_creates_successfully(self) -> None:
        """Teszt: RLCardWrapper sikeres létrehozása."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        assert env is not None
        assert isinstance(env, RLCardWrapper)

    def test_wrapper_reset(self) -> None:
        """Teszt: reset() értékét megfigyelés dict."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        obs = env.reset()
        
        assert isinstance(obs, dict)
        assert "hand" in obs
        assert "public_cards" in obs
        assert "pot" in obs
        assert "legal_actions" in obs

    def test_wrapper_step(self) -> None:
        """Teszt: step() valódi akciót jelent."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        env.reset()
        
        # Fold akció (index 0) mindig legális
        obs, reward = env.step(0)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))

    def test_wrapper_legal_actions(self) -> None:
        """Teszt: legal_actions nem üres."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        obs = env.reset()
        
        legal_actions = obs.get("legal_actions", [])
        assert len(legal_actions) > 0
        # Fold (index 0) mindig legális
        assert 0 in legal_actions

    def test_wrapper_is_over(self) -> None:
        """Teszt: is_over() logikai értéket ad vissza."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        env.reset()
        
        is_over = env.is_over()
        assert isinstance(is_over, bool)

    def test_make_env_factory(self) -> None:
        """Teszt: make_env() factory függvény."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = make_env(config)
        
        assert env is not None
        assert isinstance(env, RLCardWrapper)
        
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "hand" in obs

    def test_wrapper_multiple_steps(self) -> None:
        """Teszt: több lépés szekvenciájában (csak public API)."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        obs = env.reset()
        
        step_count = 0
        max_steps = 10
        
        # Iterate while hand is not over and step limit not reached
        while not env.is_over() and step_count < max_steps:
            # Use legal actions from the current observation
            legal_actions = obs.get("legal_actions", [0])
            action = legal_actions[0]  # Fold (always legal)
            
            obs, reward = env.step(action)
            step_count += 1
        
        # Legalább egy lépés történt
        assert step_count >= 1

    def test_observation_shape_consistency(self) -> None:
        """Teszt: megfigyelés dimenziók konzisztensek a ℝ^52 vetülethez."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        
        # Step 1: Get raw observation from wrapper
        raw_obs = env.reset()
        
        # Step 2: Verify raw observation structure (black-box API)
        assert isinstance(raw_obs, dict)
        assert "hand" in raw_obs
        assert "public_cards" in raw_obs
        assert "pot" in raw_obs
        assert "my_chips" in raw_obs
        assert "legal_actions" in raw_obs
        
        # Step 3: Verify raw card lists are proper format (SuitRank)
        hand = raw_obs.get("hand", [])
        assert isinstance(hand, list)
        assert len(hand) == 2, "hand must contain exactly 2 cards"
        
        # Verify each card is SuitRank format (Suit[0], Rank[1])
        suits = {"S", "H", "D", "C"}
        ranks = {"2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"}
        for card in hand:
            assert len(card) == 2, f"Card {card} must be 2 characters"
            assert card[0] in suits, f"Card {card} has invalid suit"
            assert card[1] in ranks, f"Card {card} has invalid rank"
        
        # Step 4: Encode the raw observation using ObservationBuilder
        builder = ObservationBuilder(ObservationConfig(num_players=6))
        encoded_obs = builder.build(raw_obs)
        
        # Step 5: Verify encoded tensor dimensions match ℝ^52 requirements
        assert "hole_cards" in encoded_obs
        assert encoded_obs["hole_cards"].shape == (52,), \
            f"hole_cards must be 52-dimensional, got {encoded_obs['hole_cards'].shape}"
        
        assert "community_cards" in encoded_obs
        assert encoded_obs["community_cards"].shape == (52,), \
            f"community_cards must be 52-dimensional, got {encoded_obs['community_cards'].shape}"
        
        # Step 6: Verify multi-hot constraints (each dimension should be 0.0 or 1.0)
        hole_cards_sum = float(encoded_obs["hole_cards"].sum().item())
        assert hole_cards_sum == 2.0, \
            f"hole_cards multi-hot sum should be 2.0 (2 cards), got {hole_cards_sum}"
        
        community_cards_sum = float(encoded_obs["community_cards"].sum().item())
        assert 0.0 <= community_cards_sum <= 5.0, \
            f"community_cards multi-hot sum should be 0-5, got {community_cards_sum}"
