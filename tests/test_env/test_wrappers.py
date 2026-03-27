"""
Wrapper smoke tests for environment integration (test_wrappers.py).

Phase 4-25: Basic functionality tests for RLCardWrapper and make_env().
Tests the poker environment wrapper against RLCard backend.
"""

from __future__ import annotations

import pytest
from src.env.wrappers import RLCardWrapper, WrapperConfig, make_env


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
        """Teszt: több lépés szekvenciájában."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        env.reset()
        
        step_count = 0
        for _ in range(10):
            legal_actions = env._latest_obs.get("legal_actions", [0])
            action = legal_actions[0]  # Fold
            obs, reward = env.step(action)
            step_count += 1
            
            if env.is_over():
                break
        
        # Legalább egy lépés történt
        assert step_count >= 1

    def test_observation_shape_consistency(self) -> None:
        """Teszt: megfigyelés dimenziók konzisztensek."""
        config = WrapperConfig(num_players=6, big_blind=2)
        env = RLCardWrapper(config=config)
        
        obs = env.reset()
        
        # Kártyák (52 bináris érték)
        assert len(obs.get("hand", [])) == 52
        assert len(obs.get("public_cards", [])) == 52
        
        # Pot és chipek (skalárok vagy listák)
        assert isinstance(obs.get("pot"), (int, float))
        assert isinstance(obs.get("my_chips"), (int, float))
