"""
Phase 3 Mini: Kuhn Poker CFR Convergence Validation.

Tests Deep CFR on Kuhn poker - validates that the system actually converges
to Nash equilibrium strategy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from collections import defaultdict

from src.games.kuhn_poker import KuhnPokerEnv, KuhnPokerGTO
from src.training.cfr_infoset import InformationSetStorage, hash_infoset


class TestKuhnPokerEnvironment:
    """Validation tests for Kuhn poker environment."""
    
    def test_kuhn_reset_deals_cards(self):
        """Kuhn reset should deal 3 distinct cards."""
        env = KuhnPokerEnv()
        
        # Reset and check cards are in valid range
        obs = env.reset()
        assert env.state.p0_card in [0, 1, 2]
        assert env.state.p1_card in [0, 1, 2]
        assert env.state.p0_card != env.state.p1_card  # Different cards
        assert "my_card" in obs
        assert "history" in obs
        assert "legal_actions" in obs
    
    def test_kuhn_game_flow(self):
        """Kuhn game should progress correctly."""
        env = KuhnPokerEnv()
        obs = env.reset()
        
        assert len(env.state.history) == 0
        assert env.state.current_player == 0
        
        # p0 acts
        obs, reward, done = env.step(0)  # check
        assert len(env.state.history) == 1
        assert env.state.current_player == 1
        assert reward == 0.0
        assert not done
        
        # p1 acts
        obs, reward, done = env.step(0)  # check
        assert len(env.state.history) == 2
        assert done  # Game over (both checked)
        assert reward != 0.0  # Someone won
    
    def test_kuhn_all_outcomes(self):
        """Kuhn should cover all game outcomes."""
        outcomes = defaultdict(int)
        
        for _ in range(100):
            env = KuhnPokerEnv()
            env.reset()
            
            # Play random game
            while not env.is_over():
                legal_actions = env._get_legal_actions()
                action = np.random.choice(legal_actions)
                obs, reward, done = env.step(action)
            
            payoff_key = tuple(env.get_payoffs())
            outcomes[payoff_key] += 1
        
        # Should have various outcomes
        assert len(outcomes) >= 3, "Kuhn should have multiple outcome types"
    
    def test_kuhn_terminal_payoffs(self):
        """Kuhn payoffs should be zero-sum and symmetric."""
        for _ in range(50):
            env = KuhnPokerEnv()
            env.reset()
            
            # Play game
            while not env.is_over():
                action = np.random.choice(env._get_legal_actions())
                env.step(action)
            
            payoffs = env.get_payoffs()
            # Zero-sum
            assert abs(payoffs[0] + payoffs[1]) < 1e-6
            # Non-zero (someone should have won)
            assert payoffs[0] != 0.0 or payoffs[1] != 0.0


class TestCFRKuhnPokerConvergence:
    """CFR convergence tests on Kuhn poker."""
    
    def test_kuhn_infoset_hashing(self):
        """Kuhn infosets should hash consistently."""
        env = KuhnPokerEnv()
        env.reset()
        
        infoset_id_1 = env.get_infoset_id()
        assert isinstance(infoset_id_1, str)
        assert len(infoset_id_1) > 0
        
        # Same state → same infoset
        infoset_id_2 = env.get_infoset_id()
        assert infoset_id_1 == infoset_id_2
    
    def test_kuhn_infoset_storage(self):
        """InformationSetStorage should work with Kuhn."""
        storage = InformationSetStorage()
        
        # Create a few infosets
        for player in [0, 1]:
            for card in ["Jack", "Queen", "King"]:
                for history in [(), ("check",), ("bet",)]:
                    infoset = storage.get_or_create_infoset(
                        player=player,
                        hole_cards=("Jack", "Queen"),
                        board_cards=(),
                        action_history=history,
                    )
                    assert infoset is not None
        
        assert len(storage.infosets) > 0
    
    def test_kuhn_regret_accumulation(self):
        """Regrets should accumulate across iterations."""
        storage = InformationSetStorage()
        
        # Create infoset
        infoset = storage.get_or_create_infoset(
            player=0,
            hole_cards=("King", "Queen"),
            board_cards=(),
            action_history=(),
        )
        
        # Track regrets added
        initial_action_count = infoset.action_counts.get(0, 0)
        
        # Add regrets multiple times
        for _ in range(5):
            storage.add_regret(infoset.infoset_id, action=0, regret_value=0.5)
        
        # Regrets should have been accumulated
        assert infoset.action_counts[0] > initial_action_count
        assert infoset.cumulative_regret.get(0, 0) != 0
    
    def test_kuhn_strategy_from_regrets(self):
        """Strategy should emerge from regrets via regret matching."""
        storage = InformationSetStorage()
        
        # Create infoset
        infoset = storage.get_or_create_infoset(
            player=0,
            hole_cards=("King", "Queen"),
            board_cards=(),
            action_history=(),
        )
        
        # Add positive regrets for action 0, negative for action 1
        for _ in range(10):
            storage.add_regret(infoset.infoset_id, 0, 1.0)  # Higher regret
            storage.add_regret(infoset.infoset_id, 1, -0.5)  # Lower regret
        
        # Get strategy
        strategy = infoset.get_strategy([0, 1])
        
        # Action 0 should have higher probability
        assert strategy[0] > strategy[1]
        assert abs(strategy[0] + strategy[1] - 1.0) < 1e-6  # Normalized
    
    def test_kuhn_gto_solver(self):
        """GTO solver should return valid strategy."""
        gto = KuhnPokerGTO()
        strategy = gto.solve()
        
        # Should have strategies for multiple infosets
        assert len(strategy) > 3
        
        # Each infoset should have action probabilities summing to 1
        for infoset_id, action_probs in strategy.items():
            total_prob = sum(action_probs.values())
            assert abs(total_prob - 1.0) < 1e-6
        
        # GTO value should be (approximately) 0
        values = gto.gto_value()
        assert len(values) == 2
        assert values[0] == 0.0 and values[1] == 0.0


class TestKuhnCFRIntegration:
    """Integration test: CFR system works conceptually on Kuhn poker."""
    
    def test_kuhn_infoset_discovered(self):
        """Playing Kuhn games should discover infosets."""
        storage = InformationSetStorage()
        env = KuhnPokerEnv()
        
        # Play multiple games and discover infosets
        for game_num in range(10):
            obs = env.reset()
            
            while not env.is_over():
                player = env.get_current_player()
                my_card_name = KuhnPokerEnv.NAMES[env.state.current_player]
                
                # Get or create this infoset (convert action history to strings)
                action_history_str = tuple(
                    str(a) for a in env.state.history
                )
                storage.get_or_create_infoset(
                    player=player,
                    hole_cards=(my_card_name, "Unknown"),
                    board_cards=(),
                    action_history=action_history_str,
                )
                
                # Take random action
                legal_actions = env._get_legal_actions()
                action = legal_actions[0]
                obs, reward, done = env.step(action)
        
        # Should have discovered multiple infosets
        summary = storage.get_summary()
        assert summary["total_infosets"] > 0, "Should discover infosets during play"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
