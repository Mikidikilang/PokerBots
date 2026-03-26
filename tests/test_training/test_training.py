"""
Egyseg tesztek a src/training/ modulhoz.

Tesztel: RolloutBuffer, OpponentPool, PPOTrainer config, RunnerConfig
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.buffer import RolloutBuffer, RolloutBufferConfig
from src.training.opponent_pool import (
    OpponentPool, create_archetype,
    CallingStationBot, ManiacBot, RandomBot, TightPassiveBot,
)
from src.training.trainer import TrainerConfig
from src.training.runner import RunnerConfig


# =============================================================================
# RolloutBuffer Tesztek
# =============================================================================

class TestRolloutBuffer:
    """A RolloutBuffer GAE es mintavetelezes logikajanek tesztjei."""

    def _make_buffer(self, size: int = 100) -> RolloutBuffer:
        return RolloutBuffer(RolloutBufferConfig(
            buffer_size=size, gamma=0.99, gae_lambda=0.95, num_mini_batches=2,
        ))

    def _fill_buffer(self, buf: RolloutBuffer, n: int) -> None:
        import torch
        for i in range(n):
            obs = {"hole_cards": torch.tensor(np.random.rand(52).astype(np.float32))}
            buf.add(
                observation=obs,
                action=torch.tensor(np.array([3])),
                reward=float(np.random.randn()),
                log_prob=torch.tensor(np.array([-1.5])),
                value=torch.tensor(np.array([0.5])),
                done=(i % 20 == 19),
            )

    def test_empty_buffer(self) -> None:
        buf = self._make_buffer()
        assert len(buf) == 0
        assert not buf.full

    def test_add_increments_size(self) -> None:
        buf = self._make_buffer(100)
        self._fill_buffer(buf, 50)
        assert len(buf) == 50
        assert not buf.full

    def test_buffer_full_flag(self) -> None:
        buf = self._make_buffer(100)
        self._fill_buffer(buf, 100)
        assert buf.full

    def test_gae_computation(self) -> None:
        buf = self._make_buffer(50)
        self._fill_buffer(buf, 50)
        buf.compute_gae(last_value=0.0)
        stats = buf.get_stats()
        assert "advantage_mean" in stats
        assert "returns_mean" in stats

    def test_gae_normalized_advantages(self) -> None:
        """GAE utan az advantage-ok normalizaltak (mean~0, std~1)."""
        buf = self._make_buffer(200)
        self._fill_buffer(buf, 200)
        buf.compute_gae(last_value=0.0)
        stats = buf.get_stats()
        assert abs(stats["advantage_mean"]) < 0.1  # ~0

    def test_mini_batches_generated(self) -> None:
        buf = self._make_buffer(100)
        self._fill_buffer(buf, 100)
        buf.compute_gae(last_value=0.0)
        batches = list(buf.get_mini_batches())
        assert len(batches) >= 2

    def test_mini_batch_keys(self) -> None:
        buf = self._make_buffer(100)
        self._fill_buffer(buf, 100)
        buf.compute_gae(last_value=0.0)
        for batch in buf.get_mini_batches():
            assert "observations" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "advantages" in batch
            assert "returns" in batch

    def test_get_mini_batches_without_gae_raises(self) -> None:
        buf = self._make_buffer(50)
        self._fill_buffer(buf, 50)
        with pytest.raises(RuntimeError, match="compute_gae"):
            list(buf.get_mini_batches())

    def test_reset_clears_buffer(self) -> None:
        buf = self._make_buffer(50)
        self._fill_buffer(buf, 50)
        buf.reset()
        assert len(buf) == 0
        assert not buf.full

    def test_stats_empty(self) -> None:
        buf = self._make_buffer()
        stats = buf.get_stats()
        assert stats["buffer_size"] == 0.0


# =============================================================================
# OpponentPool Tesztek
# =============================================================================

class TestOpponentPool:
    """Az OpponentPool archetipus es snapshot kezelesenek tesztjei."""

    def test_default_archetypes(self) -> None:
        pool = OpponentPool()
        names = pool.get_all_archetype_names()
        assert "calling_station" in names
        assert "maniac" in names
        assert "random" in names
        assert "tight_passive" in names

    def test_calling_station_always_calls(self) -> None:
        bot = CallingStationBot()
        assert bot.select_action([0, 1, 3, 8], {}) == 1  # Check/Call

    def test_calling_station_folds_if_no_call(self) -> None:
        bot = CallingStationBot()
        assert bot.select_action([0, 3, 8], {}) == 0  # Fold (nincs Call)

    def test_maniac_max_aggression(self) -> None:
        bot = ManiacBot()
        assert bot.select_action([0, 1, 3, 4, 8], {}) == 8  # All-in

    def test_maniac_raise_if_no_allin(self) -> None:
        bot = ManiacBot()
        assert bot.select_action([0, 1, 3, 4], {}) == 4  # Legmagasabb raise

    def test_random_bot_returns_legal(self) -> None:
        bot = RandomBot()
        legal = [0, 1, 5]
        for _ in range(100):
            action = bot.select_action(legal, {})
            assert action in legal

    def test_tight_passive_mostly_folds(self) -> None:
        """A TightPassive bot a legtobb esetben Fold-ot valaszt."""
        bot = TightPassiveBot(play_frequency=0.1)
        fold_count = sum(1 for _ in range(1000) if bot.select_action([0, 1], {}) == 0)
        assert fold_count > 800  # ~90% fold

    def test_create_archetype_factory(self) -> None:
        for name in ["calling_station", "maniac", "random", "tight_passive"]:
            agent = create_archetype(name)
            assert agent.name == name

    def test_create_archetype_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Ismeretlen"):
            create_archetype("unknown_bot")

    def test_snapshot_add_and_list(self) -> None:
        pool = OpponentPool(max_pool_size=5)
        pool.add_snapshot({"weights": 1}, iteration=100)
        pool.add_snapshot({"weights": 2}, iteration=200)
        ids = pool.get_snapshot_ids()
        assert len(ids) == 2
        assert "snapshot_iter_000100" in ids

    def test_snapshot_fifo_rotation(self) -> None:
        pool = OpponentPool(max_pool_size=3)
        for i in range(5):
            pool.add_snapshot({"w": i}, iteration=i * 100)
        assert len(pool.snapshots) <= 3

    def test_pool_size_includes_both(self) -> None:
        pool = OpponentPool(archetype_names=["calling_station", "maniac"])
        pool.add_snapshot({"w": 1}, iteration=100)
        assert pool.get_pool_size() == 3  # 2 arch + 1 snap

    def test_select_random_opponent(self) -> None:
        pool = OpponentPool()
        for _ in range(20):
            name = pool.select_random_opponent()
            assert name in pool.get_all_opponent_names()

    def test_pool_stats(self) -> None:
        pool = OpponentPool()
        stats = pool.get_pool_stats()
        assert stats["num_archetypes"] == 4
        assert stats["total_pool_size"] == 4


# =============================================================================
# Config from_dict Tesztek
# =============================================================================

class TestConfigFromDict:
    """A YAML config -> dataclass konverzio tesztjei."""

    def test_buffer_config_from_dict(self, sample_config: dict) -> None:
        cfg = RolloutBufferConfig.from_dict(sample_config)
        assert cfg.buffer_size == 2048
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95

    def test_trainer_config_from_dict(self, sample_config: dict) -> None:
        cfg = TrainerConfig.from_dict(sample_config)
        assert cfg.learning_rate == 3e-4
        assert cfg.clip_epsilon == 0.2
        assert cfg.num_epochs == 4
        assert cfg.entropy_coef == 0.01

    def test_runner_config_from_dict(self, sample_config: dict) -> None:
        cfg = RunnerConfig.from_dict(sample_config)
        assert cfg.max_runtime_hours == 11.5
        assert cfg.save_interval == 100
        assert cfg.eval_interval == 50
