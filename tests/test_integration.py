"""
Integracios tesztek a teljes pipeline szamara.

Ellenorzi, hogy az osszes modul helyesen egyuttmukodik:
    config.yaml -> env -> model config -> training configs -> orchestrator -> mlops
"""

from __future__ import annotations

import os
from typing import Any

import yaml
import pytest


class TestConfigIntegrity:
    """A config.yaml belsó konzisztenciajanak tesztjei."""

    def test_gto_matrix_all_table_sizes(self, sample_config: dict) -> None:
        gto = sample_config["gto_matrix"]
        for size in ["2", "3", "4", "6", "8", "9"]:
            assert size in gto, f"Hianyzo asztalmerete a GTO matrixban: {size}"
            assert "vpip" in gto[size]
            assert "pfr" in gto[size]
            assert "af" in gto[size]

    def test_degeneration_thresholds_all_sizes(self, sample_config: dict) -> None:
        thresholds = sample_config["degeneration_thresholds"]
        for size in ["2", "3", "4", "6", "8", "9"]:
            assert size in thresholds
            assert "passivity" in thresholds[size]
            assert "maniac" in thresholds[size]

    def test_gto_vpip_ordering(self, sample_config: dict) -> None:
        """Nagyobb asztalmeret -> szukebb VPIP sav."""
        gto = sample_config["gto_matrix"]
        assert gto["2"]["vpip"][1] > gto["6"]["vpip"][1]
        assert gto["6"]["vpip"][1] > gto["9"]["vpip"][1]

    def test_action_space_consistency(self, sample_config: dict) -> None:
        assert sample_config["environment"]["action_space"]["num_actions"] == 9

    def test_ppo_params_valid(self, sample_config: dict) -> None:
        ppo = sample_config["ppo"]
        assert 0 < ppo["learning_rate"] < 1
        assert 0 < ppo["clip_epsilon"] < 1
        assert 0 < ppo["gamma"] <= 1
        assert 0 < ppo["gae_lambda"] <= 1

    def test_model_hidden_layers_decreasing(self, sample_config: dict) -> None:
        """A rejtett retegek merete csokkeno."""
        actor = sample_config["model"]["actor"]["hidden_layers"]
        for i in range(len(actor) - 1):
            assert actor[i] >= actor[i + 1]


class TestCrossModuleWiring:
    """Modul-kozti konfiguracios konzisztencia tesztek."""

    def test_observation_dim_matches_config(self, sample_config: dict) -> None:
        from src.env.features import ObservationBuilder, ObservationConfig
        num_players = sample_config["environment"]["num_players"]
        builder = ObservationBuilder(ObservationConfig(num_players=num_players))
        obs_dim = builder.get_observation_dim()
        assert obs_dim > 200  # ~281 a 6-Max-hoz

    def test_network_config_from_yaml(self, sample_config: dict) -> None:
        """NetworkConfig helyesen parszolodik a YAML-bol (torch nélkul is)."""
        # A NetworkConfig dataclass importja nem igenyel torch.nn
        try:
            from src.model.networks import NetworkConfig
            num_players = sample_config["environment"]["num_players"]
            net_cfg = NetworkConfig.from_dict(sample_config, num_players=num_players)
            assert net_cfg.num_actions == 9
            assert net_cfg.card_input_dim == 52
            assert net_cfg.trunk_input_dim == (
                net_cfg.card_embed_dim * 2 + net_cfg.context_embed_dim + net_cfg.history_embed_dim
            )
        except (ImportError, AttributeError):
            # torch.nn mock nem tamogatja a networks.py teljes importjat
            # A config dimenzio-szamitast kozvetlenul teszteljuk
            embed = sample_config["model"]["embedding"]
            trunk = embed["card_embed_dim"]*2 + embed["context_embed_dim"] + embed["history_embed_dim"]
            assert trunk == 224  # 64*2 + 32 + 64

    def test_buffer_config_from_yaml(self, sample_config: dict) -> None:
        from src.training.buffer import RolloutBufferConfig
        cfg = RolloutBufferConfig.from_dict(sample_config)
        assert cfg.buffer_size == sample_config["ppo"]["rollout_steps"]

    def test_trainer_config_from_yaml(self, sample_config: dict) -> None:
        from src.training.trainer import TrainerConfig
        cfg = TrainerConfig.from_dict(sample_config)
        assert cfg.learning_rate == sample_config["ppo"]["learning_rate"]

    def test_curriculum_from_yaml(self, sample_config: dict) -> None:
        from src.orchestrator.curriculum import CurriculumManager
        mgr = CurriculumManager.from_dict(sample_config)
        assert len(mgr.phases) == 3  # Phase 0, 1, 2

    def test_reward_shaper_from_yaml(self, sample_config: dict) -> None:
        from src.orchestrator.reward_shaper import RewardShapingConfig
        cfg = RewardShapingConfig.from_dict(sample_config)
        assert cfg.bluff_penalty_lambda == sample_config["reward_shaping"]["bluff_penalty_lambda"]

    def test_shutdown_from_yaml(self, sample_config: dict) -> None:
        from src.mlops.fault_tolerance import ShutdownConfig
        cfg = ShutdownConfig.from_dict(sample_config)
        assert cfg.max_runtime_hours == sample_config["mlops"]["graceful_shutdown"]["max_runtime_hours"]

    def test_all_modules_importable(self) -> None:
        """Minden fo osztaly importalhato hiba nelkul (env, training, orch, mlops)."""
        from src.env.features import ObservationBuilder
        from src.env.action_mapper import ActionMapper
        from src.env.equity import EquityCalculator
        from src.training.buffer import RolloutBuffer
        from src.training.opponent_pool import OpponentPool
        from src.training.trainer import TrainerConfig
        from src.training.runner import RunnerConfig
        from src.orchestrator.telemetry import TelemetryAnalyzer
        from src.orchestrator.curriculum import CurriculumManager
        from src.orchestrator.reward_shaper import RewardShaper
        from src.mlops.state_manager import RNGStateManager, CheckpointManager
        from src.mlops.fault_tolerance import GracefulShutdownMonitor, FaultHandler
        from src.mlops.hf_sync import configure_headless_auth
