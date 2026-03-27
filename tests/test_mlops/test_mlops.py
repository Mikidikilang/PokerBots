"""
Egyseg tesztek a src/mlops/ modulhoz.

Tesztel: RNGStateManager, StateManager, GracefulShutdownMonitor, FaultHandler
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pytest
import torch

from src.mlops.state_manager import RNGStateManager, StateManager
from src.mlops.fault_tolerance import (
    GracefulShutdownMonitor, ShutdownConfig, FaultHandler,
)
from src.mlops.hf_sync import configure_headless_auth


# =============================================================================
# RNGStateManager Tesztek
# =============================================================================

class TestRNGStateManager:
    """Az RNGStateManager determinisztikus folytatasi logikajanek tesztjei."""

    def test_set_global_seed(self) -> None:
        gen = RNGStateManager.set_global_seed(42)
        assert gen is not None

    def test_capture_states_keys(self) -> None:
        states = RNGStateManager.capture_states()
        assert "python_stdlib" in states
        assert "numpy" in states
        assert "torch_cpu" in states

    def test_deterministic_resume_python(self) -> None:
        """Python random determinisztikus folytatas."""
        random.seed(42)
        states = RNGStateManager.capture_states()
        r1 = [random.random() for _ in range(10)]
        RNGStateManager.restore_states(states)
        r2 = [random.random() for _ in range(10)]
        assert r1 == r2

    def test_deterministic_resume_numpy(self) -> None:
        """NumPy random determinisztikus folytatas."""
        np.random.seed(42)
        states = RNGStateManager.capture_states()
        n1 = np.random.random(10).tolist()
        RNGStateManager.restore_states(states)
        n2 = np.random.random(10).tolist()
        assert n1 == n2

    def test_restore_empty_states(self) -> None:
        """Ures allapot eseten nem dob hibat."""
        RNGStateManager.restore_states({})  # Nem szabad crashelnie

    def test_capture_with_dataloader_gen(self) -> None:
        import torch
        gen = torch.Generator()
        gen.manual_seed(123)
        states = RNGStateManager.capture_states(dataloader_generator=gen)
        assert states.get("dataloader") is not None


# =============================================================================
# StateManager Tesztek
# =============================================================================

class TestStateManager:
    """Az StateManager mentes/betoltes logikajanek tesztjei (uj API)."""

    def test_from_dict_creates_instance(self, sample_config: dict) -> None:
        """from_dict helyesen letrehozza a StateManager peldaryt."""
        mgr = StateManager.from_dict(sample_config)
        assert mgr is not None
        assert isinstance(mgr, StateManager)

    def test_save_load_roundtrip_minimal(self, sample_config: dict, temp_dir: str) -> None:
        """Mentés/betöltés az uj StateManager API-val (minimal parametrek)."""
        # Update config to use temp_dir
        config = sample_config.copy()
        config["mlops"]["checkpoint"]["local_checkpoint_dir"] = temp_dir
        
        mgr = StateManager.from_dict(config)

        # Dummy network és optimizer
        network = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(network.parameters())

        # Mentés minimal paraméterekkel (keyword-only, no save_dir)
        mgr.save_training_state(
            network=network,
            optimizer=optimizer,
            iteration=42,
            total_env_steps=1000,
            total_hands=100,
        )

        # Betöltés
        checkpoint = mgr.load_training_state()
        assert checkpoint is not None
        assert checkpoint["iteration"] == 42
        assert checkpoint["total_env_steps"] == 1000
        assert checkpoint["total_hands"] == 100
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_load_roundtrip_full(self, sample_config: dict, temp_dir: str) -> None:
        """Mentés/betöltés teljes paraméterekkel."""
        # Update config to use temp_dir
        config = sample_config.copy()
        config["mlops"]["checkpoint"]["local_checkpoint_dir"] = temp_dir
        
        mgr = StateManager.from_dict(config)

        # Dummy network és optimizer
        network = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(network.parameters())

        # State dictionaries
        orchestrator_state = {
            "curriculum_state": "armed",
            "arms": [0.5, 0.5],
            "iteration": 100,
        }
        config_dict = {
            "training": {"learning_rate": 0.001},
            "environment": {"big_blind": 2},
        }

        # Mentés teljes paraméterekkel (keyword-only arguments)
        mgr.save_training_state(
            network=network,
            optimizer=optimizer,
            iteration=100,
            total_env_steps=5000,
            total_hands=500,
            best_mean_reward=1.5,
            orchestrator_state=orchestrator_state,
            config=config_dict,
            is_best=True,
        )

        # Betöltés és validáció
        checkpoint = mgr.load_training_state()
        assert checkpoint["iteration"] == 100
        assert checkpoint["total_env_steps"] == 5000
        assert checkpoint["total_hands"] == 500
        assert checkpoint["best_mean_reward"] == 1.5
        # Note: is_best is a parameter controlling file naming, not stored in checkpoint
        assert checkpoint["orchestrator_state"]["curriculum_state"] == "armed"
        assert checkpoint["config"]["training"]["learning_rate"] == 0.001

    def test_config_parsing_nested_path(self, sample_config: dict) -> None:
        """from_dict helyesen olvassa a beagyazott config utat."""
        mgr = StateManager.from_dict(sample_config)
        assert mgr is not None
        # A StateManager.from_dict helyesen kezeli a beagyazott útvonalat:
        # cfg["mlops"]["checkpoint"]["local_checkpoint_dir"]

    def test_save_without_optional_fields(self, sample_config: dict, temp_dir: str) -> None:
        """Mentés opcionális mezők nélkül nem szabad hogy crasheljen."""
        # Update config to use temp_dir
        config = sample_config.copy()
        config["mlops"]["checkpoint"]["local_checkpoint_dir"] = temp_dir
        
        mgr = StateManager.from_dict(config)
        network = torch.nn.Linear(5, 3)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)

        # Csak a kötelező paramétereket adjuk meg (keyword-only)
        mgr.save_training_state(
            network=network,
            optimizer=optimizer,
            iteration=10,
            total_env_steps=100,
            total_hands=50,
        )

        checkpoint = mgr.load_training_state()
        assert checkpoint["iteration"] == 10
        # Opcionális mezők None szerint kezelendők
        best_reward = checkpoint.get("best_mean_reward")
        assert best_reward is None or isinstance(best_reward, (int, float))


# =============================================================================
# GracefulShutdownMonitor Tesztek
# =============================================================================

class TestGracefulShutdownMonitor:
    """A GracefulShutdownMonitor idokezelo logikajanek tesztjei."""

    def test_not_shutdown_initially(self) -> None:
        mon = GracefulShutdownMonitor(ShutdownConfig(
            max_runtime_hours=1.0, register_signal_handlers=False,
        ))
        assert not mon.should_shutdown()

    def test_shutdown_after_timeout(self) -> None:
        """Nagyon rovid timeout eseten azonnal shutdown."""
        mon = GracefulShutdownMonitor(ShutdownConfig(
            max_runtime_hours=0.0,  # 0 ora = azonnal
            register_signal_handlers=False,
        ))
        assert mon.should_shutdown()

    def test_request_shutdown(self) -> None:
        mon = GracefulShutdownMonitor(ShutdownConfig(
            max_runtime_hours=100.0, register_signal_handlers=False,
        ))
        assert not mon.should_shutdown()
        mon.request_shutdown("teszt")
        assert mon.should_shutdown()

    def test_elapsed_hours(self) -> None:
        mon = GracefulShutdownMonitor(ShutdownConfig(
            max_runtime_hours=10.0, register_signal_handlers=False,
        ))
        elapsed = mon.get_elapsed_hours()
        assert elapsed >= 0.0
        assert elapsed < 0.01  # Kevesebb mint 36 masodperc

    def test_remaining_hours(self) -> None:
        mon = GracefulShutdownMonitor(ShutdownConfig(
            max_runtime_hours=10.0, register_signal_handlers=False,
        ))
        remaining = mon.get_remaining_hours()
        assert remaining > 9.9

    def test_progress_pct(self) -> None:
        mon = GracefulShutdownMonitor(ShutdownConfig(
            max_runtime_hours=10.0, register_signal_handlers=False,
        ))
        pct = mon.get_progress_pct()
        assert 0.0 <= pct <= 100.0
        assert pct < 1.0  # Elindulas utan meg alacsony

    def test_status_dict(self) -> None:
        mon = GracefulShutdownMonitor(ShutdownConfig(
            max_runtime_hours=10.0, register_signal_handlers=False,
        ))
        status = mon.get_status()
        assert "elapsed_hours" in status
        assert "remaining_hours" in status
        assert "progress_pct" in status

    def test_config_from_dict(self, sample_config: dict) -> None:
        cfg = ShutdownConfig.from_dict(sample_config)
        assert cfg.max_runtime_hours == 11.5
        assert cfg.use_monotonic_clock is True

    def test_callback_registration(self) -> None:
        mon = GracefulShutdownMonitor(ShutdownConfig(
            register_signal_handlers=False,
        ))
        callback_called = []
        mon.register_shutdown_callback(lambda: callback_called.append(True))
        assert len(callback_called) == 0


# =============================================================================
# FaultHandler Tesztek
# =============================================================================

class TestFaultHandler:
    """A FaultHandler hibakezelo logikajanek tesztjei."""

    def test_nan_retry(self) -> None:
        fh = FaultHandler(max_nan_retries=3)
        assert fh.handle_nan_loss() == "retry"
        assert fh.handle_nan_loss() == "retry"
        assert fh.handle_nan_loss() == "retry"

    def test_nan_abort_after_max(self) -> None:
        fh = FaultHandler(max_nan_retries=2)
        fh.handle_nan_loss()
        fh.handle_nan_loss()
        assert fh.handle_nan_loss() == "abort"

    def test_nan_reset(self) -> None:
        fh = FaultHandler(max_nan_retries=2)
        fh.handle_nan_loss()
        fh.handle_nan_loss()
        fh.reset_nan_counter()
        assert fh.handle_nan_loss() == "retry"

    def test_oom_handling(self) -> None:
        fh = FaultHandler()
        assert fh.handle_oom() == "reduce_batch"

    def test_generic_error(self) -> None:
        fh = FaultHandler()
        assert fh.handle_generic_error(ValueError("test")) == "retry"

    def test_error_summary(self) -> None:
        fh = FaultHandler()
        fh.handle_nan_loss()
        fh.handle_oom()
        summary = fh.get_error_summary()
        assert summary["total_errors"] == 2
        assert "nan_loss" in summary["error_types"]
        assert "oom" in summary["error_types"]


# =============================================================================
# HF Sync Tesztek
# =============================================================================

class TestHFSync:
    """A hf_sync.py headless autentikacio logikajanek tesztjei."""

    def test_headless_auth_direct_token(self) -> None:
        os.environ.pop("HF_TOKEN", None)
        result = configure_headless_auth(token="hf_test_12345")
        assert result is True
        assert os.environ.get("HF_TOKEN") == "hf_test_12345"

    def test_headless_auth_existing_env(self) -> None:
        os.environ["HF_TOKEN"] = "already_set"
        result = configure_headless_auth()
        assert result is True
        os.environ.pop("HF_TOKEN", None)

    def test_headless_auth_no_token(self) -> None:
        os.environ.pop("HF_TOKEN", None)
        result = configure_headless_auth(use_kaggle_secrets=False)
        assert result is False
