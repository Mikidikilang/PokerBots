"""
Egyseg tesztek a src/mlops/ modulhoz.

Tesztel: RNGStateManager, CheckpointManager, GracefulShutdownMonitor, FaultHandler
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pytest

from src.mlops.state_manager import RNGStateManager, CheckpointManager
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
# CheckpointManager Tesztek
# =============================================================================

class TestCheckpointManager:
    """A CheckpointManager mentes/betoltes/rotacio logikajanek tesztjei."""

    def _make_fake_network(self) -> object:
        class FakeNet:
            def state_dict(self): return {"layer": [1.0, 2.0, 3.0]}
            def load_state_dict(self, d): self._loaded = d
        return FakeNet()

    def _make_fake_optimizer(self) -> object:
        class FakeOpt:
            def state_dict(self): return {"lr": 0.001}
            def load_state_dict(self, d): self._loaded = d
        return FakeOpt()

    def test_no_checkpoint_initially(self, temp_dir: str) -> None:
        mgr = CheckpointManager(checkpoint_dir=temp_dir)
        assert not mgr.has_checkpoint()
        assert mgr.load_latest() is None

    def test_save_creates_file(self, temp_dir: str) -> None:
        mgr = CheckpointManager(checkpoint_dir=temp_dir)
        net = self._make_fake_network()
        path = mgr.save(net, iteration=100)
        assert os.path.exists(path)
        assert mgr.has_checkpoint()

    def test_save_load_roundtrip(self, temp_dir: str) -> None:
        mgr = CheckpointManager(checkpoint_dir=temp_dir)
        net = self._make_fake_network()
        opt = self._make_fake_optimizer()
        rng = RNGStateManager.capture_states()

        mgr.save(
            net, optimizer=opt, rng_states=rng,
            orchestrator_state={"phase": 1},
            training_meta={"steps": 5000},
            iteration=500,
        )

        loaded = mgr.load_latest()
        assert loaded is not None
        assert loaded["iteration"] == 500
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
        assert "rng_states" in loaded
        assert loaded["orchestrator_state"]["phase"] == 1

    def test_rotation(self, temp_dir: str) -> None:
        """Regi checkpoint-ok torlodnek ha meghaladja a max limitet."""
        mgr = CheckpointManager(checkpoint_dir=temp_dir, max_checkpoints=3)
        net = self._make_fake_network()
        for i in range(6):
            mgr.save(net, iteration=i * 100)
        assert mgr.get_checkpoint_count() <= 3

    def test_restore_full_state(self, temp_dir: str) -> None:
        mgr = CheckpointManager(checkpoint_dir=temp_dir)
        net = self._make_fake_network()
        opt = self._make_fake_optimizer()

        mgr.save(net, optimizer=opt, rng_states=RNGStateManager.capture_states(),
                 iteration=999)

        loaded = mgr.load_latest()
        net2 = self._make_fake_network()
        opt2 = self._make_fake_optimizer()
        result = mgr.restore_full_state(loaded, net2, opt2)
        assert result["iteration"] == 999


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
