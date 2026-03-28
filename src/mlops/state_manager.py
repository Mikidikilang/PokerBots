"""
MLOps State Manager (src/mlops/state_manager.py).

[FIX M-1 — 2025-03-28] weights_only=True fallback for older PyTorch.

    Kaggle T4 images may run PyTorch 2.1–2.5. Full optimizer + scheduler
    state dicts contain Python scalars and lists that require weights_only=False
    on PyTorch < 2.6. We try weights_only=True first (secure), then fall back
    to weights_only=False with a logged warning (functional but less secure).
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["RNGStateManager", "CheckpointManager", "StateManager"]

_PHASE0_FIX_VERSION: str = "v0.3.1"


# =============================================================================
# RNGStateManager
# =============================================================================

class RNGStateManager:
    """Captures and restores the state of all random number generators."""

    @staticmethod
    def set_global_seed(seed: int) -> torch.Generator:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        logger.debug("Global seed set: %d", seed)
        return generator

    @staticmethod
    def capture_states(dataloader_generator: torch.Generator | None = None) -> dict[str, Any]:
        states = {
            "python_stdlib": random.getstate(),
            "numpy":         np.random.get_state(),
            "torch_cpu":     torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
        if dataloader_generator is not None:
            states["dataloader"] = dataloader_generator.get_state()
        logger.debug("Captured RNG states: keys=%s", list(states.keys()))
        return states

    @staticmethod
    def restore_states(states: dict[str, Any]) -> None:
        if "python_stdlib" in states:
            random.setstate(states["python_stdlib"])
        if "numpy" in states:
            np.random.set_state(states["numpy"])
        if "torch_cpu" in states:
            torch.set_rng_state(states["torch_cpu"])
        if "torch_cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])
        logger.debug("Restored RNG states from checkpoint")


# =============================================================================
# CheckpointManager
# =============================================================================

class CheckpointManager:
    """Atomic checkpoint save/load with FIFO rotation."""

    CHECKPOINT_GLOB: str = "checkpoint_iter_*.pt"
    BEST_FILENAME:   str = "checkpoint_best.pt"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
    ) -> None:
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.max_to_keep:    int  = max(1, int(max_to_keep))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "CheckpointManager init: dir='%s', max_to_keep=%d",
            self.checkpoint_dir, self.max_to_keep,
        )

    # -------------------------------------------------------------------------
    # Save (atomic)
    # -------------------------------------------------------------------------

    def save(
        self,
        checkpoint: dict[str, Any],
        iteration: int,
        *,
        is_best: bool = False,
    ) -> Path:
        checkpoint.setdefault("phase0_fixes_version", _PHASE0_FIX_VERSION)
        checkpoint.setdefault("saved_at_unix", time.time())

        target_path = self._iteration_path(iteration)
        self._atomic_save(checkpoint, target_path)

        if is_best:
            best_path = self.checkpoint_dir / self.BEST_FILENAME
            self._atomic_save(checkpoint, best_path)
            logger.info("Best checkpoint updated: '%s' (iter=%d)", best_path, iteration)

        self._rotate()

        size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Checkpoint saved: '%s' (%.2f MB, iter=%d, is_best=%s)",
            target_path, size_mb, iteration, is_best,
        )
        return target_path

    # -------------------------------------------------------------------------
    # Load — [FIX M-1] weights_only fallback
    # -------------------------------------------------------------------------

    def load(
        self,
        filepath: str | Path,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any]:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: '{filepath}'")

        logger.info(
            "Loading checkpoint from '%s' (map_location=%s)", filepath, map_location
        )

        # [FIX M-1] Try weights_only=True first (secure, requires PyTorch >= 2.6
        # for full optimizer state support). Fall back to weights_only=False for
        # older PyTorch versions common on Kaggle T4 images.
        checkpoint: dict[str, Any] | None = None
        try:
            checkpoint = torch.load(
                str(filepath),
                map_location=map_location,
                weights_only=True,
            )
        except (TypeError, RuntimeError, pickle_error()) as secure_exc:
            logger.warning(
                "weights_only=True failed for '%s': %s. "
                "Retrying with weights_only=False (PyTorch < 2.6 detected). "
                "Upgrade to PyTorch >= 2.6 for fully secure checkpoint loading.",
                filepath, secure_exc,
            )
            try:
                checkpoint = torch.load(
                    str(filepath),
                    map_location=map_location,
                    weights_only=False,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Failed to load checkpoint from '{filepath}' with both "
                    f"weights_only=True and weights_only=False: {fallback_exc}"
                ) from fallback_exc

        if checkpoint is None:
            raise RuntimeError(f"Checkpoint load returned None for '{filepath}'")

        version = checkpoint.get("phase0_fixes_version", "pre-v0.3.0")
        logger.info(
            "Checkpoint loaded: iter=%s, version=%s",
            checkpoint.get("iteration", "?"), version,
        )
        return checkpoint

    def load_latest(
        self,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any] | None:
        latest = self.get_latest_path()
        if latest is None:
            logger.info("No checkpoints in '%s'. Starting fresh.", self.checkpoint_dir)
            return None
        return self.load(latest, map_location=map_location)

    def load_best(
        self,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any] | None:
        best_path = self.checkpoint_dir / self.BEST_FILENAME
        if not best_path.exists():
            return None
        return self.load(best_path, map_location=map_location)

    # -------------------------------------------------------------------------
    # Discovery helpers
    # -------------------------------------------------------------------------

    def get_latest_path(self) -> Path | None:
        paths = self._list_iteration_checkpoints()
        if not paths:
            return None
        return max(paths, key=self._parse_iteration)

    def list_checkpoints(self) -> list[Path]:
        return sorted(self._list_iteration_checkpoints(), key=self._parse_iteration)

    def has_checkpoint(self) -> bool:
        return len(self._list_iteration_checkpoints()) > 0

    # -------------------------------------------------------------------------
    # Atomic write
    # -------------------------------------------------------------------------

    def _atomic_save(self, obj: Any, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path = target_path.with_suffix(".pt.tmp")
        try:
            torch.save(obj, str(tmp_path))
            try:
                with open(str(tmp_path), "rb") as fh:
                    os.fsync(fh.fileno())
            except OSError as fsync_err:
                logger.warning(
                    "os.fsync() failed for '%s' (%s). Proceeding with os.replace().",
                    tmp_path, fsync_err,
                )
            os.replace(str(tmp_path), str(target_path))
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError as cleanup_err:
                logger.warning("Could not remove tmp file '%s': %s", tmp_path, cleanup_err)
            raise RuntimeError(
                f"Atomic checkpoint save failed for '{target_path}': {exc}"
            ) from exc

    def _rotate(self) -> None:
        paths = self.list_checkpoints()
        excess = len(paths) - self.max_to_keep
        if excess <= 0:
            return
        for old_path in paths[:excess]:
            try:
                old_path.unlink()
                logger.debug("Rotated old checkpoint: '%s'", old_path)
            except OSError as exc:
                logger.warning("Could not delete '%s': %s", old_path, exc)

    def _iteration_path(self, iteration: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_iter_{iteration:08d}.pt"

    def _list_iteration_checkpoints(self) -> list[Path]:
        return list(self.checkpoint_dir.glob(self.CHECKPOINT_GLOB))

    @staticmethod
    def _parse_iteration(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0


def pickle_error() -> type:
    """Return the pickle UnpicklingError class (for weights_only exception catch)."""
    import pickle
    return pickle.UnpicklingError


# =============================================================================
# StateManager
# =============================================================================

class StateManager:
    """High-level facade for runner.py — bundles full training state."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
    ) -> None:
        self.ckpt_mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_to_keep=max_to_keep,
        )
        logger.info(
            "StateManager ready (dir='%s', max_to_keep=%d)",
            checkpoint_dir, max_to_keep,
        )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> StateManager:
        mlops_cfg    = cfg.get("mlops", {})
        checkpoint_cfg = mlops_cfg.get("checkpoint", {})
        return cls(
            checkpoint_dir=checkpoint_cfg.get("local_checkpoint_dir", "checkpoints"),
            max_to_keep=int(checkpoint_cfg.get("max_checkpoints_to_keep", 5)),
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def save_training_state(
        self,
        *,
        network:            torch.nn.Module,
        optimizer:          torch.optim.Optimizer,
        iteration:          int,
        total_env_steps:    int          = 0,
        total_hands:        int          = 0,
        best_mean_reward:   float        = float("-inf"),
        scheduler:          Any | None   = None,
        orchestrator_state: dict | None  = None,
        config:             dict | None  = None,
        rng_states:         dict[str, Any] | None = None,
        wandb_run_id:       str | None   = None,
        is_best:            bool         = False,
    ) -> Path:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            model_state = network.module.state_dict()
        else:
            model_state = network.state_dict()

        checkpoint: dict[str, Any] = {
            "model_state_dict":     model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "iteration":            iteration,
            "total_env_steps":      total_env_steps,
            "total_hands":          total_hands,
            "best_mean_reward":     best_mean_reward,
            "orchestrator_state":   orchestrator_state,
            "config":               config,
            "rng_states":           rng_states,
            "wandb_run_id":         wandb_run_id,
            "phase0_fixes_version": _PHASE0_FIX_VERSION,
        }

        return self.ckpt_mgr.save(
            checkpoint=checkpoint,
            iteration=iteration,
            is_best=is_best,
        )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load_training_state(
        self,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any] | None:
        if map_location is None:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    map_location = {f"cuda:0": f"cuda:{local_rank}"}
            except Exception:
                pass
        return self.ckpt_mgr.load_latest(map_location=map_location)

    def load_best_state(
        self,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any] | None:
        if map_location is None:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    map_location = {f"cuda:0": f"cuda:{local_rank}"}
            except Exception:
                pass
        return self.ckpt_mgr.load_best(map_location=map_location)

    def has_checkpoint(self) -> bool:
        return self.ckpt_mgr.has_checkpoint()

    def get_latest_path(self) -> Path | None:
        return self.ckpt_mgr.get_latest_path()

    def list_checkpoints(self) -> list[Path]:
        return self.ckpt_mgr.list_checkpoints()
