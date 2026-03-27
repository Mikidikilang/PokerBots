"""
MLOps State Manager (src/mlops/state_manager.py).

Provides two classes for durable, fault-tolerant training state persistence:

CheckpointManager
    Owns every checkpoint file on disk.  The central purpose is to guarantee
    that a checkpoint file is NEVER observed in a partially-written state — not
    by the training loop, not by ``CommitScheduler``, and not on the next
    Kaggle kernel resume after a preemption.

    The guarantee is implemented via POSIX atomic rename:

        torch.save()  →  .tmp file          (failure safe: .tmp is deleted)
        os.fsync()    →  flush OS buffers   (survives hard power-off on ext4)
        os.replace()  →  atomic rename      (readers always see old OR new, never partial)

    This is the standard solution used in production ML systems (JAX checkpoint
    library, PyTorch Lightning, HuggingFace Trainer) and is the same pattern
    already implemented in ``PokerActorCritic.save_checkpoint()`` (networks.py).

StateManager
    High-level façade used by ``runner.py``.  Wraps CheckpointManager and
    exposes a single ``save_training_state()`` / ``load_training_state()``
    interface that bundles the full training state into one checkpoint dict.

Checkpoint dict schema (produced by StateManager.save_training_state):
    {
        "model_state_dict":      OrderedDict   — network weights
        "optimizer_state_dict":  dict          — Adam/SGD momentum state
        "scheduler_state_dict":  dict | None   — LR scheduler state
        "iteration":             int           — current training iteration
        "total_env_steps":       int           — cumulative environment steps
        "total_hands":           int           — cumulative hands played
        "best_mean_reward":      float         — best rolling mean reward seen
        "orchestrator_state":    dict | None   — curriculum + MAB state
        "config":                dict | None   — frozen config snapshot
        "phase0_fixes_version":  str           — "v0.3.0"  (audit trail)
    }

Bug G Fix (this file):
    CheckpointManager.save() previously called ``torch.save(checkpoint, str(filepath))``
    directly.  Any preemption mid-write produced a corrupted ``.pt`` file that
    raised ``RuntimeError`` or ``EOFError`` on the next ``torch.load()`` call,
    permanently destroying the resume capability for that Kaggle session.

    The fix: write to ``filepath + '.tmp'``, call ``os.fsync()`` on the file
    descriptor to flush OS write-back cache to physical storage, then
    ``os.replace()`` for the POSIX-atomic rename.  A ``try/except/finally``
    block guarantees the ``.tmp`` file is cleaned up on any failure.
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

# Sentinel string written into every checkpoint for audit / forward-compat checks
_PHASE0_FIX_VERSION: str = "v0.3.0"


# =============================================================================
# RNGStateManager — Deterministic Random State Capture & Restore
# =============================================================================

class RNGStateManager:
    """Captures and restores the state of all random number generators.

    This class supports deterministic resumption across multiple RNG libraries:
      - Python's built-in ``random`` module
      - NumPy's ``numpy.random``
      - PyTorch CPU and CUDA generators
      - DataLoader worker generator (optional)

    Used by ``scripts/train_local.py`` to freeze and restore training
    reproducibility when resuming from a checkpoint.

    Example usage::

        # At train start: set seed and get DataLoader generator
        dl_generator = RNGStateManager.set_global_seed(seed=42)

        # Before saving checkpoint: capture all RNG states
        rng_states = RNGStateManager.capture_states(dataloader_generator=dl_generator)

        # Save rng_states into checkpoint dict

        # On resume: restore all RNG states
        RNGStateManager.restore_states(rng_states)
    """

    @staticmethod
    def set_global_seed(seed: int) -> torch.Generator:
        """Set all global RNG seeds for reproducibility.

        This is the cold-start initialization called once at training begin.
        It sets:
          - Python's random.seed()
          - NumPy's np.random.seed()
          - PyTorch's torch.manual_seed() (CPU and CUDA)
          - Returns a torch.Generator for DataLoaders

        Args:
            seed: Integer seed value.

        Returns:
            torch.Generator configured with the same seed, for use in DataLoader.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create and seed a generator for DataLoader worker_init_fn
        generator = torch.Generator()
        generator.manual_seed(seed)

        logger.debug("Global seed set: %d (torch.Generator also seeded)", seed)
        return generator

    @staticmethod
    def capture_states(dataloader_generator: torch.Generator | None = None) -> dict[str, Any]:
        """Capture the current state of all RNG systems.

        Args:
            dataloader_generator: Optional torch.Generator from DataLoader setup.

        Returns:
            Dict with keys "python_stdlib", "numpy", "torch_cpu", "torch_cuda",
            "dataloader" (if generator provided). Each value is a pickleable state.
        """
        states = {
            "python_stdlib": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()

        if dataloader_generator is not None:
            states["dataloader"] = dataloader_generator.get_state()

        logger.debug(
            "Captured RNG states: keys=%s",
            list(states.keys())
        )
        return states

    @staticmethod
    def restore_states(states: dict[str, Any]) -> None:
        """Restore all RNG systems to a previously captured state.

        Args:
            states: Dict as returned by ``capture_states()``.
                   Any missing keys are silently ignored.
        """
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
    """Manages a directory of versioned checkpoint files with atomic writes.

    Responsibilities:
      - Atomic save:  ``torch.save → .tmp → fsync → os.replace``
      - Rotation:     keeps the ``max_to_keep`` most recent checkpoints and
                      always retains the all-time best checkpoint
      - Discovery:    finds the latest checkpoint for warm-start resumption
      - Load:         wraps ``torch.load`` with weights_only safety and
                      map_location support

    Naming convention:
        ``{checkpoint_dir}/checkpoint_iter_{iteration:08d}.pt``
        ``{checkpoint_dir}/checkpoint_best.pt``

    Example:
        >>> mgr = CheckpointManager("checkpoints/", max_to_keep=5)
        >>> mgr.save({"model_state_dict": ..., "iteration": 100}, iteration=100)
        >>> data = mgr.load_latest()
    """

    CHECKPOINT_GLOB:  str = "checkpoint_iter_*.pt"
    BEST_FILENAME:    str = "checkpoint_best.pt"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
    ) -> None:
        """Initialise the manager and ensure the checkpoint directory exists.

        Args:
            checkpoint_dir: Directory where ``.pt`` files are written.
            max_to_keep:    Maximum number of iteration checkpoints to retain
                            on disk (the best checkpoint is never rotated out).
        """
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.max_to_keep:    int  = max(1, int(max_to_keep))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "CheckpointManager init: dir='%s', max_to_keep=%d",
            self.checkpoint_dir, self.max_to_keep,
        )

    # =========================================================================
    # Core Save — Bug G Fix
    # =========================================================================

    def save(
        self,
        checkpoint: dict[str, Any],
        iteration: int,
        *,
        is_best: bool = False,
    ) -> Path:
        """Write a checkpoint atomically to disk.

        Atomic write protocol:
            1. Serialise to a temporary ``.tmp`` file (hidden from readers).
            2. ``os.fsync()`` the file descriptor — flushes the OS write-back
               cache to the physical storage device, so the bytes survive a
               hard power cut on journalled filesystems (ext4, XFS, APFS).
            3. ``os.replace()`` renames ``.tmp`` to the final path.  POSIX
               guarantees this rename is atomic: readers see either the
               previous file or the new file, never a partial write.

        Cleanup guarantee:
            A ``try/except/finally`` block removes the ``.tmp`` file if any
            step fails, so stale temp files never accumulate.

        Args:
            checkpoint: Dict to serialise (see module docstring for schema).
            iteration:  Current training iteration (used in the filename).
            is_best:    If True, also copies the checkpoint to
                        ``checkpoint_best.pt`` (second atomic write).

        Returns:
            Path to the written checkpoint file.

        Raises:
            RuntimeError: If the write or rename fails for any reason.
                          The ``.tmp`` file is always cleaned up.
        """
        # ── Inject audit trail ────────────────────────────────────────
        checkpoint.setdefault("phase0_fixes_version", _PHASE0_FIX_VERSION)
        checkpoint.setdefault("saved_at_unix", time.time())

        target_path = self._iteration_path(iteration)
        self._atomic_save(checkpoint, target_path)

        # ── Optional: persist as best checkpoint ──────────────────────
        if is_best:
            best_path = self.checkpoint_dir / self.BEST_FILENAME
            self._atomic_save(checkpoint, best_path)
            logger.info(
                "Best checkpoint updated: '%s'  (iter=%d)",
                best_path, iteration,
            )

        # ── Rotate old checkpoints ────────────────────────────────────
        self._rotate()

        size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Checkpoint saved: '%s'  (%.2f MB, iter=%d, is_best=%s)",
            target_path, size_mb, iteration, is_best,
        )
        return target_path

    # =========================================================================
    # Core Load
    # =========================================================================

    def load(
        self,
        filepath: str | Path,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint from disk.

        Args:
            filepath:     Path to a ``.pt`` checkpoint file.
            map_location: Device string or ``torch.device`` passed to
                          ``torch.load``.  Defaults to ``"cpu"`` so that
                          GPU checkpoints can be loaded on CPU-only machines.

        Returns:
            The deserialized checkpoint dict.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist.
            RuntimeError:      If deserialisation fails (e.g., file is corrupt
                               from a previous non-atomic write).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: '{filepath}'"
            )

        logger.info("Loading checkpoint from '%s' (map_location=%s)", filepath, map_location)

        try:
            # Phase 3-18: Use weights_only=True for secure deserialization.
            # This prevents arbitrary code execution during unpickling.
            # Requires PyTorch 2.6+ for full optimizer state support.
            checkpoint: dict[str, Any] = torch.load(
                str(filepath),
                map_location=map_location,
                weights_only=True,
            )
        except (RuntimeError, EOFError, Exception) as exc:
            raise RuntimeError(
                f"Failed to load checkpoint from '{filepath}': {exc}\n"
                "The file may be corrupted.  This is a known consequence of "
                "non-atomic writes on preempted Kaggle kernels.  "
                "Use CheckpointManager.save() (with atomic writes) going forward."
            ) from exc

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
        """Load the most recent iteration checkpoint, or None if none exist.

        Args:
            map_location: Passed through to ``torch.load``.

        Returns:
            Checkpoint dict, or ``None`` if the directory is empty.
        """
        latest = self.get_latest_path()
        if latest is None:
            logger.info(
                "No existing checkpoints in '%s'. Starting fresh.",
                self.checkpoint_dir,
            )
            return None
        return self.load(latest, map_location=map_location)

    def load_best(
        self,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any] | None:
        """Load the best checkpoint, or None if it does not exist.

        Args:
            map_location: Passed through to ``torch.load``.

        Returns:
            Checkpoint dict, or ``None`` if no best checkpoint exists.
        """
        best_path = self.checkpoint_dir / self.BEST_FILENAME
        if not best_path.exists():
            return None
        return self.load(best_path, map_location=map_location)

    # =========================================================================
    # Discovery Helpers
    # =========================================================================

    def get_latest_path(self) -> Path | None:
        """Return the path of the most recently saved iteration checkpoint.

        Iteration checkpoints are identified by glob ``checkpoint_iter_*.pt``.
        The latest is the one with the highest iteration number (parsed from
        the filename), not the most recently modified file (mtime can lie on
        network filesystems).

        Returns:
            ``Path`` to the latest checkpoint, or ``None`` if none exist.
        """
        paths = self._list_iteration_checkpoints()
        if not paths:
            return None
        return max(paths, key=self._parse_iteration)

    def list_checkpoints(self) -> list[Path]:
        """Return all iteration checkpoint paths, sorted by iteration (oldest first).

        Returns:
            List of ``Path`` objects; empty if no checkpoints exist.
        """
        return sorted(
            self._list_iteration_checkpoints(),
            key=self._parse_iteration,
        )

    def has_checkpoint(self) -> bool:
        """Return True if at least one checkpoint exists in the directory."""
        return len(self._list_iteration_checkpoints()) > 0

    # =========================================================================
    # Atomic Write — Private Implementation
    # =========================================================================

    def _atomic_save(
        self,
        obj: Any,
        target_path: Path,
    ) -> None:
        """Serialise ``obj`` to ``target_path`` via a temp file + fsync + rename.

        This is the single place where the atomic write pattern lives.
        All public save methods delegate here.

        Args:
            obj:         Any pickle-serialisable object (typically a dict).
            target_path: Final destination path.

        Raises:
            RuntimeError: If the write or rename fails.  The ``.tmp`` file is
                          guaranteed to be removed.
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path = target_path.with_suffix(".pt.tmp")

        try:
            # ── Step 1: serialise to the hidden temp file ─────────────
            torch.save(obj, str(tmp_path))

            # ── Step 2: flush OS write-back cache to storage ──────────
            # Opening the file and calling fsync() guarantees the bytes
            # hit the block device before we rename.  Without fsync(),
            # the rename is atomic but the data may still be in the OS
            # page cache — a power failure between rename and the
            # implicit fsync would produce a zero-byte or partial file.
            try:
                with open(str(tmp_path), "rb") as fh:
                    os.fsync(fh.fileno())
            except OSError as fsync_err:
                # fsync() can fail on network filesystems (NFS, tmpfs).
                # Log but don't abort — the rename still provides
                # crash-safety against process preemption (the most
                # common failure mode on Kaggle).
                logger.warning(
                    "os.fsync() failed for '%s' (%s). "
                    "Continuing with os.replace() — data may not be "
                    "durable against hard power-off on this filesystem.",
                    tmp_path, fsync_err,
                )

            # ── Step 3: atomic rename ─────────────────────────────────
            os.replace(str(tmp_path), str(target_path))

        except Exception as exc:
            # ── Cleanup: always remove the temp file on failure ────────
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                    logger.debug("Cleaned up temp file '%s'.", tmp_path)
            except OSError as cleanup_err:
                logger.warning(
                    "Could not remove temp file '%s': %s",
                    tmp_path, cleanup_err,
                )
            raise RuntimeError(
                f"Atomic checkpoint save failed for '{target_path}': {exc}"
            ) from exc

    # =========================================================================
    # Rotation & Naming — Private Helpers
    # =========================================================================

    def _rotate(self) -> None:
        """Delete the oldest iteration checkpoints beyond ``max_to_keep``.

        The best checkpoint (``checkpoint_best.pt``) is never rotated.
        """
        paths = self.list_checkpoints()      # sorted oldest → newest
        excess = len(paths) - self.max_to_keep
        if excess <= 0:
            return
        for old_path in paths[:excess]:
            try:
                old_path.unlink()
                logger.debug("Rotated old checkpoint: '%s'", old_path)
            except OSError as exc:
                logger.warning(
                    "Could not delete old checkpoint '%s': %s", old_path, exc
                )

    def _iteration_path(self, iteration: int) -> Path:
        """Return the canonical filename for a given iteration number."""
        return self.checkpoint_dir / f"checkpoint_iter_{iteration:08d}.pt"

    def _list_iteration_checkpoints(self) -> list[Path]:
        """Return all paths matching the iteration checkpoint glob."""
        return list(self.checkpoint_dir.glob(self.CHECKPOINT_GLOB))

    @staticmethod
    def _parse_iteration(path: Path) -> int:
        """Extract the iteration number from a checkpoint filename.

        Returns 0 if the filename does not match the expected pattern.
        """
        try:
            stem = path.stem                      # "checkpoint_iter_00001000"
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0


# =============================================================================
# StateManager — High-level façade for runner.py
# =============================================================================

class StateManager:
    """Bundles the full training state into a single checkpoint dict.

    This is the class ``runner.py`` should instantiate.  It delegates all
    I/O to ``CheckpointManager`` so that the atomic-write guarantee is always
    in effect.

    Usage in runner.py::

        self.state_manager = StateManager(
            checkpoint_dir=cfg["mlops"]["checkpoint_dir"],
            max_to_keep=cfg["mlops"].get("max_checkpoints", 5),
        )

        # Save at the end of each iteration
        self.state_manager.save_training_state(
            network=self.network,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            iteration=self.iteration,
            total_env_steps=self.total_env_steps,
            total_hands=self.total_hands,
            best_mean_reward=self.best_mean_reward,
            orchestrator_state=self.orchestrator.get_state(),
            config=self.config,
            is_best=(mean_reward > self.best_mean_reward),
        )

        # Resume at startup
        state = self.state_manager.load_training_state()
        if state is not None:
            self.network.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.iteration = state["iteration"]
            ...
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
    ) -> None:
        """Initialise the StateManager.

        Args:
            checkpoint_dir: Directory for ``.pt`` checkpoint files.
            max_to_keep:    Passed through to ``CheckpointManager``.
        """
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
        """Construct from the full ``config.yaml`` dict.

        Reads ``cfg["mlops"]["checkpoint"]["local_checkpoint_dir"]`` and
        ``cfg["mlops"]["checkpoint"]["max_checkpoints_to_keep"]``.

        Args:
            cfg: Full YAML configuration dictionary.

        Returns:
            Configured ``StateManager`` instance.
        """
        mlops_cfg = cfg.get("mlops", {})
        checkpoint_cfg = mlops_cfg.get("checkpoint", {})
        return cls(
            checkpoint_dir=checkpoint_cfg.get("local_checkpoint_dir", "checkpoints"),
            max_to_keep=int(checkpoint_cfg.get("max_checkpoints_to_keep", 5)),
        )

    # =========================================================================
    # Save
    # =========================================================================

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
        is_best:            bool         = False,
    ) -> Path:
        """Assemble and atomically save the complete training state.

        All arguments are keyword-only to prevent silent positional mistakes
        in runner.py call sites.

        Args:
            network:            The ``PokerActorCritic`` instance.
            optimizer:          The ``torch.optim.Optimizer`` being used.
            iteration:          Current training iteration index.
            total_env_steps:    Cumulative environment steps across all
                                training sessions.
            total_hands:        Cumulative poker hands played.
            best_mean_reward:   The best rolling mean reward observed so far,
                                used to decide ``is_best`` in future calls.
            scheduler:          Optional LR scheduler; its state dict is saved
                                if present.
            orchestrator_state: Serialisable dict from
                                ``AutoAdaptiveOrchestrator.get_state()``.
            config:             A serialisable snapshot of the full config
                                (for reproducibility auditing).
            rng_states:         RNG state dict from ``RNGStateManager.capture_states()``.
                                Included in checkpoint for deterministic resumption.
            is_best:            If True, the checkpoint is also written as
                                ``checkpoint_best.pt``.

        Returns:
            Path to the written iteration checkpoint file.
        """
        checkpoint: dict[str, Any] = {
            "model_state_dict":     network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "iteration":            iteration,
            "total_env_steps":      total_env_steps,
            "total_hands":          total_hands,
            "best_mean_reward":     best_mean_reward,
            "orchestrator_state":   orchestrator_state,
            "config":               config,
            "rng_states":           rng_states,
            "phase0_fixes_version": _PHASE0_FIX_VERSION,
        }

        return self.ckpt_mgr.save(
            checkpoint=checkpoint,
            iteration=iteration,
            is_best=is_best,
        )

    # =========================================================================
    # Load
    # =========================================================================

    def load_training_state(
        self,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any] | None:
        """Load the most recent training state, or ``None`` if starting fresh.

        Args:
            map_location: Passed to ``torch.load``; defaults to ``"cpu"``
                          so GPU checkpoints load on CPU-only resume machines.

        Returns:
            The full checkpoint dict, or ``None`` if no checkpoint exists.
        """
        return self.ckpt_mgr.load_latest(map_location=map_location)

    def load_best_state(
        self,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any] | None:
        """Load the best-ever checkpoint, or ``None`` if it does not exist."""
        return self.ckpt_mgr.load_best(map_location=map_location)

    # =========================================================================
    # Convenience Delegation
    # =========================================================================

    def has_checkpoint(self) -> bool:
        """Return True if any checkpoint exists (used by runner.py at startup)."""
        return self.ckpt_mgr.has_checkpoint()

    def get_latest_path(self) -> Path | None:
        """Return the path of the most recent checkpoint, or None."""
        return self.ckpt_mgr.get_latest_path()

    def list_checkpoints(self) -> list[Path]:
        """Return all checkpoint paths sorted oldest-first."""
        return self.ckpt_mgr.list_checkpoints()
