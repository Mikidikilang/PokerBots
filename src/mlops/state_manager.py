"""
Checkpoint Szerializacio es RNG Allapotkezelo (state_manager.py).

A megszakitas nelkuli folytatas (resume training) megkoveteli a
sztochasztikus folyamatok tokeletes reprodukalhatosagat. Ez a modul
ket fo feladatot lat el:

    1. RNGStateManager: Ot kulonbozo veletlenszam-generator
       allapotanak mentes es visszaallitasa:
         - Python native random
         - NumPy np.random
         - PyTorch CPU (torch.get_rng_state)
         - PyTorch CUDA (torch.cuda.get_rng_state_all)
         - DataLoader izolalt torch.Generator

    2. CheckpointManager: A halozat, optimizer, scheduler, RNG
       allapotok es az Orchestrator kontextusanak egyseges
       szerializacioja es deszerializacioja.

Hivatkozasok:
    - Specifikacio: state_manager.py — Checkpoint, RNGStateManager
    - Infrastruktura doc: Reprodukalhatosag es RNG kontroll
    - PyTorch Reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# RNG Allapotkezelo
# =============================================================================

class RNGStateManager:
    """Kezeli az osszes sztochasztikus generator allapotanak menteset es visszaallitasat.

    A poker kornyezet tele van veletlenszeruseggel (kartyaosztas, epsilon-greedy
    felfedezes, dropout retegek). Ha a rendszer ujraindul a letoltott sulyokkal
    de az RNG-k alapertelmezett seed-del indulnak, a Markov Dontesi Folyamat
    belso statisztikai ervenyessege serul.

    Tamogatott komponensek:
        - python_stdlib: random.getstate() / random.setstate()
        - numpy: np.random.get_state() / np.random.set_state()
        - torch_cpu: torch.get_rng_state() / torch.set_rng_state()
        - torch_cuda: torch.cuda.get_rng_state_all() (tobbGPU tamogatas)
        - dataloader: Izolalt torch.Generator a DataLoader shuffle-hoz

    Example:
        >>> rng_mgr = RNGStateManager()
        >>> states = rng_mgr.capture_states()
        >>> # ... training lepesek ...
        >>> rng_mgr.restore_states(states)  # Determinisztikus folytatas
    """

    @staticmethod
    def capture_states(
        dataloader_generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Lefenykepezi a globalis es lokalis generatorok aktualis allapotat.

        Args:
            dataloader_generator: Opcionalis izolalt DataLoader generator.
                Ha megadott, ennek allapota is mentodesre kerul.

        Returns:
            Szerializalhato szotar az osszes RNG allapottal.
        """
        states: dict[str, Any] = {
            "python_stdlib": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }

        # GPU allapotok dinamikus mentese ha elerheto
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
            logger.debug(
                "CUDA RNG allapot mentve: %d GPU",
                len(states["torch_cuda"]),
            )
        else:
            states["torch_cuda"] = None
            logger.debug("CUDA nem elerheto, GPU RNG kihagyva.")

        # DataLoader izolalt generator
        if dataloader_generator is not None:
            states["dataloader"] = dataloader_generator.get_state()
            logger.debug("DataLoader generator allapot mentve.")
        else:
            states["dataloader"] = None

        logger.info(
            "RNG allapotok lefenykepezve: python=%s, numpy=%s, "
            "cpu=%s, cuda=%s, dataloader=%s",
            type(states["python_stdlib"]).__name__,
            type(states["numpy"]).__name__,
            "tensor" if states["torch_cpu"] is not None else "None",
            "list" if states["torch_cuda"] is not None else "None",
            "tensor" if states["dataloader"] is not None else "None",
        )

        return states

    @staticmethod
    def restore_states(
        states: dict[str, Any],
        dataloader_generator: torch.Generator | None = None,
    ) -> None:
        """Visszaallitja a generatorokat a pontosan a menteskori allapotra.

        Args:
            states: A capture_states() altal visszaadott szotar.
            dataloader_generator: Az izolalt DataLoader generator
                (ugyanaz a peldany mint a capture_states-nel).

        Raises:
            RuntimeError: Ha az allapotok formatuma ervenytelen.
        """
        if not states:
            logger.warning(
                "Nincsenek tarolt RNG allapotok. Alapertelmezett mukodes."
            )
            return

        restored: list[str] = []

        try:
            # Python stdlib
            if "python_stdlib" in states and states["python_stdlib"] is not None:
                random.setstate(states["python_stdlib"])
                restored.append("python")

            # NumPy
            if "numpy" in states and states["numpy"] is not None:
                np.random.set_state(states["numpy"])
                restored.append("numpy")

            # PyTorch CPU
            if "torch_cpu" in states and states["torch_cpu"] is not None:
                torch.set_rng_state(states["torch_cpu"])
                restored.append("torch_cpu")

            # PyTorch CUDA
            if (torch.cuda.is_available()
                    and states.get("torch_cuda") is not None):
                torch.cuda.set_rng_state_all(states["torch_cuda"])
                restored.append("torch_cuda")

            # DataLoader generator
            if (dataloader_generator is not None
                    and states.get("dataloader") is not None):
                dataloader_generator.set_state(states["dataloader"])
                restored.append("dataloader")

            logger.info(
                "RNG allapotok visszaallitva: %s (%d/%d komponens)",
                restored, len(restored), 5,
            )

        except Exception as exc:
            logger.error(
                "Hiba az RNG allapotok visszaallitasa kozben: %s. "
                "A replikalhatosag a jelenlegi fazisban elveszhet.",
                exc,
            )

    @staticmethod
    def set_global_seed(seed: int) -> torch.Generator | None:
        """Globalis seed beallitas az osszes generatorhoz.

        Az elso indulaskor (cold start) hasznalando a determinizmus
        garantalasahoz. Resume training eseten a capture/restore
        allapotok felulirjak.

        Args:
            seed: A reprodukalhatosagi seed ertek.

        Returns:
            Egy izolalt torch.Generator a DataLoader szamara (opcionalisan).
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        dl_generator: torch.Generator | None = None

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Izolalt DataLoader generator
        dl_generator = torch.Generator()
        dl_generator.manual_seed(seed)

        logger.info(
            "Globalis seed beallitva: %d (python, numpy, torch_cpu%s, dataloader)",
            seed,
            ", torch_cuda" if torch.cuda.is_available() else "",
        )

        return dl_generator


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Egyseges checkpoint szerializacio es deszerializacio.

    A checkpoint csomag tartalma:
        - model_state_dict: A halozat sulyai
        - optimizer_state_dict: Az Adam optimizer allapota
        - rng_states: Az RNGStateManager altal mentett allapotok
        - orchestrator_state: A curriculum, MAB, telemetria allapotok
        - training_meta: Iteracio szam, lepesek, epizodok
        - config: A YAML konfiguracio pillanatkep

    Example:
        >>> ckpt_mgr = CheckpointManager("checkpoints/", max_keep=5)
        >>> ckpt_mgr.save(network, optimizer, rng_states, orch_state, meta)
        >>> loaded = ckpt_mgr.load_latest()

    Attributes:
        checkpoint_dir: A checkpoint konyvtar eleresi utja.
        max_checkpoints: Maximalis megorzendo checkpoint szam.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
    ) -> None:
        """Inicializalja a CheckpointManager-t.

        Args:
            checkpoint_dir: A checkpoint konyvtar eleresi utja.
            max_checkpoints: Maximalis megorzendo checkpoint szam (rotacio).
        """
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints: int = max_checkpoints

        logger.info(
            "CheckpointManager inicializalva: dir=%s, max_keep=%d",
            self.checkpoint_dir, max_checkpoints,
        )

    # =========================================================================
    # Mentes
    # =========================================================================

    def save(
        self,
        network: Any,
        optimizer: Any | None = None,
        rng_states: dict[str, Any] | None = None,
        orchestrator_state: dict[str, Any] | None = None,
        training_meta: dict[str, Any] | None = None,
        iteration: int = 0,
        config_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """Teljes checkpoint csomag mentese.

        Args:
            network: Az ActorCriticNetwork peldany.
            optimizer: Az Adam optimizer (opcionalisan).
            rng_states: Az RNGStateManager.capture_states() kimenete.
            orchestrator_state: Az Orchestrator.get_state() kimenete.
            training_meta: Extra metaadatok (iteracio, lepesek stb.).
            iteration: Az aktualis iteracio szama (fajlnev generalashoz).
            config_snapshot: A YAML konfiguracio pillanatkep.

        Returns:
            Az elmentett fajl eleresi utja.
        """
        filename: str = f"checkpoint_iter_{iteration:08d}.pt"
        filepath: Path = self.checkpoint_dir / filename

        checkpoint: dict[str, Any] = {
            "model_state_dict": network.state_dict(),
            "iteration": iteration,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if rng_states is not None:
            checkpoint["rng_states"] = rng_states

        if orchestrator_state is not None:
            checkpoint["orchestrator_state"] = orchestrator_state

        if training_meta is not None:
            checkpoint["training_meta"] = training_meta

        if config_snapshot is not None:
            checkpoint["config_snapshot"] = config_snapshot

        try:
            torch.save(checkpoint, str(filepath))

            if filepath.exists():
                size_bytes: int = filepath.stat().st_size
                size_mb: float = size_bytes / (1024 * 1024)
            else:
                size_mb = 0.0

            logger.info(
                "Checkpoint mentve: %s (%.2f MB, iter=%d)",
                filepath, size_mb, iteration,
            )
        except Exception as exc:
            logger.error("Checkpoint mentes sikertelen: %s — %s", filepath, exc)
            raise

        # Regi checkpoint-ok rotacioja
        self._rotate_checkpoints()

        return str(filepath)

    # =========================================================================
    # Betoltes
    # =========================================================================

    def load_latest(
        self,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any] | None:
        """Betolti a legfrissebb checkpoint-ot.

        Args:
            device: A cel eszkoz ("cpu" vagy "cuda").

        Returns:
            A checkpoint szotar, vagy None ha nincs checkpoint.
        """
        checkpoints: list[Path] = self._list_checkpoints()

        if not checkpoints:
            logger.info("Nincs elerheto checkpoint. Scratch inditas.")
            return None

        latest: Path = checkpoints[-1]  # Sorrendben az utolso
        return self.load_from_path(str(latest), device)

    def load_from_path(
        self,
        filepath: str,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Betolt egy specifikus checkpoint fajlt.

        Args:
            filepath: A .pt fajl eleresi utja.
            device: A cel eszkoz.

        Returns:
            A checkpoint szotar a kovetkezo lehetseges kulcsokkal:
                model_state_dict, optimizer_state_dict, rng_states,
                orchestrator_state, training_meta, config_snapshot, iteration.
        """
        logger.info("Checkpoint betoltes: %s -> device=%s", filepath, device)

        try:
            checkpoint: dict[str, Any] = torch.load(
                filepath, map_location=device, weights_only=False
            )

            logger.info(
                "Checkpoint betoltve: iter=%d, kulcsok=%s",
                checkpoint.get("iteration", -1),
                list(checkpoint.keys()),
            )

            return checkpoint

        except Exception as exc:
            logger.error(
                "Checkpoint betoltes sikertelen: %s — %s", filepath, exc
            )
            raise

    def restore_full_state(
        self,
        checkpoint: dict[str, Any],
        network: Any,
        optimizer: Any | None = None,
        dataloader_generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Visszaallitja a teljes rendszerallapotot egy checkpoint-bol.

        Ez az egyetlen metodus, ami a teljes resume-ot vegzi:
            1. Halozati sulyok betoltese
            2. Optimizer allapot visszaallitasa
            3. RNG allapotok visszaallitasa
            4. Training meta visszaadasa

        Args:
            checkpoint: A load_latest() vagy load_from_path() kimenete.
            network: Az ActorCriticNetwork peldany (sulyok betoltese ide).
            optimizer: Az optimizer peldany (allapot visszaallitasa ide).
            dataloader_generator: Az izolalt DataLoader generator.

        Returns:
            A checkpoint-ban tarolt training_meta es orchestrator_state.
        """
        # 1. Halozati sulyok
        if "model_state_dict" in checkpoint:
            network.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Halozati sulyok visszaallitva.")

        # 2. Optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer allapot visszaallitva.")

        # 3. RNG allapotok
        if "rng_states" in checkpoint:
            RNGStateManager.restore_states(
                checkpoint["rng_states"],
                dataloader_generator=dataloader_generator,
            )

        # Visszaadott meta
        result: dict[str, Any] = {
            "iteration": checkpoint.get("iteration", 0),
            "training_meta": checkpoint.get("training_meta", {}),
            "orchestrator_state": checkpoint.get("orchestrator_state", {}),
        }

        logger.info(
            "Teljes allapot visszaallitva: iter=%d",
            result["iteration"],
        )

        return result

    # =========================================================================
    # Checkpoint Rotacio
    # =========================================================================

    def _rotate_checkpoints(self) -> None:
        """Torli a legregiebbi checkpoint-okat ha meghaladja a max limitet."""
        checkpoints: list[Path] = self._list_checkpoints()

        while len(checkpoints) > self.max_checkpoints:
            oldest: Path = checkpoints.pop(0)
            try:
                oldest.unlink()
                logger.debug("Regi checkpoint torolve: %s", oldest)
            except OSError as exc:
                logger.warning("Checkpoint torles sikertelen: %s — %s", oldest, exc)

    def _list_checkpoints(self) -> list[Path]:
        """Visszaadja a checkpoint fajlok listajat idorendben.

        Returns:
            Path objektumok listaja (legregibbtol a legfrissabbig).
        """
        pattern: str = "checkpoint_iter_*.pt"
        checkpoints: list[Path] = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        )
        return checkpoints

    def get_checkpoint_count(self) -> int:
        """Visszaadja a tarolt checkpoint-ok szamat.

        Returns:
            Checkpoint-ok darabszama.
        """
        return len(self._list_checkpoints())

    def has_checkpoint(self) -> bool:
        """Ellenorzi, hogy van-e elerheto checkpoint.

        Returns:
            True ha legalabb egy checkpoint letezik.
        """
        return self.get_checkpoint_count() > 0
