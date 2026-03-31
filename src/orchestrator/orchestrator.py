"""
Auto-Adaptiv Curriculum Orchestrator (orchestrator.py).

[FIX C-1 — 2025-03-28] dist.barrier() REMOVED from _save_fsp_snapshot().

    ROOT CAUSE OF DEADLOCK:
    The orchestrator runs on rank 0 ONLY. When _save_fsp_snapshot() called
    dist.barrier(), rank 0 would block waiting for rank 1 to arrive at the
    same collective. But rank 1 was already executing on_ddp_sync() in
    runner.py where it calls dist.broadcast() — a completely different
    collective operation. NCCL sees barrier on rank 0 and broadcast on rank 1
    = collective mismatch = permanent hang of both processes.

    THE FIX:
    Remove all dist.barrier() calls from _save_fsp_snapshot(). This method
    runs on rank 0 only and performs local file I/O only — it needs zero
    cross-rank synchronization. Phase transition is already communicated to
    all ranks through the on_ddp_sync() broadcast path in runner.py, which
    is the correct and only synchronization point.

[FIX M5 — 2025-03-28] Entropy CAP log level WARNING -> DEBUG (normal operation).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from src.orchestrator.telemetry import TelemetryAnalyzer, HandRecord
from src.orchestrator.curriculum import CurriculumManager, CurriculumPhase

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Az Orchestrator konfiguracioja."""

    config_path: str = "config.yaml"
    num_players: int = 6
    telemetry_window: int = 100_000
    eval_interval: int = 50
    enable_hot_reload: bool = True
    hot_reload_interval_sec: float = 30.0


class AutoAdaptiveOrchestrator:
    """A rendszer fo felugyeleti entitasa (Singleton).

    Rank 0 only. Never calls dist.barrier() — see module docstring.
    """

    _instance: AutoAdaptiveOrchestrator | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> AutoAdaptiveOrchestrator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(
        cls,
        config: OrchestratorConfig | None = None,
        yaml_config: dict[str, Any] | None = None,
    ) -> AutoAdaptiveOrchestrator:
        instance = cls()
        if not hasattr(instance, "_initialized") or not instance._initialized:
            instance._init(config or OrchestratorConfig(), yaml_config or {})
        return instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def _init(
        self, config: OrchestratorConfig, yaml_config: dict[str, Any]
    ) -> None:
        self.config: OrchestratorConfig = config
        self._yaml_config: dict[str, Any] = yaml_config
        self._initialized: bool = True

        self.telemetry: TelemetryAnalyzer = TelemetryAnalyzer(
            window_size=config.telemetry_window,
            num_players=config.num_players,
        )

        self.curriculum: CurriculumManager = CurriculumManager.from_dict(yaml_config)

        table_key: str = str(config.num_players)
        self._gto_targets: dict[str, list[float]] = yaml_config.get(
            "gto_matrix", {}
        ).get(table_key, {})
        self._degen_thresholds: dict[str, dict[str, float]] = yaml_config.get(
            "degeneration_thresholds", {}
        ).get(table_key, {})

        self._config_last_modified: float = 0.0
        self._last_reload_check: float = time.monotonic()
        if config.enable_hot_reload and os.path.exists(config.config_path):
            self._config_last_modified = os.path.getmtime(config.config_path)

        self._trainer_ref: Any = None
        self._network_ref: Any = None
        self._fsp_snapshot_counter: int = 0

        # DDP world_size stored for informational logging only.
        # It is NOT used for dist.barrier() — see module docstring.
        self._ddp_world_size: int = 1

        self._intervention_count: int = 0
        self._last_anomalies: list[str] = []
        self._last_intervention_iter: int = -1000
        self._intervention_cooldown: int = 50
        self._max_entropy_coef: float = 0.1

        logger.info(
            "AutoAdaptiveOrchestrator inicializalva: players=%d, "
            "gto_keys=%s, hot_reload=%s",
            config.num_players,
            list(self._gto_targets.keys()) if self._gto_targets else "NINCS",
            config.enable_hot_reload,
        )

    # =========================================================================
    # Referencia Beallitasok
    # =========================================================================

    @property
    def total_hands(self) -> int:
        return int(self.telemetry._total_hands)

    def set_trainer_reference(self, trainer: Any) -> None:
        self._trainer_ref = trainer
        logger.debug("Trainer referencia beallitva az Orchestrator-ban.")

    def set_network_reference(self, network: Any) -> None:
        self._network_ref = network
        logger.debug("Network referencia beallitva az Orchestrator-ban.")

    def set_ddp_world_size(self, world_size: int) -> None:
        """Store DDP world_size for informational logging.

        NOTE: This value is NOT used for dist.barrier() synchronization.
        The orchestrator runs on rank 0 only and performs local I/O only.
        Cross-rank synchronization happens exclusively in runner.py's
        on_ddp_sync() callback via dist.broadcast().

        Args:
            world_size: Total number of DDP processes (1 = no DDP).
        """
        self._ddp_world_size = world_size
        logger.info(
            "DDP world_size=%d registered in Orchestrator (informational only, "
            "no barriers will be called from this class).",
            world_size,
        )

    # =========================================================================
    # Fo Callback
    # =========================================================================

    def on_iteration_callback(
        self,
        iteration: int,
        stats: dict[str, float],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "iteration": iteration,
            "phase": self.curriculum.current_phase.name,
            "anomalies": [],
            "interventions": [],
            "phase_transition": False,
        }

        if self.config.enable_hot_reload:
            self._check_hot_reload()

        if iteration % self.config.eval_interval != 0:
            return result

        metrics: dict[str, float] = self.telemetry.get_current_metrics()

        if metrics["hands_in_window"] < 5000:
            logger.debug("Orchestrator: nincs eleg adat (<%d hands), kihagyas.", 5000)
            return result

        anomalies: list[str] = self.telemetry.detect_anomalies(
            self._gto_targets, self._degen_thresholds
        )
        result["anomalies"] = anomalies
        self._last_anomalies = anomalies

        if self.telemetry.check_stagnation(
            reward_window=2000,
            threshold=0.02,
        ):
            anomalies.append("stagnation")

        interventions: list[str] = self._execute_interventions(anomalies, metrics, iteration)
        result["interventions"] = interventions

        if self.curriculum.check_phase_transition(metrics, iteration):
            result["phase_transition"] = True
            result["new_phase"] = self.curriculum.current_phase.name
            self._on_phase_transition()

        gto_dist: dict[str, float] = self.telemetry.compute_gto_distance(
            self._gto_targets
        )

        logger.info(
            "Orchestrator iter #%d: phase=%s, anomalies=%s, "
            "interventions=%s, gto_dist=%.2f",
            iteration, self.curriculum.current_phase.name,
            anomalies, interventions,
            gto_dist.get("total", 0.0),
        )

        return result

    # =========================================================================
    # Beavatkozasi Logika
    # =========================================================================

    def _execute_interventions(
        self,
        anomalies: list[str],
        metrics: dict[str, float],
        iteration: int = 0,
    ) -> list[str]:
        interventions: list[str] = []

        if not anomalies:
            return []

        if (iteration - self._last_intervention_iter) < self._intervention_cooldown:
            if anomalies:
                logger.debug(
                    "Beavatkozas cooldown aktiv: %d/%d iteracio. "
                    "Anomaliak (%s) figyelmen kivul hagyva.",
                    iteration - self._last_intervention_iter,
                    self._intervention_cooldown,
                    anomalies,
                )
            return interventions

        if "passivity" in anomalies:
            self._intervene_passivity(metrics)
            interventions.append("entropy_boost")



        if "stagnation" in anomalies:
            self._intervene_stagnation()
            interventions.append("entropy_boost_stagnation")

        if interventions:
            self._intervention_count += len(interventions)
            self._last_intervention_iter = iteration
            logger.info(
                "Beavatkozasok vegrehajtva: %s (osszes: %d, cooldown=%d iter)",
                interventions, self._intervention_count,
                self._intervention_cooldown,
            )

        return interventions

    def _intervene_passivity(self, metrics: dict[str, float]) -> None:
        """Passzivitas korrekcios beavatkozas (entropia boost only).

        Reward shaping has been removed for Deep CFR compatibility.
        [PHASE 1] Entropy boost only.
        """
        if self._trainer_ref is not None:
            current_ent: float = getattr(
                getattr(self._trainer_ref, "config", None),
                "entropy_coef", 0.01,
            )
            # Fixed boost factor (removed from reward_shaper.config)
            boost: float = 1.5
            new_ent: float = min(current_ent * boost, self._max_entropy_coef)
            at_cap = new_ent >= self._max_entropy_coef

            if at_cap:
                logger.debug(
                    "Entropy CAP reached (normal operation): %.4f >= %.4f max. "
                    "Further boost blocked — this is expected behaviour.",
                    new_ent, self._max_entropy_coef,
                )
            else:
                logger.info(
                    "Passzivitas intervencio: entropia %.4f -> %.4f (x%.1f, max=%.4f)",
                    current_ent, new_ent, boost, self._max_entropy_coef,
                )

            self._trainer_ref.update_entropy_coef(new_ent)
        else:
            logger.warning(
                "Passzivitas intervencio hibas: self._trainer_ref is None. "
                "Hivasad a set_trainer_reference()-t?"
            )

    def _intervene_stagnation(self) -> None:
        """Stagnacio korrekcios beavatkozas (entropia boost only).

        Reward shaping has been removed for Deep CFR compatibility.
        [PHASE 1] Entropy boost only.
        """
        if self._trainer_ref is not None:
            current_ent: float = getattr(
                getattr(self._trainer_ref, "config", None),
                "entropy_coef", 0.01,
            )
            # Fixed boost factor (removed from reward_shaper.config)
            boost: float = 1.5
            new_ent: float = min(current_ent * boost, self._max_entropy_coef)
            at_cap = new_ent >= self._max_entropy_coef

            if at_cap:
                logger.debug(
                    "Stagnacio: Entropy CAP reached (normal): %.4f >= %.4f max.",
                    new_ent, self._max_entropy_coef,
                )
            else:
                logger.info(
                    "Stagnacio intervencio: entropia %.4f -> %.4f (x%.1f, max=%.4f)",
                    current_ent, new_ent, boost, self._max_entropy_coef,
                )

            self._trainer_ref.update_entropy_coef(new_ent)

    def _on_phase_transition(self) -> None:
        """Callback a fazisatmenet utan."""
        if self.curriculum.current_phase.value == 2:
            self._save_fsp_snapshot()

        logger.info(
            "Fazisatmenet utomunkak: "
            "uj fazis: %s, opponents: %s",
            self.curriculum.current_phase.name,
            self.curriculum.get_current_opponents(),
        )

    # =========================================================================
    # [FIX C-1] FSP Snapshot — NO dist.barrier(), rank-0-local I/O only
    # =========================================================================

    def _save_fsp_snapshot(self) -> None:
        """Save FSP network snapshot to disk (rank 0, local I/O only).

        [FIX C-1] All dist.barrier() calls have been REMOVED.

        Why they caused a deadlock:
            - This method runs on rank 0 ONLY (orchestrator is rank-0-only).
            - When rank 0 called dist.barrier(), it blocked waiting for rank 1.
            - But rank 1 was executing on_ddp_sync() in runner.py calling
              dist.broadcast() — a different collective operation entirely.
            - NCCL encountered a collective mismatch: permanent hang.

        Why no barrier is needed:
            - This method only calls torch.save() + os.replace() — pure local
              file I/O with no GPU operations and no cross-rank data exchange.
            - Phase transitions are communicated to all ranks via the
              dist.broadcast() call already present in runner.py's on_ddp_sync().
            - That broadcast is the single correct synchronization point.
        """
        if self._network_ref is None:
            logger.warning("FSP snapshot save sikertelen: _network_ref is None")
            return

        import torch

        try:
            fsp_dir = Path("checkpoints/fsp")
            fsp_dir.mkdir(parents=True, exist_ok=True)

            # Unwrap DDP wrapper if present
            if isinstance(self._network_ref, torch.nn.parallel.DistributedDataParallel):
                state_dict = self._network_ref.module.state_dict()
            else:
                state_dict = self._network_ref.state_dict()

            self._fsp_snapshot_counter += 1
            snapshot_path = fsp_dir / f"snapshot_{self._fsp_snapshot_counter:08d}.pt"

            # Atomic write: temp file -> rename (SIGKILL-safe, M2 pattern)
            tmp_path = snapshot_path.with_suffix(".pt.tmp")
            try:
                torch.save(state_dict, str(tmp_path))
                os.replace(str(tmp_path), str(snapshot_path))
            except Exception as save_exc:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise save_exc

            # Register in MAB pool
            opponent_name = f"fsp_snapshot_{self._fsp_snapshot_counter:08d}"
            self.curriculum.register_opponent(opponent_name)

            logger.info(
                "FSP snapshot saved (atomic, rank-0 local I/O, no barrier): "
                "%s (counter=%d)",
                snapshot_path, self._fsp_snapshot_counter,
            )

        except Exception as exc:
            logger.error("FSP snapshot save hiba: %s", exc, exc_info=True)

    # =========================================================================
    # Config Hot-Reload
    # =========================================================================

    def _check_hot_reload(self) -> None:
        now: float = time.monotonic()
        if (now - self._last_reload_check) < self.config.hot_reload_interval_sec:
            return
        self._last_reload_check = now

        if not os.path.exists(self.config.config_path):
            return

        current_mtime: float = os.path.getmtime(self.config.config_path)
        if current_mtime <= self._config_last_modified:
            return

        try:
            with open(self.config.config_path, "r") as f:
                new_cfg: dict[str, Any] = yaml.safe_load(f)

            self._apply_hot_reload(new_cfg)
            self._config_last_modified = current_mtime

            logger.info(
                "Config hot-reload sikeres: %s (mtime=%.0f)",
                self.config.config_path, current_mtime,
            )
        except Exception as exc:
            logger.error(
                "Config hot-reload hiba: %s — %s",
                self.config.config_path, exc,
            )

    def _apply_hot_reload(self, new_cfg: dict[str, Any]) -> None:
        ppo_cfg = new_cfg.get("ppo", {})
        rs_cfg = new_cfg.get("reward_shaping", {})

        if self._trainer_ref is not None:
            new_lr: float = ppo_cfg.get("learning_rate", 3e-4)
            new_ent: float = ppo_cfg.get("entropy_coefficient", 0.01)
            self._trainer_ref.update_learning_rate(new_lr)
            self._trainer_ref.update_entropy_coef(new_ent)

        self._yaml_config = new_cfg
        logger.debug("Hot-reload alkalmazva: lr, entropy frissitve.")

    # =========================================================================
    # Allapot Mentes / Betoltes
    # =========================================================================

    def get_state(self) -> dict[str, Any]:
        return {
            "curriculum_state": self.curriculum.get_state(),
            "intervention_count": self._intervention_count,
            "last_anomalies": self._last_anomalies,
            "telemetry_total_hands": self.telemetry._total_hands,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if "curriculum_state" in state:
            self.curriculum.load_state(state["curriculum_state"])
        self._intervention_count = state.get("intervention_count", 0)
        self._last_anomalies = state.get("last_anomalies", [])
        logger.info("Orchestrator allapot betoltve.")

    # =========================================================================
    # Statisztikak
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        metrics: dict[str, float] = self.telemetry.get_current_metrics()
        return {
            "current_phase": self.curriculum.current_phase.name,
            "current_metrics": metrics,
            "gto_distance": self.telemetry.compute_gto_distance(self._gto_targets),
            "intervention_count": self._intervention_count,
            "last_anomalies": self._last_anomalies,
            "ucb_stats": self.curriculum.get_ucb_stats(),
        }
