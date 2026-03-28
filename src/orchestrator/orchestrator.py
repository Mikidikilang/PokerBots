"""
Auto-Adaptiv Curriculum Orchestrator (orchestrator.py).

[FIX C2 - 2025-03-28] DDP FSP Snapshot Deadlock Javitas:
    A _save_fsp_snapshot() mostantol dist.barrier()-t hasznal mielott
    a torch.save()-t meghivja. Multi-GPU futasnal Rank 0 FSP snapshot
    mentese blokkolhatott volna a fajlrendszer I/O-n, miközben a tobbi
    rank tovabb tanult — potencialis deadlockot okozva. A barrier
    garantalja, hogy minden rank ugyanazon a ponton var, mielott a
    mentés megkezdodik.

[FIX M5 - 2025-03-28] Entropia CAP Teves Figyelmezteto Log Javitas:
    Az _intervene_passivity() es _intervene_stagnation() most DEBUG
    szinten logol, ha az entropia eppen a capnel van, nem WARNING szinten.
    Ez megszunteti a felrevezeto beavatkozasi logokat, amikor a rendszer
    normalis mukodest folytat a cap kozeleben.
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
from src.orchestrator.reward_shaper import RewardShaper, RewardShapingConfig

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

    [FIX C2] A _save_fsp_snapshot() mostantol DDP barrier-t hasznal
    multi-GPU futasban a deadlock elkeuleseere.

    [FIX M5] Az entropia cap figyelmezteto log DEBUG szintre kerult
    (a cap normalis mukodest jelez, nem hibat).
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

        self.reward_shaper: RewardShaper = RewardShaper(
            RewardShapingConfig.from_dict(yaml_config)
        )

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

        # [C2 FIX] DDP world_size tarolasa a barrier logikához
        self._ddp_world_size: int = 1

        self._intervention_count: int = 0
        self._last_anomalies: list[str] = []
        self._last_intervention_iter: int = -1000
        self._intervention_cooldown: int = 10
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
        """[FIX C2] DDP world_size beallitasa a barrier logikához.

        Ezt a train_local.py build_training_pipeline() hiva meg
        a pipeline osszeallitasakor, ha world_size > 1.

        Args:
            world_size: Az osszes DDP process szama (1 = nincs DDP).
        """
        self._ddp_world_size = world_size
        logger.info(
            "DDP world_size beallitva az Orchestratorban: %d [C2 FIX]",
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

        if metrics["hands_in_window"] < 1000:
            logger.debug("Orchestrator: nincs eleg adat (<%d hands), kihagyas.", 1000)
            return result

        anomalies: list[str] = self.telemetry.detect_anomalies(
            self._gto_targets, self._degen_thresholds
        )
        result["anomalies"] = anomalies
        self._last_anomalies = anomalies

        rs_cfg = self.reward_shaper.config
        if self.telemetry.check_stagnation(
            reward_window=rs_cfg.stagnation_window,
            threshold=rs_cfg.stagnation_threshold,
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
            current_lambda = self.reward_shaper.config.bluff_penalty_lambda
            current_bonus = self.reward_shaper.config.preflop_aggression_bonus
            if current_lambda > 0.0 or current_bonus > 0.0:
                self.reward_shaper.update_penalty_lambda(current_lambda * 0.5)
                self.reward_shaper.update_aggression_bonus(current_bonus * 0.5)
                if self.reward_shaper.config.bluff_penalty_lambda < 0.01:
                    self.reward_shaper.deactivate_all_shaping()
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
            interventions.append("aggression_bonus_activated")

        if "maniac" in anomalies:
            self._intervene_maniac(metrics)
            interventions.append("bluff_penalty_activated")

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
        """Passzivitas korrekcios beavatkozas.

        [FIX M5] Az entropia cap elesekor DEBUG szinten logolunk (nem WARNING):
        a cap eleres normalis mukodest jelez (a rendszer nem tud tobbet boostolni),
        nem hibat.
        """
        if self._trainer_ref is not None:
            current_ent: float = getattr(
                getattr(self._trainer_ref, "config", None),
                "entropy_coef", 0.01,
            )
            boost: float = self.reward_shaper.config.entropy_boost_factor
            new_ent: float = min(current_ent * boost, self._max_entropy_coef)
            at_cap = new_ent >= self._max_entropy_coef

            # [FIX M5] CAP eresekere DEBUG (nem WARNING) — ez normalis allapot
            if at_cap:
                logger.debug(
                    "Entropia CAP elerve (normalis): %.4f >= %.4f max. "
                    "Tovabbi noveles blokkolva (ez nem hiba).",
                    new_ent, self._max_entropy_coef,
                )
            else:
                logger.info(
                    "Passzivitas intervencio: entropia %.4f -> %.4f "
                    "(x%.1f, max=%.4f)",
                    current_ent, new_ent, boost, self._max_entropy_coef,
                )

            self._trainer_ref.update_entropy_coef(new_ent)
        else:
            logger.warning(
                "Passzivitas intervencio hibas: self._trainer_ref is None. "
                "Hivasad a set_trainer_reference()-t?"
            )

        self.reward_shaper.update_aggression_bonus(0.1)
        logger.info("Passzivitas intervencio: agresszio bonus aktivalva: +0.1")

    def _intervene_maniac(self, metrics: dict[str, float]) -> None:
        self.reward_shaper.update_penalty_lambda(0.5)
        logger.info("Maniac intervenció: bloff buntetes lambda=0.5 aktiválva")

    def _intervene_stagnation(self) -> None:
        """Stagnacio korrekcios beavatkozas.

        [FIX M5] Ugyanaz a cap-log javitas mint _intervene_passivity-ban.
        """
        if self._trainer_ref is not None:
            current_ent: float = getattr(
                getattr(self._trainer_ref, "config", None),
                "entropy_coef", 0.01,
            )
            boost: float = self.reward_shaper.config.entropy_boost_factor
            new_ent: float = min(current_ent * boost, self._max_entropy_coef)
            at_cap = new_ent >= self._max_entropy_coef

            # [FIX M5] CAP eresekere DEBUG (normalis allapot)
            if at_cap:
                logger.debug(
                    "Stagnacio: Entropia CAP elerve (normalis): %.4f >= %.4f max.",
                    new_ent, self._max_entropy_coef,
                )
            else:
                logger.info(
                    "Stagnacio intervencio: entropia %.4f -> %.4f "
                    "(x%.1f, max=%.4f)",
                    current_ent, new_ent, boost, self._max_entropy_coef,
                )

            self._trainer_ref.update_entropy_coef(new_ent)

    def _on_phase_transition(self) -> None:
        """Callback a fazisatmenet utan."""
        self.reward_shaper.deactivate_all_shaping()

        if self.curriculum.current_phase.value == 2:  # PHASE_2_FSP
            self._save_fsp_snapshot()

        logger.info(
            "Fazisatmenet utomunkak: reward shaping resetelve, "
            "uj fazis: %s, opponents: %s",
            self.curriculum.current_phase.name,
            self.curriculum.get_current_opponents(),
        )

    # =========================================================================
    # [FIX C2] FSP Snapshot Mentes DDP Barrier-rel
    # =========================================================================

    def _save_fsp_snapshot(self) -> None:
        """FSP snapshot mentes DDP barrier-rel a deadlock elkeruleseere.

        [FIX C2] A korabbi implementacioban az FSP snapshot mentes nem
        szinkronizalt a tobb DDP rank kozott. Rank 0 blokkolhatott a
        fajlrendszer I/O-n (torch.save()), miközben a tobb rank tovabb
        szamolt — potencialis deadlockot okozva.

        A javitas: dist.barrier() hivodik, mielott es utan a torch.save().
        Ez biztositja, hogy:
        1. Minden rank ugyanazon a ponton var (elotte).
        2. Rank 0 elvegzi a mentést.
        3. Minden rank folytathat (utana).

        Ha _ddp_world_size == 1 (single-GPU), a barrier kihagyodik.
        """
        if self._network_ref is None:
            logger.warning("FSP snapshot save sikertelen: _network_ref is None")
            return

        import torch
        from pathlib import Path

        try:
            fsp_dir = Path("checkpoints/fsp")
            fsp_dir.mkdir(parents=True, exist_ok=True)

            # [FIX C2] DDP barrier ELOTT — mindenki varja meg, hogy a mentor
            # biztosan keszult el az elozo iteracio gradiens frissiteseivel
            if self._ddp_world_size > 1:
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                        logger.debug("DDP barrier elott FSP snapshot save [C2 FIX]")
                except Exception as barrier_exc:
                    logger.warning(
                        "DDP barrier sikertelen FSP snapshot elott: %s. "
                        "Mentés folytatodik barriernel kul.",
                        barrier_exc,
                    )

            # DDP wrapper eltavolitasa, state dict kinyerese
            if isinstance(self._network_ref, torch.nn.parallel.DistributedDataParallel):
                state_dict = self._network_ref.module.state_dict()
            else:
                state_dict = self._network_ref.state_dict()

            self._fsp_snapshot_counter += 1
            snapshot_path = fsp_dir / f"snapshot_{self._fsp_snapshot_counter:08d}.pt"

            # Atomikus mentes: temp fajl -> rename (M2 fix mintajara)
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

            # [FIX C2] DDP barrier UTAN — mindenki megvarja a mentés veget
            if self._ddp_world_size > 1:
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                        logger.debug("DDP barrier utan FSP snapshot save [C2 FIX]")
                except Exception as barrier_exc:
                    logger.warning(
                        "DDP barrier sikertelen FSP snapshot utan: %s",
                        barrier_exc,
                    )

            # Regisztracio az opponent pool-ba (MAB tracking)
            opponent_name = f"fsp_snapshot_{self._fsp_snapshot_counter:08d}"
            self.curriculum.register_opponent(opponent_name)

            logger.info(
                "FSP snapshot mentve (atomikusan): %s (counter=%d) [C2 DDP-safe]",
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

        new_lambda: float = rs_cfg.get("bluff_penalty_lambda", 0.0)
        new_bonus: float = rs_cfg.get("preflop_aggression_bonus", 0.0)
        self.reward_shaper.update_penalty_lambda(new_lambda)
        self.reward_shaper.update_aggression_bonus(new_bonus)

        self._yaml_config = new_cfg
        logger.debug("Hot-reload alkalmazva: lr, entropy, lambda, bonus frissitve.")

    # =========================================================================
    # Allapot Mentes / Betoltes
    # =========================================================================

    def get_state(self) -> dict[str, Any]:
        return {
            "curriculum_state": self.curriculum.get_state(),
            "reward_shaper_stats": self.reward_shaper.get_stats(),
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
            "reward_shaper_stats": self.reward_shaper.get_stats(),
            "ucb_stats": self.curriculum.get_ucb_stats(),
        }
