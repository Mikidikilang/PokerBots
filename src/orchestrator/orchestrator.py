"""
Auto-Adaptiv Curriculum Orchestrator (orchestrator.py).

A rendszer fo Singleton allapotgepe (State Machine), amely a matematikai
optimalizaciotol fuggetlenul operal. Aszinkron modon figyeli a telemetriat,
strukturalja a tanulasi fazisokat, es dinamikusan beavatkozik ha a halozat
strategiaja leter a GTO palyarol.

Fo felelossegek:
    1. Telemetria fogadas a runner.py callback-en keresztul
    2. GTO Matrix validalas es anomalia detektalas
    3. Beavatkozas: Reward Shaping, Entropia injektalas
    4. Fazisatmenetek vezerles (Phase 0 -> 1 -> 2)
    5. Config hot-reload: YAML fajl monitorozas es futasideju frissites

Hivatkozasok:
    - Specifikacio: orchestrator.py — Singleton allapotgep
    - Curriculum doc: Az Orchestrator reteg tervezesi mintai
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


# =============================================================================
# Orchestrator Konfiguracio
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Az Orchestrator konfiguracioja.

    Attributes:
        config_path: A config.yaml fajl eleresi utja (hot-reload-hoz).
        num_players: Asztal merete.
        telemetry_window: Telemetriai mozgoablak merete.
        eval_interval: Kiertekeles gyakorisaga (iteraciokent).
        enable_hot_reload: Engedelyezett-e a config hot-reload.
        hot_reload_interval_sec: Hot-reload ellenorzes intervalluma masodpercben.
    """

    config_path: str = "config.yaml"
    num_players: int = 6
    telemetry_window: int = 100_000
    eval_interval: int = 50
    enable_hot_reload: bool = True
    hot_reload_interval_sec: float = 30.0


# =============================================================================
# Fo Orchestrator Osztaly (Singleton)
# =============================================================================

class AutoAdaptiveOrchestrator:
    """A rendszer fo felugyeleti entitasa.

    Singleton minta: egy Python processzen belul csak egyetlen
    peldany letezik. A __new__ osztaly metodus biztositja.

    Az Orchestrator a runner.py on_iteration_end callback-jen
    keresztul fogadja a telemetriai adatokat, es a kovetkezo
    beavatkozasokat hajtja vegre:

        - Passzivitas detektalas -> entropia noveles + agressziv botok
        - All-in Spam detektalas -> reward penalty + Calling Station botok
        - Stagnacio detektalas -> entropia boost
        - Fazisatmenet ellenorzes -> curriculum leptetes

    Example:
        >>> orch = AutoAdaptiveOrchestrator.get_instance(cfg, yaml_config)
        >>> # A runner.py callback-keni:
        >>> def on_iter_end(iteration, stats):
        ...     orch.on_iteration_callback(iteration, stats)
        >>> runner = TrainingRunner(config, ..., on_iteration_end=on_iter_end)

    Attributes:
        telemetry: A HUD metrikak analizatora.
        curriculum: A curriculum manager.
        reward_shaper: A jutalom formazo.
        config: Az Orchestrator konfiguracioja.
    """

    _instance: AutoAdaptiveOrchestrator | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> AutoAdaptiveOrchestrator:
        """Singleton minta: csak egy peldany letezik."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(
        cls,
        config: OrchestratorConfig | None = None,
        yaml_config: dict[str, Any] | None = None,
    ) -> AutoAdaptiveOrchestrator:
        """Visszaadja (vagy letrehozza) az egyetlen Orchestrator peldanyt.

        Args:
            config: Orchestrator konfiguracio (csak az elso hivaskor szukseges).
            yaml_config: A teljes YAML konfiguracio (csak az elso hivaskor).

        Returns:
            Az egyetlen AutoAdaptiveOrchestrator peldany.
        """
        instance = cls()
        if not hasattr(instance, "_initialized") or not instance._initialized:
            instance._init(config or OrchestratorConfig(), yaml_config or {})
        return instance

    @classmethod
    def reset_instance(cls) -> None:
        """Torli a Singleton peldanyt (teszteleshez)."""
        cls._instance = None

    def _init(
        self, config: OrchestratorConfig, yaml_config: dict[str, Any]
    ) -> None:
        """Belso inicializalo (csak egyszer fut le).

        Args:
            config: Orchestrator konfiguracio.
            yaml_config: Teljes YAML konfiguracio.
        """
        self.config: OrchestratorConfig = config
        self._yaml_config: dict[str, Any] = yaml_config
        self._initialized: bool = True

        # Telemetria
        self.telemetry: TelemetryAnalyzer = TelemetryAnalyzer(
            window_size=config.telemetry_window,
            num_players=config.num_players,
        )

        # Curriculum Manager
        self.curriculum: CurriculumManager = CurriculumManager.from_dict(yaml_config)

        # Reward Shaper
        self.reward_shaper: RewardShaper = RewardShaper(
            RewardShapingConfig.from_dict(yaml_config)
        )

        # GTO Matrix es kuszobok betoltese
        table_key: str = str(config.num_players)
        self._gto_targets: dict[str, list[float]] = yaml_config.get(
            "gto_matrix", {}
        ).get(table_key, {})
        self._degen_thresholds: dict[str, dict[str, float]] = yaml_config.get(
            "degeneration_thresholds", {}
        ).get(table_key, {})

        # Hot-reload allapot
        self._config_last_modified: float = 0.0
        self._last_reload_check: float = time.monotonic()
        if config.enable_hot_reload and os.path.exists(config.config_path):
            self._config_last_modified = os.path.getmtime(config.config_path)

        # Trainer referencia (kesobb allitja be a runner)
        self._trainer_ref: Any = None

        # Beavatkozas szamlalok es cooldown
        self._intervention_count: int = 0
        self._last_anomalies: list[str] = []
        self._last_intervention_iter: int = -1000  # Cooldown: legutobbi beavatkozas iteracioja
        self._intervention_cooldown: int = 10      # Min iteraciok ket beavatkozas kozott
        self._max_entropy_coef: float = 0.1        # Entropia koefficiensz felso korlat

        logger.info(
            "AutoAdaptiveOrchestrator inicializalva: players=%d, "
            "gto_keys=%s, hot_reload=%s",
            config.num_players,
            list(self._gto_targets.keys()) if self._gto_targets else "NINCS",
            config.enable_hot_reload,
        )

    # =========================================================================
    # Trainer Referencia (Hot-Reload Celra)
    # =========================================================================

    def set_trainer_reference(self, trainer: Any) -> None:
        """Beallitja a PPOTrainer referenciat a hot-reload beavatkozasokhoz.

        Args:
            trainer: A PPOTrainer peldany.
        """
        self._trainer_ref = trainer
        logger.debug("Trainer referencia beallitva az Orchestrator-ban.")

    # =========================================================================
    # Fo Callback (Event-Driven)
    # =========================================================================

    def on_iteration_callback(
        self,
        iteration: int,
        stats: dict[str, float],
    ) -> dict[str, Any]:
        """A runner.py altal hivott callback minden iteracio vegen.

        Ez a metodus a teljes Orchestrator logikajat futtatja:
            1. Hot-reload ellenorzes
            2. Anomalia detektalas
            3. Beavatkozas ha szukseges
            4. Fazisatmenet ellenorzes

        Args:
            iteration: Az aktualis iteracio szama.
            stats: Az iteracio statisztikaival (collect + train).

        Returns:
            Dict az Orchestrator altal vegzett beavatkozasokkal.
        """
        result: dict[str, Any] = {
            "iteration": iteration,
            "phase": self.curriculum.current_phase.name,
            "anomalies": [],
            "interventions": [],
            "phase_transition": False,
        }

        # 1. Hot-reload ellenorzes (periodikus)
        if self.config.enable_hot_reload:
            self._check_hot_reload()

        # 2. Telemetria ellenorzes (csak az eval_interval-nal)
        if iteration % self.config.eval_interval != 0:
            return result

        # 3. Aktualis metrikak lekerdezese
        metrics: dict[str, float] = self.telemetry.get_current_metrics()

        if metrics["hands_in_window"] < 1000:
            logger.debug(
                "Orchestrator: nincs eleg adat (<%d hands), kihagyas.",
                1000,
            )
            return result

        # 4. Anomalia detektalas
        anomalies: list[str] = self.telemetry.detect_anomalies(
            self._gto_targets, self._degen_thresholds
        )
        result["anomalies"] = anomalies
        self._last_anomalies = anomalies

        # 5. Stagnacio ellenorzes
        rs_cfg = self.reward_shaper.config
        if self.telemetry.check_stagnation(
            reward_window=rs_cfg.stagnation_window,
            threshold=rs_cfg.stagnation_threshold,
        ):
            anomalies.append("stagnation")

        # 6. Beavatkozas (cooldown-nal)
        interventions: list[str] = self._execute_interventions(anomalies, metrics, iteration)
        result["interventions"] = interventions

        # 7. Fazisatmenet ellenorzes
        if self.curriculum.check_phase_transition(metrics, iteration):
            result["phase_transition"] = True
            result["new_phase"] = self.curriculum.current_phase.name
            self._on_phase_transition()

        # 8. GTO tavolsag logolas
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
        """Vegrehajtja a szukseges beavatkozasokat az anomaliak alapjan.

        A beavatkozasok cooldown-nal vannak vedve: ket beavatkozas kozott
        legalabb ``_intervention_cooldown`` iteracioanak kell eltelnie,
        megakadalyozva az oszcillalo beavatkozasi kaszkadot.

        A specifikacio altal definialt intervenciok:
            - passivity -> entropia noveles + agressziv botok
            - maniac -> reward penalty + calling station botok
            - stagnation -> entropia boost

        Args:
            anomalies: A detektalt anomaliak listaja.
            metrics: Az aktualis HUD metrikak.
            iteration: Az aktualis iteracio szama (cooldown szamitashoz).

        Returns:
            A vegrehajtott beavatkozasok neveinek listaja.
        """
        interventions: list[str] = []

        # Cooldown ellenorzes: ne avatkozzunk be tul gyakran
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

        1. Entropia koefficienshez szorzo (entropy_boost_factor), MAX CAP-pel
        2. Agresszio bonus aktivalas a reward shaper-ben

        Args:
            metrics: Aktualis HUD metrikak.
        """
        # Entropia noveles (capped — megakadalyozza a vegtelen szorzodast)
        if self._trainer_ref is not None:
            current_ent: float = getattr(
                getattr(self._trainer_ref, "config", None),
                "entropy_coef", 0.01,
            )
            boost: float = self.reward_shaper.config.entropy_boost_factor
            new_ent: float = min(current_ent * boost, self._max_entropy_coef)

            if new_ent >= self._max_entropy_coef:
                logger.warning(
                    "Entropia CAP elerve: %.4f >= %.4f max. "
                    "Tovabbi noveles blokkolva.",
                    new_ent, self._max_entropy_coef,
                )

            self._trainer_ref.update_entropy_coef(new_ent)
            logger.info(
                "Passzivitas intervencio: entropia %.4f -> %.4f "
                "(x%.1f, max=%.4f)",
                current_ent, new_ent, boost, self._max_entropy_coef,
            )

        # Agresszio bonus
        self.reward_shaper.update_aggression_bonus(0.1)
        logger.info("Passzivitas intervencio: agresszio bonus aktivalva: +0.1")

    def _intervene_maniac(self, metrics: dict[str, float]) -> None:
        """All-in Spam / Maniac korrekcios beavatkozas.

        1. Bloff buntetes aktivalas a reward shaper-ben
        2. (Opcionalis) Calling Station botok berotalaisa az ellenfel pool-ba

        Args:
            metrics: Aktualis HUD metrikak.
        """
        # Bloff buntetes aktivalas
        self.reward_shaper.update_penalty_lambda(0.5)
        logger.info("Maniac intervenció: bloff buntetes lambda=0.5 aktiválva")

    def _intervene_stagnation(self) -> None:
        """Stagnacio korrekcios beavatkozas.

        Entropia koefficienshez szorzo a feltaras (exploration) novelesere,
        MAX CAP-pel a vegtelen szorzodas megakadalyozasara.
        """
        if self._trainer_ref is not None:
            current_ent: float = getattr(
                getattr(self._trainer_ref, "config", None),
                "entropy_coef", 0.01,
            )
            boost: float = self.reward_shaper.config.entropy_boost_factor
            new_ent: float = min(current_ent * boost, self._max_entropy_coef)

            if new_ent >= self._max_entropy_coef:
                logger.warning(
                    "Stagnacio: Entropia CAP elerve: %.4f >= %.4f max.",
                    new_ent, self._max_entropy_coef,
                )

            self._trainer_ref.update_entropy_coef(new_ent)
            logger.info(
                "Stagnacio intervencio: entropia %.4f -> %.4f "
                "(x%.1f, max=%.4f)",
                current_ent, new_ent, boost, self._max_entropy_coef,
            )

    def _on_phase_transition(self) -> None:
        """Callback a fazisatmenet utan: reward shaping reset, pool frissites."""
        self.reward_shaper.deactivate_all_shaping()
        logger.info(
            "Fazisatmenet utomunkak: reward shaping resetelve, "
            "uj fazis: %s", self.curriculum.current_phase.name,
        )

    # =========================================================================
    # Config Hot-Reload
    # =========================================================================

    def _check_hot_reload(self) -> None:
        """Ellenorzi a config.yaml modositasi idejet es ujratölti ha valtozott.

        A file watcher periodikusan (hot_reload_interval_sec masodpercenkent)
        osszehasonlitja a fajl mtime-jat a legutobbi olvasassal.
        """
        now: float = time.monotonic()
        if (now - self._last_reload_check) < self.config.hot_reload_interval_sec:
            return
        self._last_reload_check = now

        if not os.path.exists(self.config.config_path):
            return

        current_mtime: float = os.path.getmtime(self.config.config_path)
        if current_mtime <= self._config_last_modified:
            return

        # A fajl modosult -> ujraolvasas
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
        """Alkalmazza a hot-reload-olt konfiguracios valtozasokat.

        Csak a HOT-RELOADABLE parametereket frissiti:
            - ppo.entropy_coefficient
            - ppo.learning_rate
            - reward_shaping.bluff_penalty_lambda
            - reward_shaping.preflop_aggression_bonus

        Args:
            new_cfg: Az uj YAML konfiguracio.
        """
        ppo_cfg = new_cfg.get("ppo", {})
        rs_cfg = new_cfg.get("reward_shaping", {})

        # PPO parameterek
        if self._trainer_ref is not None:
            new_lr: float = ppo_cfg.get("learning_rate", 3e-4)
            new_ent: float = ppo_cfg.get("entropy_coefficient", 0.01)
            self._trainer_ref.update_learning_rate(new_lr)
            self._trainer_ref.update_entropy_coef(new_ent)

        # Reward Shaping parameterek
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
        """Visszaadja az Orchestrator allapotot checkpoint menteshez.

        Returns:
            Szerializalhato allapot szotar.
        """
        return {
            "curriculum_state": self.curriculum.get_state(),
            "reward_shaper_stats": self.reward_shaper.get_stats(),
            "intervention_count": self._intervention_count,
            "last_anomalies": self._last_anomalies,
            "telemetry_total_hands": self.telemetry._total_hands,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Visszaallitja az Orchestrator allapotot.

        Args:
            state: A korabban mentett allapot szotar.
        """
        if "curriculum_state" in state:
            self.curriculum.load_state(state["curriculum_state"])
        self._intervention_count = state.get("intervention_count", 0)
        self._last_anomalies = state.get("last_anomalies", [])
        logger.info("Orchestrator allapot betoltve.")

    # =========================================================================
    # Statisztikak
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """Visszaadja az Orchestrator osszefoglalo statisztikait.

        Returns:
            Dict a legfontosabb mutatokkal.
        """
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
