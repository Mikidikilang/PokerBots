"""
Event-Driven Vegrehajto Ciklus (runner.py).

[FIX C1 - 2025-03-28] Bootstrap Ertek Timing Javitasa:
    A _run_single_iteration() tobbe nem hivja a collector.get_last_bootstrap_value()-t
    a compute_gae() elott. A bootstrap erteket most a collector.collect_rollout()
    atomikusan tarolja a bufferben (buffer.set_last_value()), igy a runner
    onnan olvassa ki: self.buffer.compute_gae(last_value=self.buffer.get_last_bootstrap_value()).
    Ez megszunteti a race conditiont: az ertek garantaltan a HELYES, truncated
    allapothoz tartozik, nem egy lepessessel kesobb szamolt kozelites.

A ciklus felepitese:
    1. Bootstrapping: Kornyezet, halozat, buffer, trainer inicializalasa
    2. Event-Driven Loop (while not shutdown):
        a) Adatgyujtes (collector.collect_rollout)
        b) Gradiens frissites (trainer.train_on_buffer)
        c) Telemetria feldolgozas (orchestrator callback)
        d) Checkpoint mentes (periodikus)
    3. Graceful Shutdown: Utolso mentes, HF feltoltes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.training.buffer import RolloutBuffer, RolloutBufferConfig
from src.training.collector import RolloutCollector
from src.training.trainer import PPOTrainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """A fo vegrehajtasi ciklus konfiguracioja."""

    max_iterations: int = 0
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    buffer_save_interval: int = 500
    max_runtime_hours: float = 11.5
    device: str = "auto"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> RunnerConfig:
        orch = cfg.get("orchestrator", {})
        tel = orch.get("telemetry", {})
        mlops = cfg.get("mlops", {})
        cp = mlops.get("checkpoint", {})
        gs = mlops.get("graceful_shutdown", {})

        return cls(
            log_interval=tel.get("log_interval_iterations", 10),
            eval_interval=tel.get("eval_interval_iterations", 50),
            save_interval=cp.get("save_interval_iterations", 100),
            buffer_save_interval=cp.get("buffer_save_interval_iterations", 500),
            max_runtime_hours=gs.get("max_runtime_hours", 11.5),
            device=cfg.get("runtime", {}).get("device", "auto"),
        )


class TrainingRunner:
    """A teljes RL training pipeline fo vezerlo osztalya.

    [FIX C1] A _run_single_iteration() a compute_gae() hivast a buffer-ben
    tarolt bootstrap ertek alapjan vegzi:
        self.buffer.compute_gae(last_value=self.buffer.get_last_bootstrap_value())
    A korabbi self.collector.get_last_bootstrap_value(self.network) hivast
    eltavolitottuk, mert az egy lepessessel kesob szamolt, versenyfutasi
    allapotot (race condition) okozva episode hatarokon.
    """

    def __init__(
        self,
        config: RunnerConfig,
        env: Any,
        obs_builder: Any,
        network: Any,
        trainer_config: TrainerConfig | None = None,
        buffer_config: RolloutBufferConfig | None = None,
        yaml_config: dict[str, Any] | None = None,
        on_iteration_end: Callable[[int, dict[str, float]], None] | None = None,
        on_eval_step: Callable[[int, Any], dict[str, float] | None] | None = None,
        on_checkpoint: Callable[[int, Any], None] | None = None,
        on_ddp_sync: Callable[[int], None] | None = None,
        checkpoint_dir: str = "checkpoints",
        orchestrator: Any | None = None,
    ) -> None:
        self.config: RunnerConfig = config
        self.network: Any = network
        self.env: Any = env
        self.obs_builder: Any = obs_builder

        self.device: torch.device = self._resolve_device(config.device)

        self.buffer: RolloutBuffer = RolloutBuffer(
            buffer_config or RolloutBufferConfig()
        )
        self.trainer: PPOTrainer = PPOTrainer(
            trainer_config or TrainerConfig(),
            network, self.device,
        )
        self.collector: RolloutCollector = RolloutCollector(
            network=network,
            env=env,
            obs_builder=obs_builder,
            buffer=self.buffer,
            config=yaml_config or {},
            orchestrator=orchestrator,
            device=self.device,
        )

        self._on_iteration_end = on_iteration_end
        self._on_eval_step = on_eval_step
        self._on_checkpoint = on_checkpoint
        self._on_ddp_sync = on_ddp_sync

        self.iteration: int = 0
        self._start_time: float = 0.0
        self._checkpoint_dir: str = checkpoint_dir
        self._should_stop: bool = False
        self._nan_error_occurred: bool = False

        logger.info(
            "TrainingRunner inicializalva: device=%s, max_iter=%d, "
            "save_interval=%d, max_runtime=%.1fh",
            self.device, config.max_iterations,
            config.save_interval, config.max_runtime_hours,
        )

    # =========================================================================
    # Fo Futtatasi Ciklus
    # =========================================================================

    def run(self) -> dict[str, Any]:
        """Elindítja es futtatja a teljes training ciklust."""
        self._start_time = time.monotonic()
        self._should_stop = False

        logger.info(
            "========================================\n"
            "  TRAINING CIKLUS INDUL\n"
            "  Max iterations: %s\n"
            "  Max runtime: %.1f ora\n"
            "  Device: %s\n"
            "========================================",
            self.config.max_iterations or "vegtelen",
            self.config.max_runtime_hours,
            self.device,
        )

        all_stats: list[dict[str, float]] = []

        try:
            while not self._should_stop:
                self.iteration += 1

                if self._check_time_limit():
                    logger.warning(
                        "Idokorlat elerve (%.1f ora). Graceful shutdown...",
                        self.config.max_runtime_hours,
                    )
                    break

                if (self.config.max_iterations > 0
                        and self.iteration > self.config.max_iterations):
                    logger.info(
                        "Max iteracio (%d) elerve. Training vege.",
                        self.config.max_iterations,
                    )
                    break

                iter_stats: dict[str, float] = self._run_single_iteration()
                all_stats.append(iter_stats)

                if self.iteration % self.config.log_interval == 0:
                    self._log_iteration(iter_stats)

                if self.iteration % self.config.eval_interval == 0:
                    if self._on_eval_step is not None:
                        eval_result = self._on_eval_step(self.iteration, self.network)
                        if eval_result:
                            logger.info(
                                "Eval iter #%d: %s", self.iteration, eval_result
                            )

                if self.iteration % self.config.save_interval == 0:
                    self._save_checkpoint()

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt! Graceful shutdown...")
        except FloatingPointError as exc:
            logger.critical(
                "FLOATINGPOINTERROR (Iter #%d): %s — Sulyszennyezodes, "
                "vegso checkpoint mentes kihagyva.",
                self.iteration, exc, exc_info=True,
            )
            self._nan_error_occurred = True
            raise
        except Exception as exc:
            logger.error(
                "KRITIKUS HIBA az iteracioban #%d: %s",
                self.iteration, exc, exc_info=True,
            )
            self._save_checkpoint(emergency=True)
            raise
        finally:
            if not self._nan_error_occurred:
                self._save_checkpoint(final=True)

        elapsed: float = time.monotonic() - self._start_time

        summary: dict[str, Any] = {
            "total_iterations": self.iteration,
            "total_runtime_hours": elapsed / 3600,
            "total_steps": self.collector.get_total_steps(),
            "total_episodes": self.collector.get_total_episodes(),
        }

        logger.info(
            "========================================\n"
            "  TRAINING CIKLUS BEFEJEZVE\n"
            "  Iteraciok: %d\n"
            "  Futasido: %.2f ora\n"
            "  Osszes lepes: %d\n"
            "  Osszes epizod: %d\n"
            "========================================",
            summary["total_iterations"],
            summary["total_runtime_hours"],
            summary["total_steps"],
            summary["total_episodes"],
        )

        return summary

    # =========================================================================
    # Egy Iteracio
    # =========================================================================

    def _run_single_iteration(self) -> dict[str, float]:
        """Vegrehajt egyetlen training iteraciot.

        [FIX C1] A bootstrap ertek szamitasanak javitasa:
            REGI (eltavolitott):
                last_value = self.collector.get_last_bootstrap_value(self.network)
                self.buffer.compute_gae(last_value=last_value)
            ÚJ (helyes):
                # A collector.collect_rollout() mar atomikusan beallitotta
                # a buffer._last_bootstrap_value-t a rollout vegen.
                self.buffer.compute_gae(last_value=self.buffer.get_last_bootstrap_value())
            Ez megszunteti a race conditiont: a buffer garantaltan a HELYES,
            truncated allapothoz tartozo V(s_T)-t tarolja, nem egy lepessessel
            kesobb szamolt kozelitest.

        Returns:
            Dict az iteracio statisztikaival.

        Raises:
            FloatingPointError: NaN/Inf a loss-ban.
            RuntimeError: Dimenzio mismatch vagy matematikai inkonzisztencia.
        """
        # 1. Adatgyujtes — matematikai hibak itt is elofordulhatnak
        try:
            collect_stats = self.collector.collect_rollout(
                n_steps=self.buffer.config.buffer_size
            )
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "MATEMATIKAI HIBA az adatgyujtesben (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # 2. GAE szamitasa — [FIX C1] a buffer-bol olvassuk a bootstrap erteket
        try:
            # A bootstrap erteket a collector.collect_rollout() mar atomikusan
            # tarolta a bufferben. Nincs szukseg ujra-szamolasra.
            bootstrap_value = self.buffer.get_last_bootstrap_value()
            logger.debug(
                "GAE szamitas indul: bootstrap_value=%.6f (iter #%d)",
                bootstrap_value, self.iteration,
            )
            self.buffer.compute_gae(last_value=bootstrap_value)
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "HIBA a GAE szamitasaban (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # 3. PPO Gradiens frissites — NaN/Inf detektalas
        try:
            train_stats: dict[str, float] = self.trainer.train_on_buffer(self.buffer)

            for key in ("policy_loss", "value_loss", "total_loss"):
                loss_val: float = train_stats.get(key, 0.0)
                if loss_val != loss_val or abs(loss_val) == float("inf"):
                    raise FloatingPointError(
                        f"KRITIKUS: {key}={loss_val} (NaN/Inf) detektalva "
                        f"az iteracio #{self.iteration}-ban!"
                    )

        except FloatingPointError:
            raise
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "MATEMATIKAI HIBA a training lepesben (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # 4. Osszesitett statisztikak
        iter_stats: dict[str, float] = {
            "iteration": float(self.iteration),
            **{f"collect/{k}": v for k, v in collect_stats._asdict().items()},
            **{f"train/{k}": v for k, v in train_stats.items()},
            "elapsed_hours": (time.monotonic() - self._start_time) / 3600,
            "bootstrap_value": bootstrap_value,  # diagnosztika
        }

        # 5. Orchestrator callback
        if self._on_iteration_end is not None:
            try:
                self._on_iteration_end(self.iteration, iter_stats)
            except Exception as exc:
                logger.error(
                    "Orchestrator callback hiba (iter #%d): %s — "
                    "A training folytathato, de a curriculum logika "
                    "az aktualis iteracioban kihagyasra kerult.",
                    self.iteration, exc,
                )

        # 6. DDP szinkronizacio
        if self._on_ddp_sync is not None:
            self._on_ddp_sync(self.iteration)

        # 7. Buffer reset
        self.buffer.reset()

        return iter_stats

    # =========================================================================
    # Checkpoint Kezeles
    # =========================================================================

    def _save_checkpoint(
        self, emergency: bool = False, final: bool = False
    ) -> None:
        if self._on_checkpoint is None:
            logger.warning(
                "Nincs on_checkpoint callback konfiguralva — allapot NEM lett mentve (iter #%d)",
                self.iteration,
            )
            return

        save_type = "EMERGENCY" if emergency else ("FINAL" if final else "PERIODIC")
        logger.info("%s checkpoint mentes indul (iter #%d)", save_type, self.iteration)

        try:
            self._on_checkpoint(self.iteration, self.network)
            logger.info("%s checkpoint sikeresen mentve (iter #%d)", save_type, self.iteration)
        except Exception as exc:
            logger.error(
                "Checkpoint callback hiba (iter #%d): %s",
                self.iteration, exc, exc_info=True,
            )

    # =========================================================================
    # Idozites es Leallitas
    # =========================================================================

    def _check_time_limit(self) -> bool:
        elapsed_hours: float = (time.monotonic() - self._start_time) / 3600
        return elapsed_hours >= self.config.max_runtime_hours

    def request_stop(self) -> None:
        self._should_stop = True
        logger.info("Leallitasi keres fogadva. A ciklus a kovetkezo iteracional leall.")

    def get_elapsed_hours(self) -> float:
        if self._start_time == 0:
            return 0.0
        return (time.monotonic() - self._start_time) / 3600

    # =========================================================================
    # Logolas
    # =========================================================================

    def _log_iteration(self, stats: dict[str, float]) -> None:
        logger.info(
            "Iter #%d | rew=%.4f | pl=%.4f vl=%.4f H=%.4f | "
            "kl=%.4f clip=%.1f%% | eps=%d | boot=%.4f | %.2fh",
            self.iteration,
            stats.get("collect/mean_reward", 0.0),
            stats.get("train/policy_loss", 0.0),
            stats.get("train/value_loss", 0.0),
            stats.get("train/entropy_loss", 0.0),
            stats.get("train/approx_kl", 0.0),
            stats.get("train/clip_fraction", 0.0) * 100,
            int(stats.get("collect/n_episodes", 0)),
            stats.get("bootstrap_value", 0.0),  # [C1 diagnosztika]
            stats.get("elapsed_hours", 0.0),
        )

    # =========================================================================
    # Device Feloldas
    # =========================================================================

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Device: CUDA (auto-detected)")
            else:
                device = torch.device("cpu")
                logger.info("Device: CPU (CUDA nem elerheto)")
        else:
            device = torch.device(device_str)
            logger.info("Device: %s (kezi beallitas)", device_str)
        return device
