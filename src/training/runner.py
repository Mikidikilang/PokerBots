"""
Event-Driven Vegrehajto Ciklus (runner.py).

A teljes RL training pipeline fo vezerlo modulja. Nem a klasszikus
model.learn(total_timesteps) megkozelitest alkalmazza, hanem egy
sajat, iterativ, esemenyvezrelt ciklust, amely minden lepesnel
lehetoseget ad az Orchestratornak a beavatkozasra.

A ciklus felepitese:
    1. Bootstrapping: Kornyezet, halozat, buffer, trainer inicializalasa
    2. Event-Driven Loop (while not shutdown):
        a) Adatgyujtes (collector.collect_rollout)
        b) Gradiens frissites (trainer.train_on_buffer)
        c) Telemetria feldolgozas (orchestrator callback)
        d) Checkpoint mentes (periodikus)
    3. Graceful Shutdown: Utolso mentes, HF feltoltes

Hivatkozasok:
    - Specifikacio: runner.py — event-driven loop
    - Curriculum doc: A runner.py integracioja es vegrehajtasi ciklusa
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.training.buffer import RolloutBuffer, RolloutBufferConfig
from src.training.collector import RolloutCollector, CollectorConfig
from src.training.trainer import PPOTrainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """A fo vegrehajtasi ciklus konfiguracioja.

    Attributes:
        max_iterations: Maximalis iteracioszam (0=vegtelen).
        log_interval: Logolasos iteraciok gyakorisaga.
        eval_interval: Kiertekelo iteraciok gyakorisaga.
        save_interval: Checkpoint mentes gyakorisaga (iteraciokent).
        buffer_save_interval: Buffer mentes gyakorisaga.
        max_runtime_hours: Maximalis futasi ido oraban (graceful shutdown).
        device: Szamitasi eszkoz ("cpu", "cuda", "auto").
    """

    max_iterations: int = 0
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    buffer_save_interval: int = 500
    max_runtime_hours: float = 11.5
    device: str = "auto"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> RunnerConfig:
        """YAML config szotarbol peldanyosit.

        Args:
            cfg: Teljes YAML konfiguracio.

        Returns:
            RunnerConfig peldany.
        """
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

    Egyetlen peldany vezrel az egesz betanitasi folyamatot:
    a kornyezet inicializalasatol a graceful shutdown-ig.

    Az Orchestrator es MLOps modulok callback-eken keresztul
    csatlakoznak a ciklushoz (opcionalis).

    Example:
        >>> runner = TrainingRunner(cfg, env, network, ...)
        >>> runner.run()

    Attributes:
        config: Runner konfiguracio.
        network: Az Actor-Critic halozat.
        trainer: A PPO trainer.
        collector: Az adatgyujto.
        buffer: A rollout buffer.
        iteration: Az aktualis iteracio szama.
    """

    def __init__(
        self,
        config: RunnerConfig,
        env: Any,
        obs_builder: Any,
        network: Any,
        trainer_config: TrainerConfig | None = None,
        buffer_config: RolloutBufferConfig | None = None,
        collector_config: CollectorConfig | None = None,
        on_iteration_end: Callable[[int, dict[str, float]], None] | None = None,
        on_eval_step: Callable[[int, Any], dict[str, float] | None] | None = None,
        on_checkpoint: Callable[[int, Any], None] | None = None,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Inicializalja a training runner-t.

        Args:
            config: Runner konfiguracio.
            env: Poker kornyezet peldany.
            obs_builder: ObservationBuilder peldany.
            network: ActorCriticNetwork peldany.
            trainer_config: PPO trainer konfiguracio.
            buffer_config: Rollout buffer konfiguracio.
            collector_config: Collector konfiguracio.
            on_iteration_end: Callback minden iteracio vegen.
                Parameterei: (iteracio_szam, osszesitett_statisztikak).
                Az Orchestrator telemetria feldolgozasa ide csatlakozik.
            on_eval_step: Callback a kiertekelo lepeseknel.
                Az Orchestrator curriculum logikaja ide csatlakozik.
            on_checkpoint: Callback a checkpoint menteseknel.
                Az MLOps hf_sync ide csatlakozik.
            checkpoint_dir: Checkpoint konyvtar eleresi ut.
        """
        self.config: RunnerConfig = config
        self.network: Any = network
        self.env: Any = env
        self.obs_builder: Any = obs_builder

        # Device feloldas
        self.device: torch.device = self._resolve_device(config.device)

        # Komponensek inicializalasa
        self.buffer: RolloutBuffer = RolloutBuffer(
            buffer_config or RolloutBufferConfig()
        )
        self.trainer: PPOTrainer = PPOTrainer(
            trainer_config or TrainerConfig(),
            network, self.device,
        )
        self.collector: RolloutCollector = RolloutCollector(
            collector_config or CollectorConfig(),
            env, obs_builder, network, self.buffer, self.device,
        )

        # Callback-ek
        self._on_iteration_end = on_iteration_end
        self._on_eval_step = on_eval_step
        self._on_checkpoint = on_checkpoint

        # Allapot
        self.iteration: int = 0
        self._start_time: float = 0.0
        self._checkpoint_dir: str = checkpoint_dir
        self._should_stop: bool = False

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
        """Elindítja es futtatja a teljes training ciklust.

        A ciklus addig fut, amig:
            - Eleri a max_iterations-t (ha > 0)
            - A graceful shutdown idozito lejr
            - Kulso leallitasi jelet kap (request_stop())

        Returns:
            Dict az osszesitett training eredmenyekkel.
        """
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

                # --- Idokorlat ellenorzes ---
                if self._check_time_limit():
                    logger.warning(
                        "Idokorlat elerve (%.1f ora). Graceful shutdown...",
                        self.config.max_runtime_hours,
                    )
                    break

                # --- Iteracioszam korlat ---
                if (self.config.max_iterations > 0
                        and self.iteration > self.config.max_iterations):
                    logger.info(
                        "Max iteracio (%d) elerve. Training vege.",
                        self.config.max_iterations,
                    )
                    break

                # --- EGY ITERACIO ---
                iter_stats: dict[str, float] = self._run_single_iteration()
                all_stats.append(iter_stats)

                # --- Periodikus logolas ---
                if self.iteration % self.config.log_interval == 0:
                    self._log_iteration(iter_stats)

                # --- Kiertekeles (Orchestrator callback) ---
                if self.iteration % self.config.eval_interval == 0:
                    if self._on_eval_step is not None:
                        eval_result = self._on_eval_step(self.iteration, self.network)
                        if eval_result:
                            logger.info(
                                "Eval iter #%d: %s", self.iteration, eval_result
                            )

                # --- Checkpoint mentes ---
                if self.iteration % self.config.save_interval == 0:
                    self._save_checkpoint()

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt! Graceful shutdown...")
        except Exception as exc:
            logger.error(
                "KRITIKUS HIBA az iteracioban #%d: %s",
                self.iteration, exc, exc_info=True,
            )
            # Mentsi kiserlet hiba eseten is
            self._save_checkpoint(emergency=True)
            raise
        finally:
            # Graceful shutdown: utolso mentes
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

        A hibakezeles granulaltan kulonvalasztja:
            - Matematikai hibak (NaN, Inf, dimenzio mismatch) → azonnali stop
            - Infrastrukturalis hibak (I/O, callback) → naplozas, folytatas

        Lepesek:
            1. Adatgyujtes (collector)
            2. Gradiens frissites (trainer)
            3. Callback az Orchestratornak

        Returns:
            Dict az iteracio statisztikaival.

        Raises:
            FloatingPointError: NaN/Inf a loss-ban.
            RuntimeError: Dimenzio mismatch vagy matematikai inkonzisztencia.
        """
        # 1. Adatgyujtes — matematikai hibak itt is elofordulhatnak
        try:
            collect_stats: dict[str, float] = self.collector.collect_rollout()
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "MATEMATIKAI HIBA az adatgyujtesben (iter #%d): %s",
                self.iteration, exc,
            )
            raise  # Nem recoverable — dimenzio/allapot hiba

        # 2. PPO Gradiens frissites — NaN/Inf detektalas
        try:
            train_stats: dict[str, float] = self.trainer.train_on_buffer(self.buffer)

            # NaN/Inf ellenorzes a loss ertekekben
            for key in ("policy_loss", "value_loss", "total_loss"):
                loss_val: float = train_stats.get(key, 0.0)
                if loss_val != loss_val or abs(loss_val) == float("inf"):
                    raise FloatingPointError(
                        f"KRITIKUS: {key}={loss_val} (NaN/Inf) detektalva "
                        f"az iteracio #{self.iteration}-ban!"
                    )

        except FloatingPointError:
            raise  # Propagal a fo ciklusba a FaultHandler szamara
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "MATEMATIKAI HIBA a training lepesben (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # 3. Osszesitett statisztikak
        iter_stats: dict[str, float] = {
            "iteration": float(self.iteration),
            **{f"collect/{k}": v for k, v in collect_stats.items()},
            **{f"train/{k}": v for k, v in train_stats.items()},
            "elapsed_hours": (time.monotonic() - self._start_time) / 3600,
        }

        # 4. Orchestrator callback — infrastrukturalis hiba nem allitja le a tanulast
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

        return iter_stats

    # =========================================================================
    # Checkpoint Kezeles
    # =========================================================================

    def _save_checkpoint(
        self, emergency: bool = False, final: bool = False
    ) -> None:
        """Elmenti a halozat es az optimizer allapotat.

        Args:
            emergency: True ha hibakezeles kozbeni mentes.
            final: True ha a training vegen torteno mentes.
        """
        import os
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        prefix: str = "emergency" if emergency else ("final" if final else "checkpoint")
        filepath: str = os.path.join(
            self._checkpoint_dir,
            f"{prefix}_iter_{self.iteration:06d}.pt",
        )

        extra_state: dict[str, Any] = {
            "optimizer_state_dict": self.trainer.get_optimizer_state(),
            "iteration": self.iteration,
            "total_steps": self.collector.get_total_steps(),
            "total_episodes": self.collector.get_total_episodes(),
            "trainer_config": self.trainer.config,
        }

        try:
            self.network.save_checkpoint(filepath, extra_state=extra_state)
            logger.info(
                "%s checkpoint mentve: %s (iter #%d)",
                prefix.upper(), filepath, self.iteration,
            )
        except Exception as exc:
            logger.error("Checkpoint mentes sikertelen: %s", exc)

        # MLOps callback (HF feltoltes)
        if self._on_checkpoint is not None:
            try:
                self._on_checkpoint(self.iteration, self.network)
            except Exception as exc:
                logger.error("Checkpoint callback hiba: %s", exc)

    # =========================================================================
    # Idozites es Leallitas
    # =========================================================================

    def _check_time_limit(self) -> bool:
        """Ellenorzi, hogy a futasido megkozelitette-e az idokorlatot.

        A time.monotonic()-t hasznalja az NTP ugrasok elkerulese erdekeben.

        Returns:
            True ha a futasido meghaladta a max_runtime_hours-t.
        """
        elapsed_hours: float = (time.monotonic() - self._start_time) / 3600
        return elapsed_hours >= self.config.max_runtime_hours

    def request_stop(self) -> None:
        """Kulso leallitasi keres (thread-safe flag beallitas).

        A kovetkezo iteracio elejen a ciklus leall.
        """
        self._should_stop = True
        logger.info("Leallitasi keres fogadva. A ciklus a kovetkezo iteracional leall.")

    def get_elapsed_hours(self) -> float:
        """Visszaadja az eltelt idot oraban.

        Returns:
            Eltelt ido oraban.
        """
        if self._start_time == 0:
            return 0.0
        return (time.monotonic() - self._start_time) / 3600

    # =========================================================================
    # Logolas
    # =========================================================================

    def _log_iteration(self, stats: dict[str, float]) -> None:
        """Reszletes logolast vegez egy iteraciorol.

        Args:
            stats: Az iteracio statisztikai szotarja.
        """
        logger.info(
            "Iter #%d | rew=%.4f | pl=%.4f vl=%.4f H=%.4f | "
            "kl=%.4f clip=%.1f%% | eps=%d | %.2fh",
            self.iteration,
            stats.get("collect/mean_episode_reward", 0.0),
            stats.get("train/policy_loss", 0.0),
            stats.get("train/value_loss", 0.0),
            stats.get("train/entropy_loss", 0.0),
            stats.get("train/approx_kl", 0.0),
            stats.get("train/clip_fraction", 0.0) * 100,
            int(stats.get("collect/episodes_completed", 0)),
            stats.get("elapsed_hours", 0.0),
        )

    # =========================================================================
    # Device Feloldas
    # =========================================================================

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Az "auto" device-ot feloldja CUDA-ra ha elerheto, egyebkent CPU.

        Args:
            device_str: "auto", "cpu", vagy "cuda".

        Returns:
            torch.device peldany.
        """
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
