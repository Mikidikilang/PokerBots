#!/usr/bin/env python3
"""
Lokalis Training CLI Belepesi Pont (train_local.py).

Ez a szkript a teljes RL training pipeline-t inicializalja es futtatja
lokalis geepen vagy dedikalt szerveren. A Kaggle notebook ekvivalense,
de CLI argumentumokkal vezerelve.

Hasznalat:
    python -m scripts.train_local --config config.yaml
    python -m scripts.train_local --config config.yaml --device cuda --max-iter 5000
    python -m scripts.train_local --config config.yaml --resume

A szkript a kovetkezo lepeseket hajtja vegre:
    1. Konfiguracio betoltes (YAML)
    2. Logging beallitas
    3. Globalis seed beallitas (reprodukalhatosag)
    4. Kornyezet inicializalas (RLCard/PettingZoo)
    5. Halozat es optimizer letrehozas
    6. Checkpoint resume (ha van korabbi allapot)
    7. Orchestrator es MLOps inicializalas
    8. Training ciklus inditas (runner.py)
    9. Graceful shutdown es vegso mentes

Hivatkozasok:
    - Specifikacio: scripts/train_local.py — CLI belepesi pont
    - Architektura: Funkcionalis Szerzodesek es Interakciok
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Logging Beallitas
# =============================================================================

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Konfigurálja a Python logging rendszert.

    A naplok ket helyre iranyulnak:
        1. Konzol (stdout) — szines, rovid formatum
        2. Fajl (log_dir/training.log) — reszletes, teljes formatum

    Args:
        log_level: A logging szint neve (DEBUG, INFO, WARNING, ERROR).
        log_dir: A naplofajlok konyvtara.
    """
    os.makedirs(log_dir, exist_ok=True)

    level: int = getattr(logging, log_level.upper(), logging.INFO)

    # Gyoker logger
    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(level)

    # Korabbi handlerek torlese (ujrainditas eseten)
    root_logger.handlers.clear()

    # Konzol handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # Fajl handler
    log_file: str = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Fajlba mindig DEBUG
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-40s | %(funcName)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging inicializalva: level=%s, log_file=%s", log_level, log_file)


# =============================================================================
# Konfiguracio Betoltes
# =============================================================================

def load_config(config_path: str) -> dict[str, Any]:
    """Betolti es validalja a YAML konfiguracios fajlt.

    Args:
        config_path: A config.yaml fajl eleresi utja.

    Returns:
        A konfiguracios szotar.

    Raises:
        FileNotFoundError: Ha a fajl nem talalhato.
        yaml.YAMLError: Ha a YAML formatum ervenytelen.
    """
    logger = logging.getLogger(__name__)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Konfiguracios fajl nem talalhato: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Alapveto validacio
    required_keys: list[str] = ["project", "environment", "model", "ppo", "orchestrator", "mlops"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Hianyzo konfiguracios kulcs: '{key}' a {config_path} fajlban.")

    logger.info(
        "Konfiguracio betoltve: %s (projekt: %s, v%s)",
        config_path,
        cfg["project"]["name"],
        cfg["project"]["version"],
    )

    return cfg


# =============================================================================
# Kornyezet Inicializalas
# =============================================================================

def create_environment(cfg: dict[str, Any]) -> Any:
    """Letrehozza a poker jatekkkornyezetet a konfig alapjan.

    Args:
        cfg: Teljes YAML konfiguracio.

    Returns:
        A kornyezet peldany (PokerEnvironment protocol).
    """
    from src.env.wrappers import make_env
    return make_env(cfg)


# =============================================================================
# Fo Pipeline Osszeszereleo
# =============================================================================

def build_training_pipeline(
    cfg: dict[str, Any],
    device_override: str | None = None,
    resume: bool = False,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Osszeallitja a teljes training pipeline-t a konfig alapjan.

    Ez a fuggveny letrehozza es osszekapcsolja az osszes komponenst:
        - ObservationBuilder (env)
        - PokerActorCritic (model)
        - RolloutBuffer, PPOTrainer, Collector (training)
        - Orchestrator, Telemetry, Curriculum (orchestrator)
        - CheckpointManager, GracefulShutdownMonitor (mlops)
        - TrainingRunner (training)

    Args:
        cfg: Teljes YAML konfiguracio.
        device_override: Device feluliras ("cpu", "cuda").
        resume: True ha korabbi checkpoint-bol kell folytatni.
        checkpoint_path: Specifikus checkpoint eleresi ut (opcionalis).

    Returns:
        Dict a pipeline komponenseivel es a TrainingRunner-rel.
    """
    logger = logging.getLogger(__name__)
    import torch

    # --- Device ---
    device_str: str = device_override or cfg.get("runtime", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)

    # --- Seed ---
    from src.mlops.state_manager import RNGStateManager
    seed: int = cfg.get("project", {}).get("seed", 42)
    dl_generator = RNGStateManager.set_global_seed(seed)

    # --- Environment ---
    env = create_environment(cfg)

    # --- Observation Builder ---
    from src.env.features import ObservationBuilder, ObservationConfig
    num_players: int = cfg["environment"]["num_players"]
    obs_config = ObservationConfig(num_players=num_players)
    obs_builder = ObservationBuilder(obs_config)
    logger.info("ObservationBuilder: dim=%d", obs_builder.get_observation_dim())

    # --- Network ---
    from src.model.networks import PokerActorCritic, NetworkConfig
    net_config = NetworkConfig.from_dict(cfg, num_players=num_players)
    network = PokerActorCritic(net_config).to(device)
    param_counts = network.get_param_count()
    logger.info("PokerActorCritic: %s params", f"{param_counts['total']:,}")

    # --- Training Components ---
    from src.training.buffer import RolloutBuffer, RolloutBufferConfig
    from src.training.collector import RolloutCollector, CollectorConfig
    from src.training.trainer import PPOTrainer, TrainerConfig

    buffer = RolloutBuffer(RolloutBufferConfig.from_dict(cfg))
    trainer = PPOTrainer(TrainerConfig.from_dict(cfg), network, device)
    collector_cfg = CollectorConfig.from_dict(cfg)
    collector = RolloutCollector(collector_cfg, env, obs_builder, network, buffer, device)

    # --- State Manager ---
    from src.mlops.state_manager import StateManager
    state_manager = StateManager.from_dict(cfg)

    # --- Resume Training ---
    start_iteration: int = 0
    orchestrator_state: dict[str, Any] = {}

    if resume:
        if checkpoint_path:
            result = state_manager.ckpt_mgr.load(checkpoint_path, map_location=device)
        else:
            result = state_manager.load_training_state(map_location=str(device))

        if result is not None:
            network.load_state_dict(result["model_state_dict"])
            trainer.optimizer.load_state_dict(result["optimizer_state_dict"])
            start_iteration = result.get("iteration", 0)
            orchestrator_state = result.get("orchestrator_state", {})
            logger.info("Resume training: iter=%d", start_iteration)
        else:
            logger.info("Nincs checkpoint, scratch-bol indulas.")

    # --- Orchestrator ---
    from src.orchestrator.orchestrator import AutoAdaptiveOrchestrator, OrchestratorConfig
    AutoAdaptiveOrchestrator.reset_instance()

    orch_config = OrchestratorConfig(
        config_path=cfg.get("_config_path", "config.yaml"),
        num_players=num_players,
        telemetry_window=cfg.get("orchestrator", {}).get("telemetry", {}).get(
            "sliding_window_hands", 100_000
        ),
        eval_interval=cfg.get("orchestrator", {}).get("telemetry", {}).get(
            "eval_interval_iterations", 50
        ),
        enable_hot_reload=True,
    )
    orchestrator = AutoAdaptiveOrchestrator.get_instance(orch_config, cfg)
    orchestrator.set_trainer_reference(trainer)

    if orchestrator_state:
        orchestrator.load_state(orchestrator_state)

    # --- Graceful Shutdown ---
    from src.mlops.fault_tolerance import GracefulShutdownMonitor, ShutdownConfig
    shutdown_monitor = GracefulShutdownMonitor(ShutdownConfig.from_dict(cfg))

    # --- HF Sync (opcionalis) ---
    from src.mlops.hf_sync import AsyncModelUploader, configure_headless_auth
    mlops_cfg = cfg.get("mlops", {})
    ckpt_cfg = mlops_cfg.get("checkpoint", {})
    hf_repo: str = mlops_cfg.get("hf_repo_id", "")
    async_cfg = mlops_cfg.get("async_upload", {})
    uploader: AsyncModelUploader | None = None

    if hf_repo and async_cfg.get("enabled", False):
        configure_headless_auth()
        uploader = AsyncModelUploader(
            repo_id=hf_repo,
            checkpoint_dir=ckpt_cfg.get("local_checkpoint_dir", "checkpoints"),
            sync_interval_minutes=async_cfg.get("sync_interval_minutes", 15),
        )

    # --- Fault Handler ---
    from src.mlops.fault_tolerance import FaultHandler
    fault_handler = FaultHandler(max_nan_retries=3)

    # --- Callbacks osszeallitas ---
    def on_iteration_end(iteration: int, stats: dict[str, float]) -> None:
        """Orchestrator + shutdown ellenorzes minden iteracio vegen."""
        orchestrator.on_iteration_callback(iteration, stats)
        fault_handler.reset_nan_counter()

        if shutdown_monitor.should_shutdown():
            logger.warning("Shutdown monitor trigger! Training leallitasa...")
            runner.request_stop()

    def on_checkpoint(iteration: int, net: Any) -> None:
        """Checkpoint mentes + Orchestrator allapot + RNG."""
        rng_states = RNGStateManager.capture_states(dl_generator)
        state_manager.save_training_state(
            network=net,
            optimizer=trainer.optimizer,
            iteration=iteration,
            total_env_steps=collector.get_total_steps(),
            total_hands=0,  # TODO: track total hands from orchestrator
            best_mean_reward=-float("inf"),  # TODO: track from orchestrator
            orchestrator_state=orchestrator.get_state(),
            config=cfg,
            is_best=False,  # TODO: implement best selection logic
        )

    # --- Runner ---
    from src.training.runner import TrainingRunner, RunnerConfig
    runner_config = RunnerConfig.from_dict(cfg)
    if device_override:
        runner_config.device = device_override

    runner = TrainingRunner(
        config=runner_config,
        env=env,
        obs_builder=obs_builder,
        network=network,
        trainer_config=TrainerConfig.from_dict(cfg),
        buffer_config=RolloutBufferConfig.from_dict(cfg),
        collector_config=collector_cfg,
        on_iteration_end=on_iteration_end,
        on_checkpoint=on_checkpoint,
        checkpoint_dir=ckpt_cfg.get("local_checkpoint_dir", "checkpoints"),
    )
    runner.iteration = start_iteration

    return {
        "runner": runner,
        "network": network,
        "trainer": trainer,
        "orchestrator": orchestrator,
        "state_manager": state_manager,
        "shutdown_monitor": shutdown_monitor,
        "uploader": uploader,
        "fault_handler": fault_handler,
        "config": cfg,
    }


# =============================================================================
# CLI Argumentumok
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parszol a CLI argumentumokat.

    Returns:
        Az argumentumok Namespace objektuma.
    """
    parser = argparse.ArgumentParser(
        description="PokerAI NLHE Training — Lokalis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="A YAML konfiguracios fajl eleresi utja.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "auto"],
        help="Szamitasi eszkoz feluliras.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=0,
        help="Maximalis iteracioszam (0=vegtelen, a config alapjan).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Folytatas a legutolso checkpoint-bol.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Specifikus checkpoint fajl eleresi utja a resume-hoz.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Naplozasi szint.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Globalis seed feluliras.",
    )

    return parser.parse_args()


# =============================================================================
# Fo Belepesi Pont
# =============================================================================

def main() -> None:
    """A teljes training pipeline fo belepesi pontja."""
    args = parse_args()

    # 1. Logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(
        "============================================================\n"
        "  PokerAI NLHE Training — Lokalis CLI\n"
        "  Config: %s\n"
        "  Device: %s\n"
        "  Resume: %s\n"
        "============================================================",
        args.config,
        args.device or "auto",
        args.resume,
    )

    # 2. Konfiguracio
    cfg: dict[str, Any] = load_config(args.config)
    cfg["_config_path"] = args.config

    # Felulrasok
    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
        logger.info("Seed feluliras: %d", args.seed)

    if args.max_iter > 0:
        cfg.setdefault("runtime", {})["max_iterations"] = args.max_iter

    # 3. Pipeline osszeallitas
    pipeline: dict[str, Any] = build_training_pipeline(
        cfg,
        device_override=args.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
    )

    runner = pipeline["runner"]
    if args.max_iter > 0:
        runner.config.max_iterations = args.max_iter

    # 4. Training inditás
    start_time: float = time.monotonic()

    try:
        summary: dict[str, Any] = runner.run()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt fogadva. Graceful shutdown...")
        summary = {"interrupted": True}
    except Exception as exc:
        logger.error("Kritikus hiba: %s", exc, exc_info=True)
        action: str = pipeline["fault_handler"].handle_generic_error(exc)
        logger.info("FaultHandler ajanlasa: %s", action)
        summary = {"error": str(exc)}
    finally:
        # Vegso mentes es HF feltoltes
        if pipeline.get("uploader") and pipeline["uploader"].is_active():
            pipeline["uploader"].trigger_manual_upload()
            pipeline["uploader"].shutdown()

    # 5. Vegso statisztikak
    elapsed: float = time.monotonic() - start_time

    logger.info(
        "============================================================\n"
        "  TRAINING BEFEJEZVE\n"
        "  Futasido: %.2f ora\n"
        "  Eredmeny: %s\n"
        "============================================================",
        elapsed / 3600,
        summary,
    )


if __name__ == "__main__":
    main()
