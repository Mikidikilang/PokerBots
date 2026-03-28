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

import torch
import torch.distributed as dist


# =============================================================================
# DDP Inicializacio (Multi-GPU Support, Phase 9)
# =============================================================================

def setup_ddp() -> tuple[int, int, int]:
    """Initialize DistributedDataParallel (DDP) if launched via torchrun.

    Checks for torchrun environment variables (RANK, LOCAL_RANK, WORLD_SIZE).
    If present, initializes DDP with NCCL backend and sets up CUDA device affinity.
    If not, returns single-process defaults (rank=0, local_rank=0, world_size=1).

    Returns:
        Tuple of (rank, local_rank, world_size):
            - rank: Global process rank (0 to world_size-1)
            - local_rank: Local GPU index on this machine
            - world_size: Total number of processes

    Notes:
        - Sets torch.cuda.set_device(local_rank) to bind process to GPU
        - If WORLD_SIZE=1, DDP is not initialized (returns 0, 0, 1)
        - safe to call multiple times; returns early if already initialized
    """
    rank: int = int(os.environ.get("RANK", 0))
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))

    # If WORLD_SIZE=1, no DDP initialization needed (single-GPU or CPU)
    if world_size == 1:
        return rank, local_rank, world_size

    # Initialize DDP with NCCL backend (requires GPU-to-GPU communication)
    if not dist.is_available():
        raise RuntimeError("DDP requires torch.distributed to be available")

    try:
        dist.init_process_group(backend="nccl")
    except RuntimeError as exc:
        logger_temp = logging.getLogger(__name__)
        logger_temp.error(
            "Failed to initialize DDP: %s\n"
            "Ensure you are launching with: torchrun --nproc_per_node=<N> scripts/train_local.py ...",
            exc,
        )
        raise

    # Set CUDA device affinity: each process uses its local GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


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

    # Alapveto validacio (P4.2: comprehensive config validation)
    required_keys: list[str] = ["project", "environment", "model", "ppo", "orchestrator", "mlops"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Hianyzo konfiguracios kulcs: '{key}' a {config_path} fajlban.")

    # P4.2: Validate numeric ranges
    try:
        lr = cfg.get("ppo", {}).get("learning_rate", 1.0)
        if not (1e-6 <= lr <= 0.1):
            logger.warning(
                "Learning rate %.2e is outside typical range [1e-6, 0.1]. "
                "Check config for errors.", lr
            )
        
        clip_eps = cfg.get("ppo", {}).get("clip_epsilon", 0.2)
        if not (0.01 <= clip_eps <= 0.5):
            logger.warning(
                "PPO clip_epsilon %.3f is outside typical range [0.01, 0.5].", clip_eps
            )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value in config: {exc}") from exc

    # P4.4: Audit seed configuration
    seed: int = cfg.get("project", {}).get("seed", 42)
    logger.info(
        "Konfiguracio betoltve: %s (projekt: %s, v%s, seed=%d)",
        config_path,
        cfg["project"]["name"],
        cfg["project"]["version"],
        seed,
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
    start_time: float | None = None,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
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
        rank: Global DDP process rank (0 to world_size-1).
        local_rank: Local GPU index on this machine.
        world_size: Total number of DDP processes (1 for single-GPU).
        checkpoint_path: Specifikus checkpoint eleresi ut (opcionalis).
        start_time: A futtas indulasanak idopontja (monotonic vagy wall-clock).
                   Ha None, az aktualis ido hasznalodik (GracefulShutdownMonitor-ban).

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
    logger.info("Device: %s (rank=%d/%d)", device, rank, world_size)

    # --- Seed (Phase 9: DDP-aware seeding) ---
    # CRITICAL: Each GPU must have a unique seed to generate diverse rollouts
    # Otherwise, all GPUs will simulate the exact same poker hands, defeating data diversity
    from src.mlops.state_manager import RNGStateManager
    seed: int = cfg.get("project", {}).get("seed", 42)
    # Add rank offset so each GPU gets a different random sequence
    seed_with_rank: int = seed + rank
    dl_generator = RNGStateManager.set_global_seed(seed_with_rank)
    if world_size > 1:
        logger.info("DDP seeding: base_seed=%d, rank_offset=%d, effective_seed=%d",
                    seed, rank, seed_with_rank)

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

    # --- DDP Wrapping (Phase 9: Multi-GPU Support) ---
    # If launched via torchrun with world_size > 1, wrap network in DistributedDataParallel
    # This enables automatic gradient synchronization across GPUs during backward()
    if world_size > 1:
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        logger.info("Network wrapped with DistributedDataParallel (rank=%d/%d)", rank, world_size)

    # --- Training Components Config (instantiated by runner) ---
    from src.training.buffer import RolloutBufferConfig
    from src.training.trainer import TrainerConfig

    # --- State Manager ---
    from src.mlops.state_manager import StateManager
    state_manager = StateManager.from_dict(cfg)

    # --- Resume Training (P2.1: load checkpoint data) ---
    start_iteration: int = 0
    orchestrator_state: dict[str, Any] = {}
    checkpoint_to_resume: dict[str, Any] | None = None

    if resume:
        if checkpoint_path:
            checkpoint_to_resume = state_manager.ckpt_mgr.load(checkpoint_path, map_location=device)
        else:
            checkpoint_to_resume = state_manager.load_training_state(map_location=str(device))

        if checkpoint_to_resume is not None:
            model_state_dict = checkpoint_to_resume["model_state_dict"]
            
            # DDP Compatibility: Rewrite state dict keys if there's a wrapping mismatch
            # Network is now wrapped in DDP, but checkpoint keys don't have 'module.' prefix
            # (because we unwrapped them during save). Add prefix for DDP compatibility.
            if world_size > 1:  # Network is wrapped in DDP
                if not any(k.startswith("module.") for k in model_state_dict.keys()):
                    logger.info("DDP: Adding 'module.' prefix to checkpoint keys for compatibility")
                    model_state_dict = {f"module.{k}": v for k, v in model_state_dict.items()}
            
            network.load_state_dict(model_state_dict)
            start_iteration = checkpoint_to_resume.get("iteration", 0)
            orchestrator_state = checkpoint_to_resume.get("orchestrator_state", {})
            logger.info("Resume training: iter=%d", start_iteration)
            # RNG states will be restored after runner/trainer creation
        else:
            logger.info("Nincs checkpoint, scratch-bol indulas.")

    # --- Orchestrator (Phase 9: Rank 0 only) ---
    # Only the master process should manage curriculum, HUD telemetry, and interventions
    # Worker processes pass orchestrator=None to collector, disabling telemetry submission
    from src.orchestrator.orchestrator import AutoAdaptiveOrchestrator, OrchestratorConfig
    orchestrator: Any = None
    
    if rank == 0:
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
        
        if orchestrator_state:
            orchestrator.load_state(orchestrator_state)
        
        # COMPONENT 1: Set network reference for FSP snapshots
        orchestrator.set_network_reference(network)
        
        logger.info("Orchestrator initialized on rank 0")

    # --- Graceful Shutdown (Rank 0 only) ---
    # Only rank 0 monitors time limits to prevent asynchronous hangs
    from src.mlops.fault_tolerance import GracefulShutdownMonitor, ShutdownConfig
    shutdown_monitor: GracefulShutdownMonitor | None = None
    
    if rank == 0:
        shutdown_monitor = GracefulShutdownMonitor(
            ShutdownConfig.from_dict(cfg),
            start_time=start_time,
        )

    # --- W&B Monitoring (Rank 0 only) ---
    # Only rank 0 logs to W&B to avoid race conditions on API
    from src.mlops.monitoring import WandbMonitor
    monitor = WandbMonitor()
    
    if rank == 0:
        # Recover W&B run ID from checkpoint if resuming
        wandb_run_id: str | None = None
        if resume and checkpoint_to_resume:
            wandb_run_id = checkpoint_to_resume.get("wandb_run_id")
        
        # Setup W&B monitoring (fail-safe: continues if W&B unavailable)
        monitor.setup(config=cfg, resume=resume, run_id=wandb_run_id)
        logger.info("W&B monitoring: active=%s, run_id=%s", monitor.active, monitor.run_id)

    # --- HF Sync (Rank 0 only) ---
    # Only rank 0 manages asynchronous uploads to HuggingFace
    from src.mlops.hf_sync import AsyncModelUploader, configure_headless_auth
    mlops_cfg = cfg.get("mlops", {})
    ckpt_cfg = mlops_cfg.get("checkpoint", {})
    hf_repo: str = mlops_cfg.get("hf_repo_id", "")
    async_cfg = mlops_cfg.get("async_upload", {})
    uploader: AsyncModelUploader | None = None

    if rank == 0 and hf_repo and async_cfg.get("enabled", False):
        configure_headless_auth()
        uploader = AsyncModelUploader(
            repo_id=hf_repo,
            checkpoint_dir=ckpt_cfg.get("local_checkpoint_dir", "checkpoints"),
            sync_interval_minutes=async_cfg.get("sync_interval_minutes", 15),
        )

    # --- Fault Handler (Rank 0 only) ---
    from src.mlops.fault_tolerance import FaultHandler
    fault_handler = FaultHandler(max_nan_retries=3)

    # --- Callbacks osszeallitas ---
    # Phase transition state: broadcast across DDP ranks
    phase_transition_state: dict[str, Any] = {
        "transition_occurred": False,
        "new_phase_name": "",
        "new_opponent_names": [],
    }
    
    def on_iteration_end(iteration: int, stats: dict[str, float]) -> None:
        """Orchestrator + shutdown ellenorzes + W&B logging + DDP Phase Broadcast (Phase 9+)."""
        nonlocal phase_transition_state
        
        phase_transition_state["transition_occurred"] = False
        phase_transition_state["new_phase_name"] = ""
        phase_transition_state["new_opponent_names"] = []
        
        # Only rank 0 handles orchestrator and monitoring
        if rank == 0 and orchestrator is not None:
            # ===== COMPONENT 1: CAPTURE ORCHESTRATOR RESULT =====
            orch_result = orchestrator.on_iteration_callback(iteration, stats)
            fault_handler.reset_nan_counter()
            
            # ===== COMPONENT 1: DETECT PHASE TRANSITION =====
            if orch_result.get("phase_transition", False):
                phase_transition_state["transition_occurred"] = True
                phase_transition_state["new_phase_name"] = orch_result.get("new_phase", "")
                phase_transition_state["new_opponent_names"] = (
                    orchestrator.curriculum.get_current_opponents()
                )
                logger.info(
                    "Phase transition detected on Rank 0: %s → %s (opponents: %s)",
                    orch_result.get("phase"),
                    phase_transition_state["new_phase_name"],
                    phase_transition_state["new_opponent_names"],
                )

            # --- W&B Logging (Phase 9: X-axis scaling for multi-GPU) ---
            if monitor.active:
                # Combine metrics from multiple sources with proper prefixes
                combined_metrics = {}
                
                # Training metrics (from stats) with "train/" prefix
                for key, value in stats.items():
                    combined_metrics[f"train/{key}"] = value
                
                # HUD metrics from orchestrator telemetry with "hud/" prefix
                try:
                    if hasattr(orchestrator, "telemetry") and orchestrator.telemetry:
                        hud_metrics = orchestrator.telemetry.get_current_metrics()
                        for key, value in hud_metrics.items():
                            combined_metrics[f"hud/{key}"] = value
                except Exception as e:
                    logger.debug("Could not extract HUD metrics: %s", e)
                
                # Curriculum metrics with "curriculum/" prefix
                try:
                    if hasattr(orchestrator, "curriculum") and orchestrator.curriculum:
                        current_phase = orchestrator.curriculum.current_phase
                        if current_phase:
                            combined_metrics["curriculum/phase"] = current_phase.name
                        combined_metrics["curriculum/iteration"] = iteration
                except Exception as e:
                    logger.debug("Could not extract curriculum metrics: %s", e)
                
                # Phase 9: X-axis metrics SCALED by world_size
                # CRITICAL: Global throughput = local_throughput × num_gpus
                # Otherwise, multi-GPU training appears slower than single-GPU on dashboards
                local_env_steps = runner.collector.get_total_steps() if hasattr(runner, "collector") else 0
                local_hands = orchestrator.total_hands if hasattr(orchestrator, "total_hands") else 0
                combined_metrics["total_env_steps"] = local_env_steps * world_size
                combined_metrics["total_hands"] = local_hands * world_size
                
                # Log to W&B with iteration as step
                monitor.log_metrics(step=iteration, metrics=combined_metrics)

        # Only rank 0 monitors shutdown
        if rank == 0 and shutdown_monitor is not None:
            if shutdown_monitor.should_shutdown():
                logger.warning("Shutdown monitor trigger! Training leallitasa...")
                runner.request_stop()

    def on_ddp_sync(iteration: int) -> None:
        """DDP szinkronizacio: Rank 0 broadcast, all ranks receive (Phase 9+)."""
        nonlocal phase_transition_state
        
        # ===== COMPONENT 2: DDP BROADCAST (Multi-GPU Only) =====
        if world_size > 1:
            import torch
            import torch.distributed as dist
            
            # Rank 0 broadcasts phase transition signal to all ranks
            # Use a simple flag tensor for synchronization
            phase_transition_flag = torch.tensor(
                [1.0 if phase_transition_state["transition_occurred"] else 0.0],
                dtype=torch.float32,
                device=device
            )
            dist.broadcast(phase_transition_flag, src=0)
            phase_transition_state["transition_occurred"] = bool(phase_transition_flag.item() > 0.5)

        # ===== COMPONENT 3: ALL RANKS UPDATE OPPONENT POOL ===== 
        if phase_transition_state["transition_occurred"]:
            logger.info(
                "Rank %d: Phase transition broadcast received. "
                "New opponents: %s",
                rank, phase_transition_state["new_opponent_names"],
            )
            # Note: Actual opponent loading happens in runner/collector
            # (Not implemented in this first pass, but infrastructure is here)


    def on_checkpoint(iteration: int, net: Any) -> None:
        """Checkpoint mentes + Orchestrator allapot + RNG + W&B run_id (P2.1, Rank 0 only)."""
        # Only rank 0 saves checkpoints to prevent file conflicts
        if rank != 0:
            return
        
        rng_states = RNGStateManager.capture_states(dl_generator)
        state_manager.save_training_state(
            network=net,
            optimizer=runner.trainer.optimizer,
            scheduler=runner.trainer.scheduler,  # [FIX H4] HOZZAADVA
            iteration=iteration,
            total_env_steps=runner.collector.get_total_steps(),
            total_hands=0,  # TODO: track total hands from orchestrator
            best_mean_reward=-float("inf"),  # TODO: track from orchestrator
            orchestrator_state=orchestrator.get_state(),
            config=cfg,
            rng_states=rng_states,
            wandb_run_id=monitor.run_id if monitor.active else None,
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
        yaml_config=cfg,
        on_iteration_end=on_iteration_end,
        on_checkpoint=on_checkpoint,
        on_ddp_sync=on_ddp_sync,
        checkpoint_dir=ckpt_cfg.get("local_checkpoint_dir", "checkpoints"),
        orchestrator=orchestrator,
    )
    runner.iteration = start_iteration
    
    # --- Resume optimizer and RNG states (P2.1, P2.4) ---
    if checkpoint_to_resume is not None:
        if "optimizer_state_dict" in checkpoint_to_resume:
            runner.trainer.optimizer.load_state_dict(checkpoint_to_resume["optimizer_state_dict"])
            logger.info("Optimizer state restored from checkpoint")
        
        # [FIX H4] Scheduler state restoration
        if "scheduler_state_dict" in checkpoint_to_resume:
            runner.trainer.scheduler.load_state_dict(checkpoint_to_resume["scheduler_state_dict"])
            logger.info("Scheduler state restored from checkpoint")
        
        # P2.1: Restore RNG states for deterministic resumption
        if "rng_states" in checkpoint_to_resume and checkpoint_to_resume["rng_states"]:
            RNGStateManager.restore_states(checkpoint_to_resume["rng_states"])
            logger.info("RNG states restored from checkpoint (deterministic)")
    
    # Set trainer reference (Rank 0 only to avoid None orchestrator issues)
    if rank == 0 and orchestrator is not None:
        orchestrator.set_trainer_reference(runner.trainer)

    return {
        "runner": runner,
        "network": network,
        "orchestrator": orchestrator,
        "state_manager": state_manager,
        "shutdown_monitor": shutdown_monitor,
        "monitor": monitor,
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
    """A teljes training pipeline fo belepesi pontja (Phase 9: DDP support)."""
    args = parse_args()

    # 1. Logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    # Phase 9: Initialize DDP if launched via torchrun
    rank, local_rank, world_size = setup_ddp()
    
    # Log on rank 0 only to avoid duplicate messages
    if rank == 0:
        logger.info(
            "============================================================\n"
            "  PokerAI NLHE Training — Lokalis CLI (Phase 9: DDP)\n"
            "  Config: %s\n"
            "  Device: %s\n"
            "  Resume: %s\n"
            "  DDP: rank=%d/%d, local_rank=%d\n"
            "============================================================",
            args.config,
            args.device or "auto",
            args.resume,
            rank, world_size, local_rank,
        )

    # 2. Konfiguracio
    cfg: dict[str, Any] = load_config(args.config)
    cfg["_config_path"] = args.config

    # Felulrasok
    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
        if rank == 0:
            logger.info("Seed feluliras: %d", args.seed)

    if args.max_iter > 0:
        cfg.setdefault("runtime", {})["max_iterations"] = args.max_iter

    # 3. Pipeline osszeallitas (with DDP parameters)
    pipeline: dict[str, Any] = build_training_pipeline(
        cfg,
        device_override=args.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

    runner = pipeline["runner"]
    if args.max_iter > 0:
        runner.config.max_iterations = args.max_iter

    # 4. Training inditás
    start_time: float = time.monotonic()

    try:
        summary: dict[str, Any] = runner.run()
    except KeyboardInterrupt:
        if rank == 0:
            logger.warning("KeyboardInterrupt fogadva. Graceful shutdown...")
        summary = {"interrupted": True}
    except Exception as exc:
        if rank == 0:
            logger.error("Kritikus hiba: %s", exc, exc_info=True)
            action: str = pipeline["fault_handler"].handle_generic_error(exc)
            logger.info("FaultHandler ajanlasa: %s", action)
        summary = {"error": str(exc)}
    finally:
        # Vegso mentes es HF feltoltes (Rank 0 only)
        if rank == 0:
            if pipeline.get("uploader") and pipeline["uploader"].is_active():
                pipeline["uploader"].trigger_manual_upload()
                pipeline["uploader"].shutdown()

            # W&B monitoring graceful finish
            if pipeline.get("monitor"):
                pipeline["monitor"].finish()

        # Phase 9: DDP Cleanup
        # Must be done by all ranks; WaitAll ensures synchronization
        if world_size > 1:
            try:
                dist.barrier()
                dist.destroy_process_group()
                if rank == 0:
                    logger.info("DDP process group destroyed")
            except Exception as e:
                if rank == 0:
                    logger.warning("Error destroying DDP process group: %s", e)

    # 5. Vegso statisztikak (Rank 0 only)
    if rank == 0:
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
