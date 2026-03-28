#!/usr/bin/env python3
"""
Lokalis Training CLI Belepesi Pont (train_local.py).

[FIX C-2 — 2025-03-28] Shutdown signal broadcast to all DDP ranks.

    ROOT CAUSE OF HANG:
    GracefulShutdownMonitor called runner.request_stop() on rank 0 only.
    Rank 0 exited its while-loop. Rank 1 entered the next iteration and
    called loss.backward() which blocks in DDP's all_reduce hook waiting
    for rank 0's gradient contribution — permanent hang, no error.

    THE FIX:
    on_ddp_sync() now broadcasts a shutdown flag from rank 0 to all ranks
    using dist.broadcast(). This is a synchronous collective that all ranks
    already call every iteration — the correct and only synchronization point.
    All ranks exit the training loop together on the same iteration.

[FIX H-1 — 2025-03-28] _session_start captured BEFORE build_training_pipeline().

    GracefulShutdownMonitor was starting its clock at __init__ time, which
    happens INSIDE build_training_pipeline() after DDP init, HF download,
    and model construction. Those steps can take minutes, silently eating
    into the 11.5h budget. Now _session_start is captured in main() before
    any I/O and passed through as start_time=.
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
# DDP Inicializacio
# =============================================================================

def setup_ddp() -> tuple[int, int, int]:
    """Initialize DistributedDataParallel if launched via torchrun."""
    rank:       int = int(os.environ.get("RANK",       0))
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        return rank, local_rank, world_size

    if not dist.is_available():
        raise RuntimeError("DDP requires torch.distributed to be available")

    try:
        dist.init_process_group(backend="nccl")
    except RuntimeError as exc:
        logger_temp = logging.getLogger(__name__)
        logger_temp.error(
            "Failed to initialize DDP: %s\n"
            "Launch with: torchrun --nproc_per_node=<N> scripts/train_local.py ...",
            exc,
        )
        raise

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    level: int = getattr(logging, log_level.upper(), logging.INFO)

    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.addHandler(console_handler)

    log_file: str = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-40s | %(funcName)-25s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized: level=%s, log_file=%s", log_level, log_file)


# =============================================================================
# Config
# =============================================================================

def load_config(config_path: str) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    required_keys: list[str] = [
        "project", "environment", "model", "ppo", "orchestrator", "mlops"
    ]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing config key: '{key}' in {config_path}")

    try:
        lr = cfg.get("ppo", {}).get("learning_rate", 1.0)
        if not (1e-6 <= lr <= 0.1):
            logger.warning("Learning rate %.2e outside typical range.", lr)
        clip_eps = cfg.get("ppo", {}).get("clip_epsilon", 0.2)
        if not (0.01 <= clip_eps <= 0.5):
            logger.warning("PPO clip_epsilon %.3f outside typical range.", clip_eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value in config: {exc}") from exc

    seed: int = cfg.get("project", {}).get("seed", 42)
    logger.info(
        "Config loaded: %s (project: %s, v%s, seed=%d)",
        config_path,
        cfg["project"]["name"],
        cfg["project"]["version"],
        seed,
    )
    return cfg


# =============================================================================
# Environment
# =============================================================================

def create_environment(cfg: dict[str, Any]) -> Any:
    from src.env.wrappers import make_env
    return make_env(cfg)


# =============================================================================
# Pipeline Builder
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
    """Assemble the full training pipeline.

    Args:
        start_time: Monotonic clock captured BEFORE this call in main() or
                    in Cell 1-A of the Kaggle notebook. Passed directly to
                    GracefulShutdownMonitor so it measures true session
                    duration (including DDP init and HF download time).
                    [FIX H-1]
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

    # --- Seed (DDP-aware: each rank gets a unique seed for rollout diversity) ---
    from src.mlops.state_manager import RNGStateManager
    seed: int = cfg.get("project", {}).get("seed", 42)
    seed_with_rank: int = seed + rank
    dl_generator = RNGStateManager.set_global_seed(seed_with_rank)
    if world_size > 1:
        logger.info(
            "DDP seeding: base=%d, rank_offset=%d, effective=%d",
            seed, rank, seed_with_rank,
        )

    # --- Environment ---
    env = create_environment(cfg)

    # --- ObservationBuilder ---
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

    # --- DDP Wrapping ---
    if world_size > 1:
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        logger.info(
            "Network wrapped with DistributedDataParallel (rank=%d/%d)",
            rank, world_size,
        )

    # --- Config objects ---
    from src.training.buffer import RolloutBufferConfig
    from src.training.trainer import TrainerConfig

    # --- State Manager ---
    from src.mlops.state_manager import StateManager
    state_manager = StateManager.from_dict(cfg)

    # --- Resume ---
    start_iteration: int = 0
    orchestrator_state: dict[str, Any] = {}
    checkpoint_to_resume: dict[str, Any] | None = None

    if resume:
        if checkpoint_path:
            checkpoint_to_resume = state_manager.ckpt_mgr.load(
                checkpoint_path, map_location=device
            )
        else:
            checkpoint_to_resume = state_manager.load_training_state(
                map_location=str(device)
            )

        if checkpoint_to_resume is not None:
            model_state_dict = checkpoint_to_resume["model_state_dict"]
            if world_size > 1:
                if not any(k.startswith("module.") for k in model_state_dict.keys()):
                    logger.info("DDP: adding 'module.' prefix to checkpoint keys.")
                    model_state_dict = {
                        f"module.{k}": v for k, v in model_state_dict.items()
                    }
            network.load_state_dict(model_state_dict)
            start_iteration  = checkpoint_to_resume.get("iteration", 0)
            orchestrator_state = checkpoint_to_resume.get("orchestrator_state", {})
            logger.info("Resume: iter=%d", start_iteration)
        else:
            logger.info("No checkpoint found — cold start.")

    # --- Orchestrator (rank 0 only) ---
    from src.orchestrator.orchestrator import (
        AutoAdaptiveOrchestrator, OrchestratorConfig,
    )
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
        orchestrator.set_network_reference(network)
        orchestrator.set_ddp_world_size(world_size)
        logger.info("Orchestrator initialized on rank 0")

    # --- Graceful Shutdown (rank 0 only) ---
    from src.mlops.fault_tolerance import GracefulShutdownMonitor, ShutdownConfig
    shutdown_monitor: GracefulShutdownMonitor | None = None

    if rank == 0:
        # [FIX H-1] start_time captured in main() BEFORE this call.
        # Without this the monitor's clock started AFTER DDP init + HF download,
        # silently losing minutes of the 11.5h Kaggle budget.
        shutdown_monitor = GracefulShutdownMonitor(
            ShutdownConfig.from_dict(cfg),
            start_time=start_time,  # None → monitor captures time.monotonic() now
        )
        if start_time is not None:
            logger.info(
                "GracefulShutdownMonitor: clock anchored to session start "
                "(start_time=%.4f monotonic) [FIX H-1]",
                start_time,
            )

    # --- W&B (rank 0 only) ---
    from src.mlops.monitoring import WandbMonitor
    monitor = WandbMonitor()
    if rank == 0:
        wandb_run_id: str | None = None
        if resume and checkpoint_to_resume:
            wandb_run_id = checkpoint_to_resume.get("wandb_run_id")
        monitor.setup(config=cfg, resume=resume, run_id=wandb_run_id)
        logger.info(
            "W&B monitoring: active=%s, run_id=%s", monitor.active, monitor.run_id
        )

    # --- HF Sync (rank 0 only) ---
    from src.mlops.hf_sync import AsyncModelUploader, configure_headless_auth
    mlops_cfg = cfg.get("mlops", {})
    ckpt_cfg  = mlops_cfg.get("checkpoint", {})
    hf_repo:  str = mlops_cfg.get("hf_repo_id", "")
    async_cfg = mlops_cfg.get("async_upload", {})
    uploader: AsyncModelUploader | None = None

    if rank == 0 and hf_repo and async_cfg.get("enabled", False):
        configure_headless_auth()
        uploader = AsyncModelUploader(
            repo_id=hf_repo,
            checkpoint_dir=ckpt_cfg.get("local_checkpoint_dir", "checkpoints"),
            sync_interval_minutes=async_cfg.get("sync_interval_minutes", 15),
        )

    # --- Fault Handler ---
    from src.mlops.fault_tolerance import FaultHandler
    fault_handler = FaultHandler(max_nan_retries=3)

    # =========================================================================
    # Callbacks
    # =========================================================================

    # Shared mutable state for on_ddp_sync phase broadcast
    phase_transition_state: dict[str, Any] = {
        "transition_occurred": False,
        "new_phase_name":      "",
        "new_opponent_names":  [],
    }

    def on_iteration_end(iteration: int, stats: dict[str, float]) -> None:
        """Orchestrator callback — runs on rank 0 only."""
        nonlocal phase_transition_state

        phase_transition_state["transition_occurred"] = False
        phase_transition_state["new_phase_name"]      = ""
        phase_transition_state["new_opponent_names"]  = []

        if rank == 0 and orchestrator is not None:
            orch_result = orchestrator.on_iteration_callback(iteration, stats)
            fault_handler.reset_nan_counter()

            if orch_result.get("phase_transition", False):
                phase_transition_state["transition_occurred"] = True
                phase_transition_state["new_phase_name"] = orch_result.get("new_phase", "")
                phase_transition_state["new_opponent_names"] = (
                    orchestrator.curriculum.get_current_opponents()
                )
                logger.info(
                    "Phase transition on rank 0: %s → %s",
                    orch_result.get("phase"),
                    phase_transition_state["new_phase_name"],
                )

            if monitor.active:
                combined_metrics: dict[str, Any] = {}
                for key, value in stats.items():
                    combined_metrics[f"train/{key}"] = value
                try:
                    if hasattr(orchestrator, "telemetry") and orchestrator.telemetry:
                        hud = orchestrator.telemetry.get_current_metrics()
                        for key, value in hud.items():
                            combined_metrics[f"hud/{key}"] = value
                except Exception as e:
                    logger.debug("HUD metrics extraction failed: %s", e)
                local_steps = runner.collector.get_total_steps() if hasattr(runner, "collector") else 0
                combined_metrics["total_env_steps"] = local_steps * world_size
                monitor.log_metrics(step=iteration, metrics=combined_metrics)

        # Shutdown check (rank 0 only)
        if rank == 0 and shutdown_monitor is not None:
            if shutdown_monitor.should_shutdown():
                logger.warning(
                    "Shutdown monitor triggered at iter %d! "
                    "Setting stop flag on rank 0.",
                    iteration,
                )
                runner.request_stop()

    def on_ddp_sync(iteration: int) -> None:
        """Cross-rank synchronization — called on ALL ranks every iteration.

        This is the ONLY place where DDP collectives are called from within
        the training loop callbacks. Two broadcasts per iteration:

        1. Phase transition flag: rank 0 tells all ranks about curriculum change.
        2. [FIX C-2] Shutdown flag: rank 0 tells all ranks to exit the loop.

        Without fix C-2:
            rank 0 calls runner.request_stop() (sets _should_stop=True)
            rank 0 exits while-loop
            rank 1 enters next iteration → loss.backward() → blocks in all_reduce
            → permanent hang, no error message, session killed at 12h timeout.

        With fix C-2:
            all ranks synchronize the shutdown flag here (before backward())
            all ranks set _should_stop=True and exit together on the same iter
        """
        nonlocal phase_transition_state

        if world_size <= 1:
            return  # Single-GPU: no collectives needed

        # --- Broadcast 1: Phase transition ---
        phase_flag = torch.tensor(
            [1.0 if phase_transition_state["transition_occurred"] else 0.0],
            dtype=torch.float32,
            device=device,
        )
        dist.broadcast(phase_flag, src=0)
        phase_transition_state["transition_occurred"] = bool(phase_flag.item() > 0.5)

        # --- Broadcast 2: Shutdown flag [FIX C-2] ---
        # rank 0 writes its _should_stop value; all other ranks receive it.
        # If rank 0 has set _should_stop=True (via GracefulShutdownMonitor or
        # KeyboardInterrupt), all ranks learn this here and exit together.
        should_stop_val = 1.0 if runner._should_stop else 0.0
        stop_flag = torch.tensor(
            [should_stop_val], dtype=torch.float32, device=device
        )
        dist.broadcast(stop_flag, src=0)

        if bool(stop_flag.item() > 0.5) and not runner._should_stop:
            logger.info(
                "[Rank %d] Shutdown broadcast received — "
                "requesting stop to exit training loop cleanly.",
                rank,
            )
            runner.request_stop()

    def on_checkpoint(iteration: int, net: Any) -> None:
        """Checkpoint save (rank 0 only — prevents file conflicts)."""
        if rank != 0:
            return

        rng_states = RNGStateManager.capture_states(dl_generator)
        state_manager.save_training_state(
            network=net,
            optimizer=runner.trainer.optimizer,
            scheduler=runner.trainer.scheduler,
            iteration=iteration,
            total_env_steps=runner.collector.get_total_steps(),
            total_hands=0,
            best_mean_reward=-float("inf"),
            orchestrator_state=orchestrator.get_state() if orchestrator else {},
            config=cfg,
            rng_states=rng_states,
            wandb_run_id=monitor.run_id if monitor.active else None,
            is_best=False,
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

    # --- Restore optimizer / scheduler / RNG from checkpoint ---
    if checkpoint_to_resume is not None:
        if "optimizer_state_dict" in checkpoint_to_resume:
            runner.trainer.optimizer.load_state_dict(
                checkpoint_to_resume["optimizer_state_dict"]
            )
            logger.info("Optimizer state restored from checkpoint")

        if (
            "scheduler_state_dict" in checkpoint_to_resume
            and checkpoint_to_resume["scheduler_state_dict"] is not None
            and runner.trainer.scheduler is not None
        ):
            runner.trainer.scheduler.load_state_dict(
                checkpoint_to_resume["scheduler_state_dict"]
            )
            logger.info("LR scheduler state restored from checkpoint [FIX H-4]")

        if "rng_states" in checkpoint_to_resume and checkpoint_to_resume["rng_states"]:
            RNGStateManager.restore_states(checkpoint_to_resume["rng_states"])
            logger.info("RNG states restored (deterministic resume)")

    if rank == 0 and orchestrator is not None:
        orchestrator.set_trainer_reference(runner.trainer)

    return {
        "runner":          runner,
        "network":         network,
        "orchestrator":    orchestrator,
        "state_manager":   state_manager,
        "shutdown_monitor": shutdown_monitor,
        "monitor":         monitor,
        "uploader":        uploader,
        "fault_handler":   fault_handler,
        "config":          cfg,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PokerAI NLHE Training — Local CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     type=str,  default="config.yaml")
    parser.add_argument("--device",     type=str,  default=None,
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max-iter",   type=int,  default=0)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--checkpoint", type=str,  default=None)
    parser.add_argument("--log-level",  type=str,  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed",       type=int,  default=None)
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main entry point. [FIX H-1] Session clock captured before any I/O."""
    args = parse_args()
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    # [FIX H-1] Capture monotonic clock BEFORE setup_ddp() and BEFORE
    # build_training_pipeline(). DDP init (NCCL rendezvous) and HF checkpoint
    # download can each take minutes. Without this fix, the GracefulShutdown-
    # Monitor's clock started after those operations, silently losing that
    # time from the 11.5h Kaggle budget.
    _session_start: float = time.monotonic()
    logger.info(
        "Session clock started: %.4f (monotonic) [FIX H-1]", _session_start
    )

    # DDP init
    rank, local_rank, world_size = setup_ddp()

    if rank == 0:
        logger.info(
            "=" * 60 + "\n"
            "  PokerAI NLHE Training\n"
            "  Config: %s | Device: %s | Resume: %s\n"
            "  DDP: rank=%d/%d | local_rank=%d\n"
            "=" * 60,
            args.config, args.device or "auto", args.resume,
            rank, world_size, local_rank,
        )

    cfg: dict[str, Any] = load_config(args.config)
    cfg["_config_path"] = args.config

    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
        if rank == 0:
            logger.info("Seed override: %d", args.seed)

    if args.max_iter > 0:
        cfg.setdefault("runtime", {})["max_iterations"] = args.max_iter

    # Pass _session_start so GracefulShutdownMonitor measures true wall time
    pipeline: dict[str, Any] = build_training_pipeline(
        cfg,
        device_override=args.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        start_time=_session_start,   # [FIX H-1]
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

    runner = pipeline["runner"]
    if args.max_iter > 0:
        runner.config.max_iterations = args.max_iter

    try:
        summary: dict[str, Any] = runner.run()
    except KeyboardInterrupt:
        if rank == 0:
            logger.warning("KeyboardInterrupt — graceful shutdown...")
        summary = {"interrupted": True}
    except Exception as exc:
        if rank == 0:
            logger.error("Critical error: %s", exc, exc_info=True)
            action: str = pipeline["fault_handler"].handle_generic_error(exc)
            logger.info("FaultHandler recommendation: %s", action)
        summary = {"error": str(exc)}
    finally:
        if rank == 0:
            # Flush async upload THEN stop thread — correct order [FIX M-2]
            if pipeline.get("uploader") and pipeline["uploader"].is_active():
                pipeline["uploader"].trigger_manual_upload()
                pipeline["uploader"].shutdown()

            if pipeline.get("monitor"):
                pipeline["monitor"].finish()

        # DDP cleanup — all ranks must participate
        if world_size > 1:
            try:
                dist.barrier()
                dist.destroy_process_group()
                if rank == 0:
                    logger.info("DDP process group destroyed cleanly.")
            except Exception as e:
                if rank == 0:
                    logger.warning("Error destroying DDP process group: %s", e)

    if rank == 0:
        elapsed: float = time.monotonic() - _session_start
        logger.info(
            "=" * 60 + "\n"
            "  TRAINING COMPLETE\n"
            "  Runtime: %.2f hours\n"
            "  Result: %s\n"
            "=" * 60,
            elapsed / 3600, summary,
        )


if __name__ == "__main__":
    main()
