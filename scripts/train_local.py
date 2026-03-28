#!/usr/bin/env python3
"""
Lokalis Training CLI Belepesi Pont (train_local.py).

[FIX P0-A — 2025-03-28] Opponent pool wired into the training loop.

    build_training_pipeline() now:
      1. Creates an OpponentPool with the four Phase-0 archetypes.
      2. Passes it to make_env() so a MultiAgentRLCardWrapper is used.
      3. Registers the archetypes with CurriculumManager's UCB1 arms.
      4. In on_iteration_end(), calls curriculum.select_opponent() /
         select_opponent_fsp() and routes the result to
         env.set_active_opponent() so the correct bot is in place for
         the NEXT rollout.

[FIX P1-A — 2025-03-28] Average-strategy network added (NFSP).

    A second network avg_network is kept as an exponential moving average
    of the best-response network:
        α  = 1 / max(iteration, 1)   (time-decaying, exact FSP average)
        avg_p ← lerp(avg_p, br_p, α)

    This is the minimal NFSP implementation that prevents the RPS cycling
    (rock-paper-scissors dynamics) in Phase 2 self-play.  The avg_network
    is registered as a NeuralOpponentAgent ("avg_strategy") in the
    opponent pool so Phase 2 can sample from it.

[FIX R-1 — 2025-03-28] DDP Checkpoint Deadlock fixed in on_checkpoint.
[FIX C-2 — 2025-03-28] Shutdown signal broadcast to all DDP ranks.
[FIX H-1 — 2025-03-28] _session_start captured BEFORE build_training_pipeline().
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

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

def create_environment(
    cfg: dict[str, Any],
    opponent_pool: Any | None = None,
) -> Any:
    """Create the poker environment, optionally wired to an opponent pool.

    [FIX P0-A] When opponent_pool is provided, make_env returns a
    MultiAgentRLCardWrapper that injects opponent actions from the pool
    for all non-hero seats.  When None, falls back to the original
    self-play RLCardWrapper.
    """
    from src.env.wrappers import make_env
    return make_env(cfg, opponent_pool=opponent_pool)


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

    [FIX P0-A] Creates an OpponentPool and wires it into the environment.
    [FIX P1-A] Creates an average-strategy network for NFSP convergence.

    Args:
        start_time: Monotonic clock captured BEFORE this call. [FIX H-1]
    """
    logger = logging.getLogger(__name__)

    # --- Device ---
    device_str: str = device_override or cfg.get("runtime", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s (rank=%d/%d)", device, rank, world_size)

    # --- Seed (DDP-aware) ---
    from src.mlops.state_manager import RNGStateManager
    seed: int = cfg.get("project", {}).get("seed", 42)
    seed_with_rank: int = seed + rank
    dl_generator = RNGStateManager.set_global_seed(seed_with_rank)
    if world_size > 1:
        logger.info(
            "DDP seeding: base=%d, rank_offset=%d, effective=%d",
            seed, rank, seed_with_rank,
        )

    # =========================================================================
    # [FIX P0-A] Opponent Pool — must be created before environment
    # =========================================================================
    from src.training.opponent_pool import OpponentPool
    mab_cfg = cfg.get("orchestrator", {}).get("mab", {})
    opponent_pool = OpponentPool(
        archetype_names=["calling_station", "maniac", "random", "tight_passive"],
        max_pool_size=int(mab_cfg.get("max_pool_size", 20)),
        device=device,
    )
    logger.info(
        "[P0-A] OpponentPool created: %s",
        opponent_pool.get_all_archetype_names(),
    )

    # --- Environment (wired to opponent pool) ---
    env = create_environment(cfg, opponent_pool=opponent_pool)

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

    # =========================================================================
    # [FIX P1-A] Average-Strategy Network (NFSP)
    # =========================================================================
    # We maintain a second network whose parameters are the time-average of
    # all past best-response networks.  This implements Fictitious Self-Play
    # (Heinrich & Silver 2015) at the weight level: the average strategy
    # cannot be exploited by any single counter-strategy, so Phase-2
    # training converges toward Nash instead of cycling.
    #
    # The EMA update in on_iteration_end():
    #     α = 1 / max(iteration, 1)       (exact FSP time-average)
    #     avg_p ← lerp(avg_p, br_p, α)
    # =========================================================================
    _base_net = (
        network.module
        if isinstance(network, torch.nn.parallel.DistributedDataParallel)
        else network
    )
    avg_network = copy.deepcopy(_base_net)
    avg_network.eval()
    for p in avg_network.parameters():
        p.requires_grad_(False)
    logger.info("[P1-A] Average-strategy network created (NFSP)")

    # Register avg_network as a NeuralOpponentAgent in the pool so that
    # Phase-2 FSP opponent sampling can use it.
    from src.training.opponent_pool import NeuralOpponentAgent
    # Build a fresh obs_builder for the avg agent (no state shared with hero)
    avg_obs_builder = ObservationBuilder(obs_config)
    avg_agent = NeuralOpponentAgent(
        name="avg_strategy",
        network=avg_network,
        obs_builder=avg_obs_builder,
        device=device,
    )
    opponent_pool.archetypes["avg_strategy"] = avg_agent
    logger.info("[P1-A] avg_strategy agent registered in OpponentPool")

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
        logger.info("Resume mode enabled. Attempting to load checkpoint...")

        if checkpoint_path:
            checkpoint_to_resume = state_manager.ckpt_mgr.load(
                checkpoint_path, map_location=device
            )
        else:
            ckpt_list = state_manager.list_checkpoints()
            if ckpt_list:
                logger.info(
                    "Resume: found %d checkpoints. Latest: %s",
                    len(ckpt_list),
                    ckpt_list[-1].name,
                )
            else:
                logger.warning("Resume: NO checkpoints found. Cold start.")

            checkpoint_to_resume = state_manager.load_training_state(
                map_location=str(device)
            )

        if checkpoint_to_resume is not None:
            model_state_dict = checkpoint_to_resume["model_state_dict"]
            resumed_iteration = checkpoint_to_resume.get("iteration", 0)

            if world_size > 1:
                if not any(k.startswith("module.") for k in model_state_dict.keys()):
                    model_state_dict = {
                        f"module.{k}": v for k, v in model_state_dict.items()
                    }

            network.load_state_dict(model_state_dict)
            start_iteration = resumed_iteration
            orchestrator_state = checkpoint_to_resume.get("orchestrator_state", {})

            # [P1-A] Restore avg_network from its dedicated file (saved
            # alongside every main checkpoint as avg_network_latest.pt).
            ckpt_dir = Path(
                cfg.get("mlops", {})
                .get("checkpoint", {})
                .get("local_checkpoint_dir", "checkpoints")
            )
            avg_latest = ckpt_dir / "avg_network_latest.pt"
            if avg_latest.exists():
                try:
                    avg_state = torch.load(
                        str(avg_latest), map_location=device, weights_only=True
                    )
                    avg_network.load_state_dict(avg_state)
                    logger.info("[P1-A] avg_network restored from %s", avg_latest)
                except Exception as avg_exc:
                    logger.warning(
                        "[P1-A] avg_network restore failed (%s); "
                        "starting from fresh deepcopy of main network.",
                        avg_exc,
                    )
            else:
                logger.info(
                    "[P1-A] No avg_network_latest.pt found; "
                    "avg_network stays as fresh deepcopy."
                )

            logger.info("✓ RESUME SUCCESSFUL: iteration=%d", start_iteration)
        else:
            logger.warning("✗ RESUME FAILED: cold start.")

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

        # [FIX P0-A] Register static archetypes in the CurriculumManager's
        # UCB1 arms so opponent selection works from iteration 0.
        for _opp in ["calling_station", "maniac", "random", "tight_passive"]:
            orchestrator.curriculum.register_opponent(_opp)
        logger.info(
            "[P0-A] 4 archetypes registered in Curriculum UCB1 arms; "
            "starting with 'calling_station'."
        )
        # Set a sensible default for the very first rollout
        if hasattr(env, "set_active_opponent"):
            env.set_active_opponent("calling_station")

        logger.info("Orchestrator initialized on rank 0")

    # --- Graceful Shutdown (rank 0 only) ---
    from src.mlops.fault_tolerance import GracefulShutdownMonitor, ShutdownConfig
    shutdown_monitor: GracefulShutdownMonitor | None = None

    if rank == 0:
        shutdown_monitor = GracefulShutdownMonitor(
            ShutdownConfig.from_dict(cfg),
            start_time=start_time,
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

            # FSP PolicyAverager snapshot (Phase 2 only)
            if orchestrator is not None:
                orchestrator.curriculum.maybe_record_fsp_snapshot(
                    network=network,
                    iteration=iteration,
                )

            # ESS metrics
            if monitor.active and orchestrator is not None:
                avg_stats = orchestrator.curriculum._policy_averager.get_stats()
                monitor.log_metrics(step=iteration, metrics={
                    "fsp/pool_size":     float(avg_stats["pool_size"]),
                    "fsp/ess":           float(avg_stats["effective_ess"]),
                    "fsp/total_weight":  float(avg_stats["total_weight"]),
                })

            # =================================================================
            # [FIX P0-A] Wire opponent selection to the environment.
            #
            # We select AFTER the orchestrator callback so that any phase
            # transitions have already been applied and the curriculum phase
            # is up-to-date.  The selected opponent will be active for the
            # NEXT collect_rollout() call.
            # =================================================================
            phase_val = orchestrator.curriculum.current_phase.value
            if phase_val < 2:
                opp_name = orchestrator.curriculum.select_opponent()
            else:
                # Phase 2: FSP-aware sampling (time-weighted average strategy)
                opp_name = orchestrator.curriculum.select_opponent_fsp(iteration)

            if hasattr(runner, "env") and hasattr(runner.env, "set_active_opponent"):
                runner.env.set_active_opponent(opp_name)
                logger.debug(
                    "[P0-A] Opponent for iter %d: '%s' (phase=%d)",
                    iteration + 1, opp_name, phase_val,
                )

        # =====================================================================
        # [FIX P1-A] Update average-strategy network via EMA (NFSP).
        #
        # α = 1 / max(iteration, 1) implements the exact FSP time-average:
        #     σ̄(t) = (1/t) Σ_{k=1}^t σ(k)
        # which can be written as the recursive update:
        #     σ̄(t) = (1 - 1/t) * σ̄(t-1) + (1/t) * σ(t)
        #
        # This runs on rank 0 only.  All gradient operations are disabled.
        # =====================================================================
        if rank == 0:
            actual_net = (
                network.module
                if isinstance(network, torch.nn.parallel.DistributedDataParallel)
                else network
            )
            ema_alpha = 1.0 / max(float(iteration), 1.0)
            with torch.no_grad():
                for avg_p, br_p in zip(
                    avg_network.parameters(), actual_net.parameters()
                ):
                    avg_p.data.lerp_(br_p.data, ema_alpha)
            logger.debug(
                "[P1-A] avg_network EMA updated: α=%.6f (iter=%d)",
                ema_alpha, iteration,
            )

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

        [FIX C-2] Broadcasts shutdown flag from rank 0 to all ranks.
        """
        nonlocal phase_transition_state

        if world_size <= 1:
            return

        # Broadcast 1: Phase transition
        phase_flag = torch.tensor(
            [1.0 if phase_transition_state["transition_occurred"] else 0.0],
            dtype=torch.float32,
            device=device,
        )
        dist.broadcast(phase_flag, src=0)
        phase_transition_state["transition_occurred"] = bool(phase_flag.item() > 0.5)

        # Broadcast 2: Shutdown flag [FIX C-2]
        should_stop_val = 1.0 if runner._should_stop else 0.0
        stop_flag = torch.tensor(
            [should_stop_val], dtype=torch.float32, device=device
        )
        dist.broadcast(stop_flag, src=0)

        if bool(stop_flag.item() > 0.5) and not runner._should_stop:
            logger.info(
                "[Rank %d] Shutdown broadcast received — requesting stop.",
                rank,
            )
            runner.request_stop()

    # =========================================================================
    # [FIX R-1] on_checkpoint — unconditional DDP barrier after rank-0 I/O
    # =========================================================================

    def on_checkpoint(iteration: int, net: Any) -> None:
        """Periodic checkpoint callback — rank-0 writes, ALL ranks synchronize."""
        if rank == 0:
            try:
                rng_states: dict[str, Any] = RNGStateManager.capture_states(
                    dataloader_generator=dl_generator
                )

                # [P1-A] Save avg_network state separately.
                # StateManager.save_training_state() has a fixed signature and
                # does not accept arbitrary extra keys, so we persist the
                # average-strategy network to a dedicated file alongside the
                # main checkpoint.  A single "latest" file is sufficient — we
                # only ever need to resume from the most recent avg state.
                try:
                    avg_ckpt_dir = Path(
                        ckpt_cfg.get("local_checkpoint_dir", "checkpoints")
                    )
                    avg_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    avg_tmp  = avg_ckpt_dir / "avg_network_latest.pt.tmp"
                    avg_path = avg_ckpt_dir / "avg_network_latest.pt"
                    torch.save(avg_network.state_dict(), str(avg_tmp))
                    os.replace(str(avg_tmp), str(avg_path))   # atomic (SIGKILL-safe)
                    logger.info("[P1-A] avg_network saved: %s", avg_path)
                except Exception as avg_exc:
                    logger.error(
                        "[P1-A] avg_network save failed at iter %d: %s",
                        iteration, avg_exc,
                    )

                state_manager.save_training_state(
                    network=net,
                    optimizer=runner.trainer.optimizer,
                    scheduler=runner.trainer.scheduler,
                    iteration=iteration,
                    total_env_steps=runner.collector.get_total_steps(),
                    total_hands=runner.collector.get_total_episodes(),
                    best_mean_reward=-float("inf"),
                    orchestrator_state=(
                        orchestrator.get_state()
                        if orchestrator is not None
                        else {}
                    ),
                    config=cfg,
                    rng_states=rng_states,
                    wandb_run_id=(
                        monitor.run_id
                        if (monitor is not None and monitor.active)
                        else None
                    ),
                    is_best=False,
                )
                logger.info(
                    "[Rank 0] Checkpoint saved: iter=%d, steps=%d",
                    iteration,
                    runner.collector.get_total_steps(),
                )

            except Exception as save_exc:
                logger.error(
                    "[Rank 0] Checkpoint save FAILED at iter %d: %s",
                    iteration,
                    save_exc,
                    exc_info=True,
                )

        # Unconditional DDP barrier [FIX R-1]
        if world_size > 1:
            try:
                dist.barrier()
                if rank == 0:
                    logger.debug(
                        "[Rank 0] DDP barrier cleared post-checkpoint (iter=%d)",
                        iteration,
                    )
            except Exception as barrier_exc:
                logger.critical(
                    "[Rank %d] dist.barrier() failed after checkpoint at iter %d: %s",
                    rank, iteration, barrier_exc, exc_info=True,
                )
                raise RuntimeError(
                    f"[DDP] Irrecoverable barrier failure in on_checkpoint "
                    f"at iteration {iteration}. Original error: {barrier_exc}"
                ) from barrier_exc

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
            logger.info("LR scheduler state restored from checkpoint")

        if "rng_states" in checkpoint_to_resume and checkpoint_to_resume["rng_states"]:
            RNGStateManager.restore_states(checkpoint_to_resume["rng_states"])
            logger.info("RNG states restored (deterministic resume)")

    if rank == 0 and orchestrator is not None:
        orchestrator.set_trainer_reference(runner.trainer)

    return {
        "runner":           runner,
        "network":          network,
        "avg_network":      avg_network,    # [P1-A] average-strategy network
        "opponent_pool":    opponent_pool,  # [P0-A] wired opponent pool
        "orchestrator":     orchestrator,
        "state_manager":    state_manager,
        "shutdown_monitor": shutdown_monitor,
        "monitor":          monitor,
        "uploader":         uploader,
        "fault_handler":    fault_handler,
        "config":           cfg,
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
    # build_training_pipeline().
    _session_start: float = time.monotonic()
    logger.info(
        "Session clock started: %.4f (monotonic) [FIX H-1]", _session_start
    )

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

    if args.max_iter > 0:
        cfg.setdefault("runtime", {})["max_iterations"] = args.max_iter

    pipeline: dict[str, Any] = build_training_pipeline(
        cfg,
        device_override=args.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        start_time=_session_start,
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
            if pipeline.get("uploader") and pipeline["uploader"].is_active():
                pipeline["uploader"].trigger_manual_upload()
                pipeline["uploader"].shutdown()

            if pipeline.get("monitor"):
                pipeline["monitor"].finish()

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
