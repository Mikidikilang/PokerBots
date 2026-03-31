#!/usr/bin/env python3
"""
VR-DeepPDCFR+ Master Training Script for 6-Max No-Limit Hold'em

This script orchestrates the complete VR-DeepPDCFR+ pipeline including:
- Game environment initialization
- Per-player network and buffer management
- External Sampling MCCFR traversal
- Network training loop
- WandB logging and monitoring
- Checkpoint management
- Periodic LBR exploitability evaluation

Usage:
    python scripts/train_6max_vr_deep.py [--config config.yaml]

Author: VR-DeepPDCFR+ Team
Date: March 31, 2026
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
import yaml

# Project imports
from src.env.action_mapper import ActionMapper
from src.env.equity import EquityCalculator
from src.env.features import ObservationBuilder
from src.env.wrappers import RLCardWrapper, WrapperConfig
from src.mlops.monitoring import WandbMonitor
from src.mlops.state_manager import CheckpointManager
from src.model.networks import VRDeepPDCFRNetworks
from src.training.buffers import BufferManager
from src.training.runner import GameStateAdapter
from src.training.vr_deep_pdcfr_engine import VRDeepPDCFREngine
from src.evaluation.nash_evaluator import LocalBestResponseEvaluator, NashEvalConfig

logger = logging.getLogger(__name__)


class VRDeepPDCFRTrainer:
    """Master trainer for VR-DeepPDCFR+ on 6-Max NLHE."""

    def __init__(self, config_path: str | Path = "config.yaml"):
        """Initialize trainer and all components.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        logger.info(f"Loaded config from {self.config_path}")
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize MLOps
        self.wandb_monitor = WandbMonitor()
        self.wandb_monitor.setup(config=self.config, resume=False)
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
            max_to_keep=self.config.get("max_checkpoints", 5),
        )
        logger.info("MLOps initialized")
        
        # Initialize game environment
        self.env = self._init_environment()
        self.obs_builder = ObservationBuilder(config=None)
        self.action_mapper = ActionMapper()
        logger.info("Game environment initialized")
        
        # Initialize per-player components
        self.num_players = self.config["environment"]["num_players"]
        self.buffer_managers: Dict[int, BufferManager] = {}
        self.networks: Dict[int, VRDeepPDCFRNetworks] = {}
        self.optimizers: Dict[int, Dict[str, optim.Optimizer]] = {}
        
        self._init_player_components()
        logger.info(f"Initialized components for {self.num_players} players")
        
        # Initialize engine
        self.engine = VRDeepPDCFREngine(
            buffer_managers=self.buffer_managers,
            networks=self.networks,
            optimizers=self.optimizers,
            device=self.device,
            max_depth=self.config["cfr"].get("max_tree_depth", 60),
        )
        logger.info("VRDeepPDCFREngine initialized")
        
        # Setup evaluation (LBR oracle for Player 0)
        self.evaluator: Optional[LocalBestResponseEvaluator] = None
        self._init_evaluator()
        
        # Training state
        self.current_iteration = 1
        self.total_iterations = self.config["cfr"].get("num_iterations", 50000)
        
        logger.info(f"Trainer initialized. Will run {self.total_iterations} iterations.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configure logging to console and file."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logger.info(f"Logging to {log_file}")
    
    def _get_wandb_tags(self) -> list[str]:
        """Get WandB tags for this run."""
        tags = [
            "vr-deep-pdcfr+",
            f"players-{self.config['environment']['num_players']}",
            f"iterations-{self.config['cfr'].get('num_iterations', 50000)}",
        ]
        if self.config.get("environment", {}).get("game_type") == "no-limit-holdem":
            tags.append("nlhe")
        return tags
    
    def _init_environment(self) -> RLCardWrapper:
        """Initialize RLCard 6-Max NLHE environment."""
        wrapper_config = WrapperConfig.from_dict(self.config)
        env = RLCardWrapper(config=wrapper_config)
        return env
    
    def _init_player_components(self):
        """Initialize buffer managers, networks, and optimizers for each player."""
        obs_dim = self.obs_builder.get_observation_dim()
        num_actions = self.config["environment"]["action_space"]["num_actions"]
        
        logger.info(f"Creating components with obs_dim={obs_dim}, num_actions={num_actions}")
        
        for player_id in range(self.num_players):
            # Buffer manager
            self.buffer_managers[player_id] = BufferManager(
                advantage_capacity=self.config["buffers"].get("advantage_buffer_size", 100_000),
                strategy_capacity=self.config["buffers"].get("strategy_buffer_size", 1_000_000),
                time_decay_power=self.config["buffers"].get("time_decay_power", 1.0),
            )
            
            # Networks
            self.networks[player_id] = VRDeepPDCFRNetworks(
                input_dim=obs_dim,
                output_dim=num_actions,
                hidden_dims=self.config["networks"]["shared_architecture"]["hidden_dims"],
                activation=torch.nn.ReLU,
                use_layer_norm=self.config["networks"]["shared_architecture"].get("use_layer_norm", False),
                dropout_p=self.config["networks"]["shared_architecture"].get("dropout_p", 0.0),
            )
            
            # Move networks to device CRITICAL: Must happen BEFORE creating optimizers
            self.networks[player_id].to_device(self.device)
            
            # Set networks to training mode
            self.networks[player_id].train_mode()
            
            # Optimizers (4 per player: cumulative, instantaneous, value, strategy)
            lr_config = self.config["networks"]
            self.optimizers[player_id] = {
                "cumulative": optim.Adam(
                    self.networks[player_id].cumulative_advantage.parameters(),
                    lr=lr_config["cumulative_advantage"]["learning_rate"],
                    eps=lr_config["cumulative_advantage"]["adam_epsilon"],
                ),
                "instantaneous": optim.Adam(
                    self.networks[player_id].instantaneous_advantage.parameters(),
                    lr=lr_config["instantaneous_advantage"]["learning_rate"],
                    eps=lr_config["instantaneous_advantage"]["adam_epsilon"],
                ),
                "value": optim.Adam(
                    self.networks[player_id].value.parameters(),
                    lr=lr_config["value_baseline"]["learning_rate"],
                    eps=lr_config["value_baseline"]["adam_epsilon"],
                ),
                "strategy": optim.Adam(
                    self.networks[player_id].strategy.parameters(),
                    lr=lr_config["average_strategy"]["learning_rate"],
                    eps=lr_config["average_strategy"]["adam_epsilon"],
                ),
            }
            
            logger.debug(f"Player {player_id}: buffers, networks, and optimizers initialized")
    
    def _init_evaluator(self):
        """Initialize LBR evaluator for Player 0."""
        try:
            eval_config = NashEvalConfig(eval_hands=10)  # Smoke test: 10 hands only
            equity_calc = EquityCalculator()
            
            self.evaluator = LocalBestResponseEvaluator(
                strategy_network=self.networks[0].strategy,
                env=self.env,
                obs_builder=self.obs_builder,
                action_mapper=self.action_mapper,
                equity_calc=equity_calc,
                config=eval_config,
                device=self.device,
            )
            logger.info("LBR Evaluator initialized for Player 0")
        except Exception as e:
            logger.warning(f"Failed to initialize LBR Evaluator: {e}. Evaluation disabled.")
            self.evaluator = None
    
    def train(self):
        """Execute the main training loop."""
        logger.info("=" * 80)
        logger.info("Starting VR-DeepPDCFR+ Training Loop")
        logger.info("=" * 80)
        
        try:
            for iteration in range(1, self.total_iterations + 1):
                self.current_iteration = iteration
                
                # Log iteration start
                logger.info(f"Iteration {iteration}/{self.total_iterations} - Starting")
                
                # Start iteration
                self.engine.start_iteration()
                logger.info(f"Iteration {iteration} - Engine iteration started")
                
                # External Sampling MCCFR: traverse for each updating player
                # CRITICAL: Reset environment and reach probs INSIDE the loop for independent MC sampling
                for updating_player in range(self.num_players):
                    logger.info(f"Iteration {iteration} - Player {updating_player} traversal starting")
                    # Fresh environment reset for each updating player (independent deck sampling)
                    self.env.reset()
                    root_state = GameStateAdapter(self.env, self.obs_builder)
                    
                    # Fresh reach probability initialization for each traversal
                    initial_reach_probs = {i: 1.0 for i in range(self.num_players)}
                    
                    # Execute traversal
                    values = self.engine.traverse(
                        state=root_state,
                        player_reach_probs=initial_reach_probs,
                        updating_player=updating_player,
                        depth=0,
                    )
                    logger.info(f"Iteration {iteration} - Player {updating_player} traversal completed")
                
                # Train all networks
                logger.info(f"Iteration {iteration} - Training networks")
                losses = self.engine.train_networks()
                logger.info(f"Iteration {iteration} - Networks trained")
                
                # End iteration
                self.engine.end_iteration()
                logger.info(f"Iteration {iteration} - Engine iteration ended")
                
                # Log to WandB
                self._log_iteration(iteration, losses)
                
                # Checkpoint periodically
                save_interval = self.config.get("save_interval_iterations", 100)
                if iteration % save_interval == 0:
                    self._save_checkpoint(iteration)
                
                # Evaluate periodically
                eval_interval = self.config["cfr"].get("exploitability_update_freq", 250)
                if self.evaluator is not None and iteration % eval_interval == 0:
                    self._run_evaluation(iteration)
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self._save_emergency_checkpoint()
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            self._save_emergency_checkpoint()
            raise
        
        logger.info("=" * 80)
        logger.info("Training completed successfully")
        logger.info("=" * 80)
        
        # Final checkpoint
        self._save_checkpoint(self.total_iterations)
    
    def _log_iteration(self, iteration: int, losses: Dict[str, float]):
        """Log iteration results to WandB."""
        log_data = {"iteration": iteration}
        log_data.update(losses)
        
        self.wandb_monitor.log_metrics(iteration, log_data)
        
        if iteration % 50 == 0:
            loss_summary = ", ".join(
                f"{k}={v:.6f}" for k, v in list(losses.items())[:3]
            )
            logger.debug(f"Iteration {iteration}: {loss_summary}")
    
    def _save_checkpoint(self, iteration: int):
        """Save checkpoint of networks and optimizers."""
        try:
            checkpoint_data = {
                "iteration": iteration,
                "networks": {
                    pid: net.state_dict()
                    for pid, net in self.networks.items()
                },
                "optimizers": {
                    pid: {
                        opt_name: opt.state_dict()
                        for opt_name, opt in opts_dict.items()
                    }
                    for pid, opts_dict in self.optimizers.items()
                },
            }
            
            self.checkpoint_manager.save(checkpoint_data, iteration=iteration)
            logger.info(f"Checkpoint saved at iteration {iteration}")
            
            # Log to WandB
            self.wandb_monitor.log_metrics(iteration, {"checkpoint_saved": 1})
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on shutdown."""
        try:
            checkpoint_data = {
                "iteration": self.current_iteration,
                "networks": {
                    pid: net.state_dict()
                    for pid, net in self.networks.items()
                },
                "optimizers": {
                    pid: {
                        opt_name: opt.state_dict()
                        for opt_name, opt in opts_dict.items()
                    }
                    for pid, opts_dict in self.optimizers.items()
                },
                "emergency": True,
            }
            
            self.checkpoint_manager.save(checkpoint_data, iteration=self.current_iteration, is_best=False)
            logger.warning(f"Emergency checkpoint saved at iteration {self.current_iteration}")
            
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
    
    def _run_evaluation(self, iteration: int):
        """Run LBR evaluation and log results."""
        if self.evaluator is None:
            return
        
        try:
            logger.info(f"Running LBR evaluation at iteration {iteration}")
            
            # Run evaluation - returns NashEvalResults object
            results = self.evaluator.run_evaluation()
            
            # Log to WandB
            eval_data = {
                "evaluation_iteration": iteration,
                "nash_distance_pct": results.nash_distance_pct,
                "oracle_mbb_hand": results.oracle_mbb_hand,
            }
            self.wandb_monitor.log_metrics(iteration, eval_data)
            
            logger.info(
                f"Evaluation results: "
                f"Nash distance={results.nash_distance_pct:.2f}%, "
                f"Oracle mbb/hand={results.oracle_mbb_hand:.6f}"
            )
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VR-DeepPDCFR+ Master Training Script for 6-Max NLHE"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Disable W&B logging (offline mode)",
    )
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = VRDeepPDCFRTrainer(config_path=args.config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.warning(f"Received signal {sig}, shutting down gracefully...")
        trainer._save_emergency_checkpoint()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
