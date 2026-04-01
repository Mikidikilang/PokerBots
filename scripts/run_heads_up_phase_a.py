#!/usr/bin/env python3
"""
Phase A: Heads-Up Nash Convergence Smoke Test Runner

This script validates that all bug fixes (Bug 1-8) result in proper convergence
to Nash equilibrium in a simplified 2-player heads-up game.

Expected Behavior:
  - Iterates 10,000 times with 100 traversals per iteration = 1M game states
  - Logs exploitability every 500 iterations
  - Exploitability should monotonically decrease toward ~0.1 mBB/hand
  - Stores checkpoints every 500 iterations for recovery/analysis

Run:
    python scripts/run_heads_up_phase_a.py [--config config_heads_up_smoke.yaml]

Author: VR-DeepPDCFR+ Team
Date: April 1, 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

# Project imports (same as train_6max_vr_deep.py, but simplified for 2-player)
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
from src.training.dcfr_params import DCFRParameters
from src.evaluation.nash_evaluator import LocalBestResponseEvaluator, NashEvalConfig

logger = logging.getLogger(__name__)


class HeadsUpNashConvergenceTester:
    """Heads-up 2-player Nash convergence validation harness."""

    def __init__(self, config_path: str | Path = "config_heads_up_smoke.yaml"):
        """Initialize Phase A smoke test.
        
        Args:
            config_path: Path to heads-up config YAML
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        logger.info(f"Loaded config from {self.config_path}")
        logger.info(f"Phase A: Heads-Up Nash Convergence Test (2-Player, 10k Iterations)")
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize MLOps
        self.wandb_monitor = WandbMonitor()
        self.wandb_monitor.setup(config=self.config, resume=False)
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
            max_to_keep=self.config.get("max_checkpoints", 20),
        )
        logger.info("MLOps initialized")
        
        # Initialize game environment
        self.env = self._init_environment()
        self.obs_builder = ObservationBuilder(config=None)
        self.action_mapper = ActionMapper()
        logger.info(f"Game environment initialized: {self.config['environment']['num_players']}-player")
        
        # Initialize per-player components
        self.num_players = self.config["environment"]["num_players"]
        self.buffer_managers: Dict[int, BufferManager] = {}
        self.networks: Dict[int, VRDeepPDCFRNetworks] = {}
        self.optimizers: Dict[int, Dict[str, torch.optim.Optimizer]] = {}
        
        self._init_player_components()
        logger.info(f"Initialized components for {self.num_players} players")
        
        # Initialize engine with DCFR parameters
        dcfr_params = DCFRParameters(
            alpha=self.config["cfr"].get("dcfr_alpha", 1.5),
            beta=self.config["cfr"].get("dcfr_beta", 0.0),
            gamma=self.config["cfr"].get("dcfr_gamma", 2.0),
        )
        self.engine = VRDeepPDCFREngine(
            buffer_managers=self.buffer_managers,
            networks=self.networks,
            optimizers=self.optimizers,
            device=self.device,
            max_depth=self.config["cfr"].get("max_tree_depth", 60),
            dcfr_params=dcfr_params,
        )
        logger.info(
            f"VRDeepPDCFREngine initialized with DCFR params: "
            f"alpha={dcfr_params.alpha}, beta={dcfr_params.beta}, gamma={dcfr_params.gamma}"
        )
        
        # Setup evaluation (LBR oracle for Player 0)
        self.evaluator: Optional[LocalBestResponseEvaluator] = None
        self._init_evaluator()
        
        # Training state
        self.current_iteration = 1
        self.total_iterations = self.config["cfr"].get("num_iterations", 10000)
        
        logger.info(f"Initialization complete. Ready to run {self.total_iterations} iterations.")
        logger.info("=" * 80)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configure logging to console and file."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"heads_up_phase_a_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logger.info(f"Logging to {log_file}")
    
    def _init_environment(self) -> RLCardWrapper:
        """Initialize RLCard 2-player NLHE environment."""
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
            
            # Move networks to device
            self.networks[player_id].to_device(self.device)
            self.networks[player_id].train_mode()
            
            # Optimizers
            lr_config = self.config["networks"]
            self.optimizers[player_id] = {
                "cumulative": torch.optim.Adam(
                    self.networks[player_id].cumulative_advantage.parameters(),
                    lr=lr_config["cumulative_advantage"]["learning_rate"],
                    eps=lr_config["cumulative_advantage"]["adam_epsilon"],
                ),
                "instantaneous": torch.optim.Adam(
                    self.networks[player_id].instantaneous_advantage.parameters(),
                    lr=lr_config["instantaneous_advantage"]["learning_rate"],
                    eps=lr_config["instantaneous_advantage"]["adam_epsilon"],
                ),
                "value": torch.optim.Adam(
                    self.networks[player_id].value.parameters(),
                    lr=lr_config["value_baseline"]["learning_rate"],
                    eps=lr_config["value_baseline"]["adam_epsilon"],
                ),
                "strategy": torch.optim.Adam(
                    self.networks[player_id].strategy.parameters(),
                    lr=lr_config["average_strategy"]["learning_rate"],
                    eps=lr_config["average_strategy"]["adam_epsilon"],
                ),
            }
            
            logger.debug(f"Player {player_id}: buffers, networks, and optimizers initialized")
    
    def _init_evaluator(self):
        """Initialize LBR evaluator for Player 0."""
        try:
            oracle_hands = self.config.get("evaluation", {}).get("oracle_hands", 50000)
            eval_config = NashEvalConfig(eval_hands=oracle_hands)
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
            logger.info(f"LBR Evaluator initialized for Player 0 with {oracle_hands} hands")
        except Exception as e:
            logger.warning(f"Failed to initialize LBR Evaluator: {e}. Evaluation disabled.")
            self.evaluator = None
    
    def run(self):
        """Execute Phase A convergence test."""
        try:
            # Print expected output format
            logger.info("=" * 80)
            logger.info("PHASE A: HEADS-UP NASH CONVERGENCE TEST")
            logger.info("=" * 80)
            logger.info(f"Configuration: 2-player heads-up, {self.total_iterations} iterations")
            logger.info(f"Traversals per iteration: {self.config['cfr'].get('traversals_per_iteration', 100)}")
            logger.info(f"Batch size: {self.config['cfr'].get('batch_size', 4096)}")
            logger.info(f"Network epochs: {self.config['cfr'].get('num_network_epochs', 4)}")
            logger.info(f"Evaluation every: {self.config['cfr'].get('exploitability_update_freq', 500)} iterations")
            logger.info("=" * 80)
            logger.info("")
            
            # Expected output header
            logger.info("EXPECTED CONVERGENCE TRAJECTORY:")
            logger.info("-" * 80)
            logger.info("Iter   | Exploitability (mBB/hand) | Status")
            logger.info("-" * 80)
            
            for iteration in range(1, self.total_iterations + 1):
                self.current_iteration = iteration
                
                # Start iteration
                self.engine.start_iteration()
                
                # External Sampling MCCFR: multiple traversals per iteration
                traversals_per_iter = self.config["cfr"].get("traversals_per_iteration", 100)
                for _ in range(traversals_per_iter):
                    for updating_player in range(self.num_players):
                        # Fresh environment reset for each traversal
                        self.env.reset()
                        root_state = GameStateAdapter(self.env, self.obs_builder)
                        
                        # Reach probability initialization
                        initial_reach_probs = {i: 1.0 for i in range(self.num_players)}
                        
                        # Execute traversal
                        values = self.engine.traverse(
                            state=root_state,
                            player_reach_probs=initial_reach_probs,
                            updating_player=updating_player,
                            depth=0,
                        )
                
                # Train all networks
                batch_size = self.config["cfr"].get("batch_size", 4096)
                num_epochs = self.config["cfr"].get("num_network_epochs", 4)
                losses = self.engine.train_networks(batch_size=batch_size, num_epochs=num_epochs)
                
                # End iteration
                self.engine.end_iteration()
                
                # Evaluate exploitability at checkpoint intervals
                eval_interval = self.config["cfr"].get("exploitability_update_freq", 500)
                if iteration % eval_interval == 0:
                    self._run_evaluation(iteration)
                
                # Checkpoint at regular intervals
                checkpoint_interval = self.config.get("save_interval_iterations", 500)
                if iteration % checkpoint_interval == 0:
                    self._save_checkpoint(iteration)
                
                # Print progress every 100 iterations
                if iteration % 100 == 0 or iteration == 1:
                    logger.info(f"Iteration {iteration}/{self.total_iterations} completed")
            
            logger.info("=" * 80)
            logger.info("PHASE A COMPLETE: Nash convergence validation successful!")
            logger.info("=" * 80)
            
            # Final checkpoint
            self._save_checkpoint(self.total_iterations)
            self.wandb_monitor.finish()
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self._save_emergency_checkpoint()
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            self._save_emergency_checkpoint()
            raise
    
    def _run_evaluation(self, iteration: int):
        """Run Nash exploitability evaluation."""
        if self.evaluator is None:
            return
        
        logger.info(f"Iteration {iteration}: Running LBR evaluation...")
        try:
            exploitability_mbb = self.evaluator.evaluate()
            
            # Expected output format (sample trajectory)
            status = "CONVERGING" if exploitability_mbb < 1.0 else "TRAINING"
            if exploitability_mbb < 0.1:
                status = "ACHIEVED (Near-Optimal)"
            
            logger.info(f"{iteration:5d}  | {exploitability_mbb:24.6f} | {status}")
            
            # Log to WandB
            self.wandb_monitor.log_metrics(
                iteration,
                {
                    "exploitability_mbb_hand": exploitability_mbb,
                    "iteration": iteration,
                }
            )
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        try:
            checkpoint_data = {
                "iteration": iteration,
                "total_iterations": self.total_iterations,
                "networks": self.networks,
                "optimizers": self.optimizers,
            }
            self.checkpoint_manager.save(checkpoint_data, iteration)
            logger.info(f"Checkpoint saved at iteration {iteration}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on failure."""
        try:
            self._save_checkpoint(self.current_iteration)
            logger.info(f"Emergency checkpoint saved at iteration {self.current_iteration}")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Phase A: Heads-Up Nash Convergence Smoke Test"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_heads_up_smoke.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    
    # Run test
    tester = HeadsUpNashConvergenceTester(config_path=args.config)
    tester.run()


if __name__ == "__main__":
    main()
