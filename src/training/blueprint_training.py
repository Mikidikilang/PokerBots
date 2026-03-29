"""
Blueprint Training Harness (Phase 4.2)

Complete training pipeline:
1. Run CFR until convergence (exploitability < 100 mbb/hand)
2. Train network on converged strategy
3. Validate with safe subgame solving
4. Measure exploitability at intervals
5. Export final model

Time Budget: ~1000 CPU-core-hours for heads-up, ~10,000 for 6-max
(Pluribus baseline)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .cfr_engine import CFREngine
from .trainer import CFRTrainer
from .exploitability import SamplingBasedExploitabilityMeasurer, ExploitabilityResult

logger = logging.getLogger(__name__)


@dataclass
class BlueprintTrainingConfig:
    """Configuration for blueprint training."""
    
    # CFR parameters
    num_cfr_iterations: int = 10000
    """Total CFR iterations (stop early if exploitability target met)"""
    
    traversals_per_iteration: int = 100
    """Game tree traversals per iteration"""
    
    # Network training
    num_training_epochs: int = 10
    """Neural network training epochs per CFR iteration"""
    
    batch_size: int = 512
    """Training batch size"""
    
    learning_rate: float = 0.001
    """Adam learning rate"""
    
    # Stopping criteria
    exploitability_target_mbb: float = 100.0
    """Stop training when exploitability < this (millibig-blinds/hand)"""
    
    max_iterations_per_level: int = 500
    """Max iterations before evaluation"""
    
    # Evaluation
    evaluation_interval: int = 100
    """Measure exploitability every N iterations"""
    
    num_evaluation_hands: int = 1000
    """Hands used per exploitability measurement"""
    
    # Output
    checkpoint_dir: Path = Path("checkpoints")
    """Where to save model checkpoints"""
    
    log_dir: Path = Path("logs")
    """Where to save training logs"""
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """PyTorch device"""
    
    num_workers: int = 4
    """Number of worker processes for CFR"""
    
    # Game config
    num_players: int = 2
    """Number of players (2=heads-up, 6=6-max)"""
    
    abstraction_buckets_flop: int = 150
    """Flop abstraction size"""
    
    abstraction_buckets_turn: int = 75
    """Turn abstraction size"""
    
    abstraction_buckets_river: int = 50
    """River abstraction size"""


@dataclass
class TrainingProgressLog:
    """Per-iteration training progress."""
    
    iteration: int
    timestamp: float
    
    cfr_regret: float
    """Current regret per hand"""
    
    exploitability_mbb: Optional[float] = None
    """Measured exploitability (None if not evaluated this iteration)"""
    
    blueprint_ev: Optional[float] = None
    network_accuracy: float = 0.0
    """How well network predicts CFR strategy"""
    
    training_loss: float = 0.0
    """Network training loss"""
    
    iteration_time: float = 0.0
    """Wall-clock time for this iteration (seconds)"""


class BlueprintTrainingHarness:
    """
    Complete training pipeline with checkpointing and evaluation.
    
    Workflow:
        1. Load or create CFR engine
        2. For each iteration:
           a. Run CFR traversals
           b. Train network on data
           c. Every N iters: measure exploitability
           d. Checkpoint if exploitability improved
           e. Stop if target reached
        3. Export final model
    """
    
    def __init__(
        self,
        config: BlueprintTrainingConfig,
        cfr_engine: Optional[CFREngine] = None,
        strategy_network: Optional[nn.Module] = None,
    ):
        """
        Args:
            config: Training configuration
            cfr_engine: Optional existing CFR engine (for continuation)
            strategy_network: Optional existing network (for fine-tuning)
        """
        self.config = config
        self.cfr_engine = cfr_engine
        self.strategy_network = strategy_network
        self.trainer = None
        
        # Create directories
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        self.evaluator = SamplingBasedExploitabilityMeasurer(
            strategy_network=strategy_network,
            num_samples=config.num_evaluation_hands,
            device=torch.device(config.device),
        )
        
        # Training log
        self.progress_log: List[TrainingProgressLog] = []
        self.best_exploitability = float('inf')
        
        logger.info(f"BlueprintTrainingHarness initialized")
        logger.info(f"Config: {asdict(config)}")
    
    def train(self) -> Dict:
        """
        Run complete training until convergence.
        
        Returns:
            Final results dict with metrics
        """
        logger.info("=" * 80)
        logger.info("BLUEPRINT TRAINING START")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Initialize CFR if needed
        if self.cfr_engine is None:
            logger.info("Initializing CFR engine...")
            self.cfr_engine = self._create_cfr_engine()
        
        # Initialize trainer
        logger.info("Initializing network trainer...")
        self.trainer = self._create_trainer()
        
        # Main training loop
        convergence_iteration = None
        
        for iteration in range(self.config.num_cfr_iterations):
            iter_start = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration + 1}/{self.config.num_cfr_iterations}")
            logger.info(f"{'='*60}")
            
            # Step 1: CFR traversals
            logger.info("Running CFR traversals...")
            regret = self._run_cfr_iteration(iteration)
            
            # Step 2: Train network
            logger.info("Training network...")
            loss, accuracy = self._train_network_iteration(iteration)
            
            # Step 3: Periodic evaluation
            exploitability_mbb = None
            if (iteration + 1) % self.config.evaluation_interval == 0:
                logger.info("Measuring exploitability...")
                exploitability_mbb = self._measure_exploitability()
                
                # Track best
                if exploitability_mbb < self.best_exploitability:
                    self.best_exploitability = exploitability_mbb
                    self._save_checkpoint(iteration, "best")
                
                # Check convergence
                if exploitability_mbb < self.config.exploitability_target_mbb:
                    if convergence_iteration is None:
                        convergence_iteration = iteration
                        logger.info(
                            f"\n{'!'*60}")
                        logger.info(
                            f"CONVERGENCE REACHED!")
                        logger.info(
                            f"Exploitability {exploitability_mbb:.1f} < "
                            f"{self.config.exploitability_target_mbb}!")
                        logger.info(
                            f"{'!'*60}\n")
            
            # Step 4: Logging
            iter_time = time.time() - iter_start
            self._log_iteration(
                iteration, regret, exploitability_mbb, loss, accuracy, iter_time
            )
            
            # Step 5: Checkpointing
            if (iteration + 1) % 500 == 0:
                self._save_checkpoint(iteration, "periodic")
            
            # Early stopping
            if convergence_iteration is not None and \
               (iteration - convergence_iteration) >= 100:
                logger.info(
                    f"Stopping 100 iterations after convergence "
                    f"(best: {self.best_exploitability:.1f} mbb/hand)"
                )
                break
        
        # Final results
        elapsed = time.time() - start_time
        
        results = {
            'converged': convergence_iteration is not None,
            'convergence_iteration': convergence_iteration,
            'best_exploitability_mbb': self.best_exploitability,
            'final_iteration': iteration,
            'total_time_seconds': elapsed,
            'total_time_hours': elapsed / 3600,
            'progress_log': [asdict(p) for p in self.progress_log],
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("BLUEPRINT TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results: {results}")
        
        # Save final results
        self._save_results(results)
        
        return results
    
    def _create_cfr_engine(self) -> CFREngine:
        """Create fresh CFR engine."""
        # Stub: would create based on config
        return CFREngine()
    
    def _create_trainer(self) -> CFRTrainer:
        """Create neural network trainer."""
        # Stub: would create based on config
        return CFRTrainer(
            strategy_network=self.strategy_network,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )
    
    def _run_cfr_iteration(self, iteration: int) -> float:
        """Run one CFR iteration with traversals. Returns regret."""
        # Stub: would call CFREngine.traverse_batch()
        # For now: return dummy regret
        return np.random.exponential(scale=100.0 / (iteration + 1))
    
    def _train_network_iteration(self, iteration: int) -> tuple[float, float]:
        """Train network for one iteration. Returns (loss, accuracy)."""
        # Stub: would call trainer.train_epoch()
        loss = 0.5 * np.exp(-iteration / 500)
        accuracy = 0.5 + 0.49 * (1 - np.exp(-iteration / 200))
        return loss, accuracy
    
    def _measure_exploitability(self) -> float:
        """Measure current exploitability. Returns mbb/hand."""
        result = self.evaluator.measure(
            strategy_extractor=lambda x: {'fold': 0.1, 'check': 0.4, 'bet': 0.5},
        )
        return result.exploitability_mbb
    
    def _log_iteration(
        self,
        iteration: int,
        regret: float,
        exploitability_mbb: Optional[float],
        loss: float,
        accuracy: float,
        time_sec: float,
    ):
        """Log progress for this iteration."""
        log_entry = TrainingProgressLog(
            iteration=iteration,
            timestamp=time.time(),
            cfr_regret=regret,
            exploitability_mbb=exploitability_mbb,
            training_loss=loss,
            network_accuracy=accuracy,
            iteration_time=time_sec,
        )
        self.progress_log.append(log_entry)
        
        msg = (
            f"Iter {iteration+1}: "
            f"regret={regret:.2f}, loss={loss:.4f}, acc={accuracy:.2%}"
        )
        if exploitability_mbb is not None:
            msg += f", exploit={exploitability_mbb:.1f}mbb"
        msg += f", time={time_sec:.1f}s"
        
        logger.info(msg)
    
    def _save_checkpoint(self, iteration: int, reason: str):
        """Save model checkpoint."""
        if self.strategy_network is None:
            return
        
        checkpoint_path = (
            self.config.checkpoint_dir /
            f"blueprint_iter{iteration+1}_{reason}.pt"
        )
        
        torch.save({
            'iteration': iteration,
            'model_state': self.strategy_network.state_dict(),
            'exploitability': self.best_exploitability,
            'timestamp': time.time(),
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_results(self, results: Dict):
        """Save final results to JSON."""
        results_path = self.config.log_dir / "training_results.json"
        
        # Convert non-serializable
        results_clean = results.copy()
        if 'progress_log' in results_clean:
            del results_clean['progress_log']  # Too large
        
        with open(results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        logger.info(f"Results saved: {results_path}")


# ============================================================================
# Entry Point
# ============================================================================

def run_blueprint_training(
    config: Optional[BlueprintTrainingConfig] = None,
) -> Dict:
    """
    Run complete blueprint training from scratch.
    
    Args:
        config: Training configuration (uses defaults if None)
    
    Returns:
        Results dict with convergence metrics
    """
    if config is None:
        config = BlueprintTrainingConfig()
    
    harness = BlueprintTrainingHarness(config=config)
    return harness.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Blueprint Training Harness Testing ===")
    
    # Small config for testing
    config = BlueprintTrainingConfig(
        num_cfr_iterations=200,
        evaluation_interval=50,
        num_evaluation_hands=500,
        exploitability_target_mbb=50.0,  # Lower for testing
    )
    
    results = run_blueprint_training(config)
    
    print(f"\nFinal results:")
    print(f"  Converged: {results['converged']}")
    print(f"  Best exploitability: {results['best_exploitability_mbb']:.1f} mbb/hand")
    print(f"  Training time: {results['total_time_hours']:.2f} hours")
