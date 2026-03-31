#!/usr/bin/env python3
"""
VR-DeepPDCFR+ Evaluation Script for Strategy Analysis

Evaluates a trained strategy against:
- Nash equilibrium (LBR exploitability)
- Best Response opponents
- Exploitability bounds

Usage:
    python scripts/evaluate_strategy.py --checkpoint path/to/checkpoint.pt [--output results.json]

Author: VR-DeepPDCFR+ Team
Date: March 31, 2026
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Project imports
from src.env.features import ObservationBuilder
from src.evaluation.nash_evaluator import LocalBestResponseEvaluator, NashEvalConfig
from src.model.networks import VRDeepPDCFRNetworks
from src.training.runner import RLCardWrapper

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """Evaluate trained VR-DeepPDCFR+ strategies."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        num_games: int = 10000,
    ):
        """Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: torch device
            num_games: Number of games to sample
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        self.num_games = num_games
        
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Initialize environment and builders
        self.env = RLCardWrapper(
            num_players=6,
            game_type="no-limit-holdem",
            initial_stacks=200,
            big_blind=1.0,
            small_blind=0.5,
        )
        self.obs_builder = ObservationBuilder(config=None)
        
        # Load networks
        obs_dim = self.obs_builder.get_observation_dim()
        num_actions = 14  # Standard action space
        
        self.networks: Dict[int, VRDeepPDCFRNetworks] = {}
        
        if "networks" in self.checkpoint:
            for player_id, state_dict in self.checkpoint["networks"].items():
                net = VRDeepPDCFRNetworks(
                    input_dim=obs_dim,
                    num_actions=num_actions,
                    hidden_dims=[512, 512],
                    device=self.device,
                )
                net.load_state_dict(state_dict)
                net.eval()
                self.networks[player_id] = net
        
        logger.info(f"Loaded {len(self.networks)} networks")
        
        # Initialize evaluator
        if len(self.networks) > 0:
            self.lbr_evaluator = LocalBestResponseEvaluator(
                strategy_network=self.networks[0].strategy,
                env=self.env,
                obs_builder=self.obs_builder,
                action_mapper=None,
                equity_calc=None,
                config=NashEvalConfig(),
                device=self.device,
            )
        else:
            self.lbr_evaluator = None
    
    def evaluate_exploitability(self) -> Dict[str, float]:
        """Compute LBR exploitability."""
        if self.lbr_evaluator is None:
            logger.warning("No networks loaded, skipping exploitability evaluation")
            return {}
        
        logger.info("Computing Local Best Response exploitability...")
        nash_distance, oracle_value = self.lbr_evaluator.evaluate()
        
        results = {
            "nash_distance_pct": nash_distance * 100,
            "oracle_mbb_hand": oracle_value,
        }
        
        logger.info(f"Exploitability: {nash_distance:.4f} ({nash_distance*100:.2f}%)")
        logger.info(f"Oracle value: {oracle_value:.6f} mbb/hand")
        
        return results
    
    def evaluate_game_play(self) -> Dict[str, float]:
        """Play games and evaluate win rate statistics."""
        logger.info(f"Playing {self.num_games} games...")
        
        payoffs = {i: [] for i in range(6)}
        
        for game_idx in range(self.num_games):
            obs = self.env.reset()
            done = False
            
            while not done:
                legal_actions = self.env.get_legal_actions()
                current_player = self.env.get_state()[2]
                
                if current_player < len(self.networks):
                    # Get action from network
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        action_logits = self.networks[current_player].strategy(obs_tensor)
                        action_probs = torch.softmax(action_logits, dim=-1)[0]
                        action = torch.argmax(action_probs).item()
                else:
                    # Random action for players without networks
                    action = np.random.choice(legal_actions)
                
                obs, _ = self.env.step(action)
                done = self.env.is_over()
            
            # Get payoffs
            payoffs_dict = self.env.get_payoffs()
            for player_id, payoff in payoffs_dict.items():
                payoffs[player_id].append(payoff)
            
            if (game_idx + 1) % 1000 == 0:
                logger.debug(f"Played {game_idx + 1}/{self.num_games} games")
        
        # Summarize statistics
        results = {}
        for player_id in range(6):
            if payoffs[player_id]:
                payoffs_array = np.array(payoffs[player_id])
                results[f"player_{player_id}_mean_payoff"] = float(np.mean(payoffs_array))
                results[f"player_{player_id}_std_payoff"] = float(np.std(payoffs_array))
                results[f"player_{player_id}_median_payoff"] = float(np.median(payoffs_array))
        
        return results
    
    def evaluate_network_statistics(self) -> Dict[str, float]:
        """Compute network layer statistics."""
        results = {}
        
        for player_id, network in self.networks.items():
            # Network size
            total_params = sum(p.numel() for p in network.parameters())
            results[f"player_{player_id}_total_params"] = total_params
            
            # Weight statistics
            for name, param in network.strategy.named_parameters():
                if len(param.shape) > 0:
                    results[f"player_{player_id}_strategy_{name}_mean"] = float(param.mean().item())
                    results[f"player_{player_id}_strategy_{name}_std"] = float(param.std().item())
        
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Full Strategy Evaluation")
        logger.info("=" * 80)
        
        results = {
            "checkpoint": str(self.checkpoint_path),
            "iteration": self.checkpoint.get("iteration", "unknown"),
            "timestamp": str(Path(self.checkpoint_path).stat().st_mtime),
        }
        
        # Exploitability
        logger.info("\n[1/3] Evaluating exploitability...")
        results["exploitability"] = self.evaluate_exploitability()
        
        # Game play
        logger.info("\n[2/3] Evaluating game play...")
        results["game_play"] = self.evaluate_game_play()
        
        # Network statistics
        logger.info("\n[3/3] Computing network statistics...")
        results["network_stats"] = self.evaluate_network_statistics()
        
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Complete")
        logger.info("=" * 80)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate VR-DeepPDCFR+ trained strategies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: eval_<checkpoint>.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10000,
        help="Number of games to play",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Create evaluator
    evaluator = StrategyEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_games=args.num_games,
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Save results
    output_path = args.output or f"eval_{Path(args.checkpoint).stem}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
