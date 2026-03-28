"""
Nash Distance Evaluation Script (scripts/evaluate_nash.py).

Standalone CLI for offline evaluation of a model's exploitability using the
Local Best Response (LBR) evaluator. This script can be run asynchronously
from training, making it suitable for evaluation on saved checkpoints.

Usage:
    python scripts/evaluate_nash.py --checkpoint checkpoints/snapshot_iter_000500.pt
    python scripts/evaluate_nash.py --checkpoint snapshots/best_model.pt --device cuda
    python scripts/evaluate_nash.py --checkpoint model.pt --hands 100000 --config config.yaml

The script:
    1. Loads config.yaml (or specified config) to get environment and model settings
    2. Creates RLCardWrapper environment and initializes the model
    3. Loads the specified checkpoint via StateManager
    4. Runs LocalBestResponseEvaluator for the configured number of hands
    5. Logs final exploitability metrics and convergence status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Any

import torch

from src.env.action_mapper import ActionMapper
from src.env.equity import EquityCalculator
from src.env.features import ObservationBuilder, ObservationConfig
from src.env.wrappers import make_env
from src.evaluation.nash_evaluator import LocalBestResponseEvaluator, NashEvalConfig
from src.mlops.state_manager import StateManager
from src.model.networks import NetworkConfig, PokerActorCritic

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        logger.info("Config loaded from: %s", config_path)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse {config_path}: {e}") from e


# =============================================================================
# Model and Environment Initialization
# =============================================================================

def create_model(
    network_config: NetworkConfig,
    device: str | torch.device,
) -> PokerActorCritic:
    """Create and initialize the neural network model.

    Args:
        network_config: NetworkConfig object with architecture settings.
        device: PyTorch device (cpu/cuda).

    Returns:
        Initialized PokerActorCritic model.
    """
    model = PokerActorCritic(network_config)
    model.to(device)
    model.eval()  # Set to evaluation mode
    logger.info("Model created: %s", model.__class__.__name__)
    return model


def load_checkpoint(
    model: PokerActorCritic,
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> None:
    """Load model weights from checkpoint file.

    Args:
        model: PokerActorCritic instance to load weights into.
        checkpoint_path: Path to .pt checkpoint file.
        device: PyTorch device for loading (cpu/cuda).

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If checkpoint format is invalid.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Handle both full checkpoint dicts and bare state_dicts
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Full checkpoint from StateManager
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(
                    "Loaded full checkpoint from: %s (iteration %s)",
                    checkpoint_path,
                    checkpoint.get("iteration", "?"),
                )
            else:
                # Assume it's a bare state_dict
                model.load_state_dict(checkpoint)
                logger.info("Loaded state_dict from: %s", checkpoint_path)
        else:
            raise RuntimeError(f"Invalid checkpoint format: {type(checkpoint)}")

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e


# =============================================================================
# Evaluation Orchestration
# =============================================================================

def run_evaluation(
    checkpoint_path: str,
    config: dict[str, Any],
    device: str = "cpu",
    hands: int | None = None,
) -> dict[str, Any]:
    """Run Nash distance evaluation on a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration dictionary (from config.yaml).
        device: PyTorch device (cpu/cuda).
        hands: Override eval_hands from config (optional).

    Returns:
        Dictionary with evaluation results.
    """
    logger.info("=" * 80)
    logger.info("Starting Nash Distance Evaluation")
    logger.info("=" * 80)

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Extract configuration sections
    env_cfg = config.get("environment", {})
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    nash_cfg = eval_cfg.get("nash_distance", {})

    # Create environment
    logger.info("Creating environment...")
    env = make_env(env_cfg)

    # Create observation builder
    obs_config = ObservationConfig(num_players=env_cfg.get("num_players", 2))
    obs_builder = ObservationBuilder(obs_config)

    # Create action mapper
    action_mapper = ActionMapper()

    # Create equity calculator
    equity_calc = EquityCalculator()

    # Create model
    logger.info("Creating model...")
    network_config = NetworkConfig.from_dict(model_cfg, num_players=env_cfg.get("num_players", 2))
    model = create_model(network_config, device)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", checkpoint_path)
    load_checkpoint(model, checkpoint_path, device)

    # Configure Nash evaluation
    eval_hands = hands or nash_cfg.get("eval_hands", 50_000)
    target_pct = nash_cfg.get("target_pct", 0.3)

    nash_eval_config = NashEvalConfig(
        eval_hands=eval_hands,
        target_pct=target_pct,
        equity_iterations=200,  # Fixed low value for performance
        model_deterministic=True,
    )

    logger.info(
        "Nash Evaluation Config: eval_hands=%d, target_pct=%.2f%%, "
        "equity_iters=%d",
        nash_eval_config.eval_hands,
        nash_eval_config.target_pct,
        nash_eval_config.equity_iterations,
    )

    # Run evaluation
    logger.info("Running evaluation...")
    evaluator = LocalBestResponseEvaluator(
        model=model,
        env=env,
        obs_builder=obs_builder,
        action_mapper=action_mapper,
        equity_calc=equity_calc,
        config=nash_eval_config,
        device=device,
    )

    results = evaluator.run_evaluation()

    # Format results
    results_dict = {
        "checkpoint": str(checkpoint_path),
        "total_hands": results.total_hands,
        "oracle_chip_delta": float(results.oracle_chip_delta),
        "total_pot": float(results.total_pot),
        "oracle_mbb_hand": round(results.oracle_mbb_hand, 3),
        "nash_distance_pct": round(results.nash_distance_pct, 4),
        "oracle_win_rate_pct": round(results.oracle_win_rate_pct, 2),
        "target_nash_pct": nash_eval_config.target_pct,
        "is_converged": results.is_converged,
    }

    return results_dict


# =============================================================================
# CLI and Main
# =============================================================================

def main() -> None:
    """Main entry point for Nash evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate model exploitability using Local Best Response oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a checkpoint with default config
    python scripts/evaluate_nash.py --checkpoint snapshots/model_iter_500.pt

    # Evaluate with custom config and device
    python scripts/evaluate_nash.py --checkpoint model.pt --config config.yaml --device cuda

    # Override eval hands
    python scripts/evaluate_nash.py --checkpoint model.pt --hands 100000
        """,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="PyTorch device for evaluation (default: auto)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=None,
        help="Override eval_hands from config (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load config
        config = load_config(args.config)

        # Run evaluation
        results = run_evaluation(
            checkpoint_path=args.checkpoint,
            config=config,
            device=args.device,
            hands=args.hands,
        )

        # Log results
        logger.info("=" * 80)
        logger.info("Evaluation Complete")
        logger.info("=" * 80)
        for key, value in results.items():
            logger.info("%s: %s", key, value)

        # Final convergence message
        if results["is_converged"]:
            logger.info(
                "✓ SUCCESS: Nash Distance (%.4f%%) < Target (%.2f%%)",
                results["nash_distance_pct"],
                results["target_nash_pct"],
            )
        else:
            logger.info(
                "✗ NOT CONVERGED: Nash Distance (%.4f%%) >= Target (%.2f%%)",
                results["nash_distance_pct"],
                results["target_nash_pct"],
            )

        # Write output file if specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Results written to: %s", output_path)

        # Print JSON output to stdout for scripting
        print(json.dumps(results, indent=2))

        # Exit with appropriate code
        sys.exit(0 if results["is_converged"] else 1)

    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        print(
            json.dumps(
                {"error": str(e), "checkpoint": args.checkpoint},
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
