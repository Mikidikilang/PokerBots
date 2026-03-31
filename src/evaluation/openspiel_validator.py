"""
OpenSpiel CFR Validator (Phase 5)

Benchmark against OpenSpiel's reference CFR implementation on Leduc Hold'em.

Key Tests:
    1. **Convergence Rate**: Compare regret per iteration
    2. **Strategy Correctness**: Verify our strategy matches OpenSpiel
    3. **Exploitability**: Cross-check exact exploitability
    4. **Wall-Clock Performance**: Optimization speed

Reference:
    - OpenSpiel: https://github.com/deepmind/open_spiel
    - Leduc Hold'em: Small 2-player poker game (good for verification)
    - CFR Paper: Zinkevich et al. (2007)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import OpenSpiel
try:
    import pyspiel
    OPENSPIEL_AVAILABLE = True
except ImportError:
    OPENSPIEL_AVAILABLE = False
    logger.warning("OpenSpiel not installed: 'pip install open-spiel'")


# ============================================================================
# OPENSPIEL CFR REFERENCE IMPLEMENTATION
# ============================================================================

class OpenSpielCFRReference:
    """
    Run OpenSpiel's reference CFR as baseline.
    
    Provides:
        - Ground truth convergence rates
        - Known-good strategies
        - Exploitability measurements
    """
    
    def __init__(self, game_name: str = "leduc_poker"):
        """
        Args:
            game_name: OpenSpiel game name (e.g., 'leduc_poker', 'kuhn_poker')
        """
        if not OPENSPIEL_AVAILABLE:
            raise RuntimeError(
                "OpenSpiel not available. Install: pip install open-spiel"
            )
        
        self.game_name = game_name
        self.game = pyspiel.load_game(game_name)
        self.num_actions = self.game.num_distinct_actions()
        
        logger.info(f"Loaded OpenSpiel game: {game_name}")
    
    def run_cfr_iterations(
        self,
        num_iterations: int = 1000,
        verbose: bool = True,
    ) -> Dict:
        """
        Run CFR and return convergence metrics.
        
        Args:
            num_iterations: Number of CFR iterations
            verbose: Log progress
        
        Returns:
            Dict with:
                - regrets_by_iter: regret values per iteration
                - final_strategy: converged strategy
                - exploitability: final exploitability
        """
        logger.info(f"Running OpenSpiel CFR: {num_iterations} iterations")
        
        # OpenSpiel CFR implementation
        solver = pyspiel.CFRSolver(self.game)
        
        regrets = []
        start_time = time.time()
        
        for iteration in range(num_iterations):
            solver.evaluate_and_update_policy()
            
            # Get current exploitability (approximation)
            # In full OpenSpiel: use exploitability_nash() if available
            regret = solver.exploitability(self.game)
            regrets.append(regret)
            
            if verbose and (iteration + 1) % max(1, num_iterations // 10) == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Iteration {iteration+1}/{num_iterations}: "
                    f"exploit={regret:.4f}, time={elapsed:.1f}s"
                )
        
        elapsed = time.time() - start_time
        
        # Get final strategy
        final_strategy = solver.average_strategy()
        
        # Compute exploitability
        try:
            final_exploit = solver.exploitability(self.game)
        except Exception as e:
            logger.warning(f"Could not compute final exploitability: {e}")
            final_exploit = regrets[-1] if regrets else 0.0
        
        results = {
            'game': self.game_name,
            'num_iterations': num_iterations,
            'regrets_by_iteration': regrets,
            'final_exploitability': final_exploit,
            'final_strategy': final_strategy,
            'elapsed_seconds': elapsed,
            'iterations_per_second': num_iterations / elapsed if elapsed > 0 else 0,
        }
        
        logger.info(
            f"OpenSpiel CFR complete: "
            f"exploit={final_exploit:.4f}, "
            f"time={elapsed:.2f}s ({results['iterations_per_second']:.1f} iter/s)"
        )
        
        return results


# ============================================================================
# CONVERGENCE COMPARISON
# ============================================================================

@dataclass
class ConvergenceComparison:
    """
    Comparison of convergence rates between implementations.
    """
    
    iterations: List[int]
    """Iteration numbers"""
    
    openspiel_regrets: List[float]
    """OpenSpiel regret values"""
    
    our_regrets: List[float]
    """Our implementation regret values"""
    
    openspiel_iterations_per_second: float
    our_iterations_per_second: float
    
    max_regret_difference: float
    """Maximum regret difference between implementations"""
    
    convergence_rate_ratio: float
    """How much faster/slower we are (ratio of regret decrease per iter)"""
    
    def is_convergence_compatible(self, tolerance: float = 0.05) -> bool:
        """
        Check if convergence rates are similar (within tolerance).
        
        Args:
            tolerance: Acceptable difference (5% = convergence is very similar)
        
        Returns:
            True if convergence rates match within tolerance
        """
        return self.max_regret_difference < tolerance


class ConvergenceValidator:
    """
    Compare our CFR implementation against OpenSpiel reference.
    """
    
    def __init__(self):
        if not OPENSPIEL_AVAILABLE:
            logger.warning("OpenSpiel not available for validation")
    
    def compare_convergence(
        self,
        our_cfr_runner,  # Callable that runs our CFR
        num_iterations: int = 200,
        game_name: str = "leduc_poker",
    ) -> ConvergenceComparison:
        """
        Run both implementations and compare convergence.
        
        Args:
            our_cfr_runner: Function that runs our CFR and returns regrets_by_iter
            num_iterations: How many iterations to run
            game_name: OpenSpiel game name
        
        Returns:
            ConvergenceComparison with full metrics
        """
        logger.info(f"Starting convergence comparison on {game_name}")
        
        # Run OpenSpiel reference
        ref_impl = OpenSpielCFRReference(game_name)
        ref_results = ref_impl.run_cfr_iterations(num_iterations)
        
        # Run our implementation
        logger.info("Running our CFR implementation...")
        start = time.time()
        our_regrets = our_cfr_runner(num_iterations)
        our_time = time.time() - start
        
        # Align lengths (might differ if implementation stops early)
        min_len = min(len(our_regrets), len(ref_results['regrets_by_iteration']))
        our_regrets = our_regrets[:min_len]
        ref_regrets = ref_results['regrets_by_iteration'][:min_len]
        
        # Compute differences
        max_diff = max(
            abs(o - r) for o, r in zip(our_regrets, ref_regrets)
        ) if our_regrets and ref_regrets else 0.0
        
        # Convergence rate: how much regret decreases per iteration
        our_rate = (our_regrets[0] - our_regrets[-1]) / len(our_regrets) if len(our_regrets) > 1 else 0
        ref_rate = (ref_regrets[0] - ref_regrets[-1]) / len(ref_regrets) if len(ref_regrets) > 1 else 0
        
        convergence_ratio = our_rate / ref_rate if ref_rate > 0 else 1.0
        
        comparison = ConvergenceComparison(
            iterations=list(range(min_len)),
            openspiel_regrets=ref_regrets,
            our_regrets=our_regrets,
            openspiel_iterations_per_second=ref_results['iterations_per_second'],
            our_iterations_per_second=min_len / our_time if our_time > 0 else 0,
            max_regret_difference=max_diff,
            convergence_rate_ratio=convergence_ratio,
        )
        
        logger.info(f"Convergence comparison complete:")
        logger.info(f"  Max regret diff: {max_diff:.6f}")
        logger.info(f"  Convergence rate ratio: {convergence_ratio:.2f}x")
        logger.info(f"  Our speed: {comparison.our_iterations_per_second:.1f} iter/s")
        logger.info(f"  OpenSpiel speed: {comparison.openspiel_iterations_per_second:.1f} iter/s")
        
        return comparison


# ============================================================================
# STRATEGY CORRECTNESS VALIDATION
# ============================================================================

@dataclass
class StrategyValidationResult:
    """
    Results of strategy correctness validation.
    """
    
    strategy_distribution_distance: float
    """L2 distance between our strategy and OpenSpiel's"""
    
    num_actions: int
    """Number of actions in game"""
    
    num_divergent_actions: int
    """Actions where strategy differs significantly"""
    
    max_action_probability_difference: float
    """Maximum difference in action probability"""
    
    is_correct: bool = True
    """Whether strategy is compatible (within epsilon)"""


class StrategyValidator:
    """
    Validate that our strategy is correct by comparing to OpenSpiel.
    """
    
    def validate_strategy(
        self,
        our_strategy: np.ndarray,
        openspiel_strategy: np.ndarray,
        tolerance: float = 0.01,
    ) -> StrategyValidationResult:
        """
        Compare strategies for correctness.
        
        Args:
            our_strategy: Our computed strategy
            openspiel_strategy: OpenSpiel's strategy
            tolerance: Acceptable difference (1% = very strict)
        
        Returns:
            StrategyValidationResult with comparison
        """
        # Ensure same length
        min_len = min(len(our_strategy), len(openspiel_strategy))
        ours = our_strategy[:min_len]
        ref = openspiel_strategy[:min_len]
        
        # L2 distance
        l2_distance = np.linalg.norm(ours - ref)
        
        # Per-action differences
        differences = np.abs(ours - ref)
        max_diff = np.max(differences)
        num_divergent = np.sum(differences > tolerance)
        
        is_correct = l2_distance < 0.1  # L2 distance < 0.1
        
        result = StrategyValidationResult(
            strategy_distribution_distance=l2_distance,
            num_actions=min_len,
            num_divergent_actions=int(num_divergent),
            max_action_probability_difference=float(max_diff),
            is_correct=is_correct,
        )
        
        logger.info(f"Strategy validation: {result}")
        return result


# ============================================================================
# EXPLOITABILITY CROSS-CHECK
# ============================================================================

def cross_check_exploitability(
    openspiel_exploit: float,
    our_exploit: float,
    tolerance_percent: float = 5.0,
) -> bool:
    """
    Verify exploitability measurements agree (within tolerance).
    
    Args:
        openspiel_exploit: OpenSpiel's measured exploitability
        our_exploit: Our measured exploitability
        tolerance_percent: Acceptable difference (5% = pretty close)
    
    Returns:
        True if measurements agree within tolerance
    """
    if openspiel_exploit == 0:
        return our_exploit < 0.01
    
    percent_diff = abs(our_exploit - openspiel_exploit) / openspiel_exploit * 100
    
    logger.info(
        f"Exploitability cross-check: "
        f"OpenSpiel={openspiel_exploit:.6f}, "
        f"Ours={our_exploit:.6f}, "
        f"Diff={percent_diff:.2f}%"
    )
    
    return percent_diff < tolerance_percent


# ============================================================================
# FULL VALIDATION HARNESS
# ============================================================================

class FullValidationSuite:
    """
    Complete validation against OpenSpiel on reference games.
    """
    
    def __init__(self):
        self.convergence_validator = ConvergenceValidator()
        self.strategy_validator = StrategyValidator()
    
    def run_full_validation(
        self,
        our_cfr_runner,
        num_iterations: int = 200,
        games: List[str] = None,
    ) -> Dict:
        """
        Run comprehensive validation suite.
        
        Args:
            our_cfr_runner: Callable that runs our CFR
            num_iterations: Iterations per game
            games: List of game names (default: Leduc, Kuhn)
        
        Returns:
            Results dict with all validations
        """
        if games is None:
            games = ["leduc_poker", "kuhn_poker"] if OPENSPIEL_AVAILABLE else []
        
        if not OPENSPIEL_AVAILABLE:
            logger.error("OpenSpiel not available for validation")
            return {'error': 'OpenSpiel not installed'}
        
        results = {}
        
        for game_name in games:
            logger.info(f"\n{'='*60}")
            logger.info(f"Validating on {game_name}")
            logger.info(f"{'='*60}")
            
            try:
                # Convergence comparison
                comparison = self.convergence_validator.compare_convergence(
                    our_cfr_runner, num_iterations, game_name
                )
                
                results[game_name] = {
                    'convergence_compatible': comparison.is_convergence_compatible(),
                    'max_regret_difference': comparison.max_regret_difference,
                    'convergence_rate_ratio': comparison.convergence_rate_ratio,
                    'speed_ratio': comparison.our_iterations_per_second / (
                        comparison.openspiel_iterations_per_second + 1e-6
                    ),
                }
            
            except Exception as e:
                logger.error(f"Validation failed on {game_name}: {e}")
                results[game_name] = {'error': str(e)}
        
        return results


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== OpenSpiel CFR Validator ===")
    
    if not OPENSPIEL_AVAILABLE:
        print("OpenSpiel not installed. Run: pip install open-spiel")
        print("Proceeding with mock validation...")
        
        # Mock OpenSpiel reference for testing
        ref = OpenSpielCFRReference.__new__(OpenSpielCFRReference)
        ref.game_name = "mock_game"
        
        print("Mock validation complete (OpenSpiel offline)")
    else:
        # Real validation
        validator = FullValidationSuite()
        
        def mock_cfr(n_iters):
            # Mock CFR: exponentially decaying regret
            return [100 / (1 + 0.01 * i) for i in range(n_iters)]
        
        results = validator.run_full_validation(
            our_cfr_runner=mock_cfr,
            num_iterations=100,
            games=["kuhn_poker"],
        )
        
        print(f"\nValidation results: {results}")
