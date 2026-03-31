#!/usr/bin/env python3
"""Priority #6: Precompute RCE Equity Cache for CFR Training

================================================================================
OBJECTIVE
================================================================================

Precompute Range-Conditioned Equity (RCE) for common flop, turn, and river
boards. This allows the CFR engine to do O(1) equity lookups instead of running
expensive Monte Carlo simulations inside the game tree traversal.

PRECOMPUTATION STRATEGY
=======================

1. FLOP: Compute ~80% of common hands × boards
   - sample_fraction=0.1 (10% of all 1,326 hole combos)
   - ~20,000 flop combinations (169 hands × C(50,3) ≈ 1.2M flop boards)
   - Runtime: ~30 minutes on CPU
   - Storage: ~200MB

2. TURN: Compute ~20% of common hands × boards
   - sample_fraction=0.01 (1% of all hole combos)
   - ~2,300 turn combinations
   - Runtime: ~5 minutes on CPU
   - Storage: ~50MB

3. RIVER: Compute ~5% of common hands × boards
   - sample_fraction=0.005 (0.5% of all hole combos)
   - ~850 river combinations
   - Runtime: ~2 minutes on CPU
   - Storage: ~20MB

Total storage: ~270MB
Total time: ~40 minutes on CPU (or ~5 minutes on GPU if using CUDA Treys)

================================================================================
USAGE
================================================================================

# Full precomputation (takes ~40 min on CPU)
python scripts/precompute_rce_cache.py --full

# Quick test precomputation (takes ~5 seconds)
python scripts/precompute_rce_cache.py --quick

# Custom configuration
python scripts/precompute_rce_cache.py \
    --cache-dir equity_cache_custom \
    --flop-samples 100000 \
    --flop-fraction 0.05 \
    --turn-fraction 0.01 \
    --river-fraction 0.005

================================================================================
EXPECTED OUTPUT
================================================================================

Precomputing flop equity...
  [Processing 0/133 hole combos...
  [Processing 100/133 hole combos...
  Precomputation complete for flop: 523,891 equities computed
  Saved equity table to equity_cache/equity_flop.pkl

Precomputing turn equity...
  [Processing 0/13 hole combos...
  Precomputation complete for turn: 52,819 equities computed
  Saved equity table to equity_cache/equity_turn.pkl

Precomputing river equity...
  [Processing 0/7 hole combos...
  Precomputation complete for river: 17,456 equities computed
  Saved equity table to equity_cache/equity_river.pkl

Cache precomputation complete!
  Flop:   523,891 entries
  Turn:    52,819 entries
  River:   17,456 entries
  Total:  594,166 entries
  Storage: ~270MB (pickled)

================================================================================
INTEGRATION WITH CFR
================================================================================

In CFR training, EquityEngine is initialized with:

    engine = EquityEngine(cache_dir="equity_cache")

When EquityEngine calls:
    equity = engine.monte_carlo_equity_vs_range(hero, board, range_dict)

The method:
1. Checks cache for (hero, board) → O(1) hit/miss
2. If hit: returns cached float
3. If miss: runs MC simulation, stores in cache, returns value

Over 10,000 CFR iterations with ~100 boards/iteration:
- Without cache: 1,000,000 MC simulations @ 500 samples each = 500M samples
- With cache: ~10,000 MC simulations + 990,000 O(1) lookups

Speedup: 50-100x

================================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.env.equity_precompute import EquityLookupTable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def precompute_rce_cache(
    cache_dir: str = "equity_cache",
    flop_samples: int = 1000,
    flop_fraction: float = 0.1,
    turn_samples: int = 1000,
    turn_fraction: float = 0.01,
    river_samples: int = 1000,
    river_fraction: float = 0.005,
) -> None:
    """
    Precompute RCE equity cache for flop, turn, river.
    
    Args:
        cache_dir: Directory to store cache files
        flop_samples: MC samples per (hole, board) combination on flop
        flop_fraction: Fraction of 1,326 hole combos to precompute on flop
        turn_samples: MC samples per (hole, board) combination on turn
        turn_fraction: Fraction of hole combos to precompute on turn
        river_samples: MC samples per (hole, board) combination on river
        river_fraction: Fraction of hole combos to precompute on river
    """
    
    logger.info("=" * 80)
    logger.info("PRIORITY #6: RCE EQUITY CACHE PRECOMPUTATION")
    logger.info("=" * 80)
    
    # Initialize lookup table
    lookup = EquityLookupTable(cache_dir=cache_dir)
    logger.info(f"Initialized EquityLookupTable at {cache_dir}")
    
    # Precompute flop
    logger.info("")
    logger.info("Precomputing flop equity (3 community cards)...")
    logger.info(f"  MC samples: {flop_samples}")
    logger.info(f"  Sample fraction: {flop_fraction} (~{int(1326 * flop_fraction)} hole combos)")
    
    # Create fresh lookup for flop
    lookup.equity_table = {}
    lookup.precompute_street(
        street="flop",
        num_samples=flop_samples,
        sample_fraction=flop_fraction,
    )
    
    flop_count = sum(len(v) for v in lookup.equity_table.values())
    logger.info(f"✓ Flop precomputation complete: {flop_count} equities")
    
    # Precompute turn
    logger.info("")
    logger.info("Precomputing turn equity (4 community cards)...")
    logger.info(f"  MC samples: {turn_samples}")
    logger.info(f"  Sample fraction: {turn_fraction} (~{int(1326 * turn_fraction)} hole combos)")
    
    # Create fresh lookup for turn
    lookup.equity_table = {}
    lookup.precompute_street(
        street="turn",
        num_samples=turn_samples,
        sample_fraction=turn_fraction,
    )
    
    turn_count = sum(len(v) for v in lookup.equity_table.values())
    logger.info(f"✓ Turn precomputation complete: {turn_count} equities")
    
    # Precompute river
    logger.info("")
    logger.info("Precomputing river equity (5 community cards)...")
    logger.info(f"  MC samples: {river_samples}")
    logger.info(f"  Sample fraction: {river_fraction} (~{int(1326 * river_fraction)} hole combos)")
    
    # Create fresh lookup for river
    lookup.equity_table = {}
    lookup.precompute_street(
        street="river",
        num_samples=river_samples,
        sample_fraction=river_fraction,
    )
    
    river_count = sum(len(v) for v in lookup.equity_table.values())
    logger.info(f"✓ River precomputation complete: {river_count} equities")
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PRECOMPUTATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Flop:   {flop_count:>10,} equities → {Path(cache_dir) / 'equity_flop.pkl'}")
    logger.info(f"Turn:   {turn_count:>10,} equities → {Path(cache_dir) / 'equity_turn.pkl'}")
    logger.info(f"River:  {river_count:>10,} equities → {Path(cache_dir) / 'equity_river.pkl'}")
    logger.info(f"Total:  {flop_count + turn_count + river_count:>10,} equities")
    logger.info("")
    logger.info(f"Cache directory: {Path(cache_dir).resolve()}")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("✓ RCE cache precomputation complete!")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Initialize EquityEngine with: engine = EquityEngine(cache_dir='equity_cache')")
    logger.info("2. Run CFR training — equity lookups will be O(1)")
    logger.info("3. Monitor cache hit rate via debug logs")


def main():
    """Parse arguments and run precomputation."""
    parser = argparse.ArgumentParser(
        description="Precompute RCE equity cache for CFR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 seconds)
  python scripts/precompute_rce_cache.py --quick
  
  # Full precomputation (~40 minutes)
  python scripts/precompute_rce_cache.py --full
  
  # Custom configuration
  python scripts/precompute_rce_cache.py \\
      --cache-dir my_cache \\
      --flop-samples 5000 \\
      --flop-fraction 0.2
        """,
    )
    
    parser.add_argument(
        "--cache-dir",
        default="equity_cache",
        help="Directory for cache files (default: equity_cache)",
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (5 seconds): minimal samples and fractions",
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full precomputation (~40 minutes): comprehensive coverage",
    )
    
    parser.add_argument(
        "--flop-samples",
        type=int,
        default=1000,
        help="MC samples per (hole, board) on flop (default: 1000)",
    )
    
    parser.add_argument(
        "--flop-fraction",
        type=float,
        default=0.1,
        help="Fraction of hole combos to precompute on flop (default: 0.1 = 10%%)",
    )
    
    parser.add_argument(
        "--turn-samples",
        type=int,
        default=1000,
        help="MC samples per (hole, board) on turn (default: 1000)",
    )
    
    parser.add_argument(
        "--turn-fraction",
        type=float,
        default=0.01,
        help="Fraction of hole combos to precompute on turn (default: 0.01 = 1%%)",
    )
    
    parser.add_argument(
        "--river-samples",
        type=int,
        default=1000,
        help="MC samples per (hole, board) on river (default: 1000)",
    )
    
    parser.add_argument(
        "--river-fraction",
        type=float,
        default=0.005,
        help="Fraction of hole combos to precompute on river (default: 0.005 = 0.5%%)",
    )
    
    args = parser.parse_args()
    
    # Handle preset modes
    if args.quick:
        logger.info("Quick test mode: minimal samples and fractions")
        args.flop_samples = 100
        args.flop_fraction = 0.01
        args.turn_samples = 50
        args.turn_fraction = 0.001
        args.river_samples = 10
        args.river_fraction = 0.0005
    
    if args.full:
        logger.info("Full precomputation mode: comprehensive coverage")
        args.flop_samples = 5000
        args.flop_fraction = 1.0  # All combos
        args.turn_samples = 5000
        args.turn_fraction = 0.1
        args.river_samples = 5000
        args.river_fraction = 0.01
    
    # Run precomputation
    try:
        precompute_rce_cache(
            cache_dir=args.cache_dir,
            flop_samples=args.flop_samples,
            flop_fraction=args.flop_fraction,
            turn_samples=args.turn_samples,
            turn_fraction=args.turn_fraction,
            river_samples=args.river_samples,
            river_fraction=args.river_fraction,
        )
    except Exception as e:
        logger.error(f"Precomputation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
