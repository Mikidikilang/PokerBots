"""
Phase 2: Offline Equity Precomputation & Bucket Lookup Table (equity_precompute.py)

[PHASE 2] Offline precomputation of hand equities for all (hole, board) combinations.

ARCHITECTURE
============

Purpose: Precompute equity for ~1.3B (hole cards, board) combinations offline,
then at runtime, do O(1) lookup instead of O(10k) MC simulation per hand.

Process:
  1. Enumerate all 1,326 hole card combinations (or 169 canonical hands)
  2. For each hole hand, iterate over all community card combinations
  3. Run 10,000 MC samples vs random opponent
  4. Store (hole_id, board_id) → equity in serialized lookup table (~2GB)
  5. At train time: hash_infoset → canonical hand → lookup key → equity → bucket

Storage:
  - Nested dict: lookup[hole_id][board_id] = equity (float32, 4 bytes)
  - Full Texas Hold'em: 1,326 × ~2M boards × 4 bytes ≈ 10GB (compressed ≈ 2GB)
  - Preflop only: 169 × 1 (no board) = trivial (~1KB)
  - Flop: 169 × C(50,3) ≈ 169 × 20k ≈ 3.4M entries ≈ 14MB
  - Turn: 169 × C(50,4) ≈ 169 × 230k ≈ 39M entries ≈ 150MB
  - River: 169 × C(50,5) ≈ 169 × 2.1M ≈ 350M entries ≈ 1.4GB

Total: ~1.5GB (compressed: ~300-500MB as pickle/parquet)

OPTIMIZATION: Store only postflop (flop/turn/river), not preflop
  (preflop uses suit isomorphism directly, not equity)

References:
  - Treys: https://github.com/ihendley/treys
  - Poker hand evaluation: https://en.wikipedia.org/wiki/Hand_rankings
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Configuration
# ============================================================================

DEFAULT_MC_SAMPLES = 10_000
"""Number of Monte Carlo samples per (hole, board) equity computation."""

PREFLOP_HANDS = 169
"""Number of canonical preflop hands (suit isomorphism)."""

# Rank and suit constants
RANKS = "23456789TJQKA"
SUITS = "shdc"


# ============================================================================
# Hand & Board Representation
# ============================================================================

@dataclass(frozen=True)
class CardCombo:
    """Immutable card combination."""
    rank: str  # '2'-'A'
    suit: str  # 's', 'h', 'd', 'c'
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    @staticmethod
    def from_str(card_str: str) -> "CardCombo":
        if len(card_str) != 2:
            raise ValueError(f"Invalid card: {card_str}")
        rank, suit = card_str[0].upper(), card_str[1].lower()
        return CardCombo(rank, suit)


def generate_all_cards() -> list[CardCombo]:
    """Generate all 52 unique cards."""
    return [CardCombo(r, s) for r in RANKS for s in SUITS]


def generate_all_hole_combos() -> list[Tuple[CardCombo, CardCombo]]:
    """Generate all C(52,2) = 1,326 possible hole card combinations."""
    all_cards = generate_all_cards()
    return list(combinations(all_cards, 2))


def generate_all_boards(hole_cards: set[CardCombo], num_streets: int = 3) -> list[list[CardCombo]]:
    """
    Generate all possible community board combinations for a given number of streets.
    
    Args:
        hole_cards: Set of hole cards already dealt (we exclude these)
        num_streets: Number of community cards (3=flop, 4=turn, 5=river)
    
    Returns:
        List of all possible boards (each is a list of community cards)
    """
    remaining = [c for c in generate_all_cards() if c not in hole_cards]
    boards = []
    for board_combo in combinations(remaining, num_streets):
        boards.append(list(board_combo))
    return boards


# ============================================================================
# Equity Computation (Wrapper around Treys)
# ============================================================================

class TreysEquityCalculator:
    """Fast hand equity estimation using Treys evaluator."""
    
    def __init__(self):
        """Initialize Treys evaluator."""
        try:
            from treys import Evaluator, Card as TreysCard
            self.evaluator = Evaluator()
            self.TreysCard = TreysCard
            self.available = True
            logger.info("Treys evaluator loaded successfully")
        except ImportError:
            logger.warning("Treys not available; equity computation will be slow")
            self.available = False
            self.evaluator = None
            self.TreysCard = None
    
    def card_to_treys(self, card: CardCombo) -> int:
        """Convert CardCombo to Treys card integer."""
        if not self.available:
            raise RuntimeError("Treys not available")
        card_str = f"{card.rank}{card.suit}"
        return self.TreysCard.new(card_str)
    
    def evaluate_hand(self, hole: list[CardCombo], board: list[CardCombo]) -> int:
        """Evaluate hand strength (lower = stronger in Treys)."""
        if not self.available:
            raise RuntimeError("Treys not available")
        
        hole_treys = [self.card_to_treys(c) for c in hole]
        board_treys = [self.card_to_treys(c) for c in board]
        
        return self.evaluator.evaluate(board_treys, hole_treys)
    
    def compute_equity_mc(
        self,
        hero_hole: Tuple[CardCombo, CardCombo],
        board: list[CardCombo],
        num_samples: int = DEFAULT_MC_SAMPLES,
    ) -> float:
        """
        Estimate hero equity vs random opponent hand via Monte Carlo.
        
        Args:
            hero_hole: Hero's 2 hole cards
            board: Community board cards (0-5)
            num_samples: Number of MC samples
        
        Returns:
            Equity (win rate) in [0, 1]
        """
        if not self.available:
            logger.warning("Treys not available; using dummy equity")
            return 0.5  # Fallback
        
        # Build set of used cards
        used_cards = set(hero_hole) | set(board)
        remaining_cards = [c for c in generate_all_cards() if c not in used_cards]
        
        if len(remaining_cards) < 2:
            logger.warning(f"Not enough remaining cards for MC (have {len(remaining_cards)})")
            return 0.5
        
        hero_hand = list(hero_hole)
        hero_strength = self.evaluate_hand(hero_hand, board)
        
        wins = 0
        ties = 0
        
        for _ in range(num_samples):
            # Sample 2 random opponent hole cards
            opp_hole = random.sample(remaining_cards, 2)
            opp_strength = self.evaluate_hand(opp_hole, board)
            
            if hero_strength < opp_strength:
                wins += 1
            elif hero_strength == opp_strength:
                ties += 1
        
        # Equity = wins + (ties * 0.5)
        equity = (wins + ties * 0.5) / num_samples
        return equity


# ============================================================================
# Equity Lookup Table Manager
# ============================================================================

class EquityLookupTable:
    """Manages offline-precomputed equity table."""
    
    def __init__(self, cache_dir: Path | str = "./equity_cache"):
        """
        Initialize lookup table manager.
        
        Args:
            cache_dir: Directory to store/load precomputed tables
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache: {hole_key: {board_key: equity}}
        self.equity_table: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"EquityLookupTable initialized: cache_dir={self.cache_dir}")
    
    def _hole_key(self, hole: Tuple[CardCombo, CardCombo]) -> str:
        """Generate unique key for hole cards."""
        card1, card2 = sorted([str(c) for c in hole])  # Canonical order
        return f"{card1}_{card2}"
    
    def _board_key(self, board: list[CardCombo]) -> str:
        """Generate unique key for board."""
        cards = sorted([str(c) for c in board])
        return "-".join(cards)
    
    def _street_name(self, num_board_cards: int) -> str:
        """Get street name from board card count."""
        names = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
        return names.get(num_board_cards, f"custom_{num_board_cards}")
    
    def get_cache_path(self, street: str) -> Path:
        """Get file path for cached equity table."""
        return self.cache_dir / f"equity_{street}.pkl"
    
    def add_equity(self, hole: Tuple[CardCombo, CardCombo], board: list[CardCombo], equity: float):
        """Store equity in memory cache."""
        hole_key = self._hole_key(hole)
        board_key = self._board_key(board)
        
        if hole_key not in self.equity_table:
            self.equity_table[hole_key] = {}
        
        self.equity_table[hole_key][board_key] = equity
    
    def get_equity(self, hole: Tuple[CardCombo, CardCombo], board: list[CardCombo]) -> Optional[float]:
        """Retrieve cached equity."""
        hole_key = self._hole_key(hole)
        board_key = self._board_key(board)
        
        return self.equity_table.get(hole_key, {}).get(board_key)
    
    def save(self, street: str):
        """Save cached table to disk."""
        path = self.get_cache_path(street)
        with open(path, "wb") as f:
            pickle.dump(self.equity_table, f)
        logger.info(f"Saved equity table to {path}")
    
    def load(self, street: str):
        """Load cached table from disk."""
        path = self.get_cache_path(street)
        if path.exists():
            with open(path, "rb") as f:
                self.equity_table = pickle.load(f)
            logger.info(f"Loaded equity table from {path}")
        else:
            logger.warning(f"Equity table not found: {path}")
    
    def precompute_street(
        self,
        street: str = "flop",
        num_samples: int = DEFAULT_MC_SAMPLES,
        sample_fraction: float = 1.0,
    ):
        """
        Precompute equity for an entire street.
        
        Args:
            street: "flop", "turn", or "river"
            num_samples: MC samples per (hole, board)
            sample_fraction: Fraction of (hole, board) to compute (for testing)
        """
        street_to_cards = {"flop": 3, "turn": 4, "river": 5}
        if street not in street_to_cards:
            raise ValueError(f"Unknown street: {street}")
        
        num_board_cards = street_to_cards[street]
        calc = TreysEquityCalculator()
        
        # Generate all hole combos
        hole_combos = generate_all_hole_combos()
        
        # Sample fraction if specified
        if sample_fraction < 1.0:
            hole_combos = random.sample(hole_combos, int(len(hole_combos) * sample_fraction))
        
        total_computed = 0
        
        for hole_idx, hole in enumerate(hole_combos):
            # Generate all boards for this hole
            used_cards = set(hole)
            boards = generate_all_boards(used_cards, num_board_cards)
            
            for board in boards:
                equity = calc.compute_equity_mc(hole, board, num_samples)
                self.add_equity(hole, board, equity)
                total_computed += 1
            
            if (hole_idx + 1) % 100 == 0:
                logger.info(f"Processed {hole_idx + 1}/{len(hole_combos)} hole combos, "
                           f"total equities: {total_computed}")
        
        logger.info(f"Precomputation complete for {street}: {total_computed} equities")
        self.save(street)


# ============================================================================
# Quick Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test basic functionality
    logger.info("Testing equity precomputation module...")
    
    # Create calculator
    calc = TreysEquityCalculator()
    if calc.available:
        # Test single hand
        hole = (CardCombo('A', 's'), CardCombo('K', 'h'))
        board = [CardCombo('Q', 's'), CardCombo('T', 'c'), CardCombo('9', 'd')]
        equity = calc.compute_equity_mc(hole, board, num_samples=1000)
        logger.info(f"Sample equity (AKs vs QT9): {equity:.4f}")
    else:
        logger.warning("Treys not available for testing")
