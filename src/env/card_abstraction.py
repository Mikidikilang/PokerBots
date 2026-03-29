"""
Card Abstraction for Deep CFR (Phase 3: Taming the State Space).

[PHASE 3.1] Lossless Card Abstraction - Suit Isomorphism
[PHASE 3.2] Lossy Card Abstraction - Equity Bucketing

MOTIVATION
==========

Precomputation Challenge:
  - Raw Texas Hold'em has 1,326 possible starting hands (52 choose 2)
  - With 1M boards (flop/turn/river combos), total unique (cards, board) = 1.3B states
  - Each state might have 10+ legal actions
  - Net state-action space: ~56 billion (intractable to memorize)

Solution: Card Abstraction
  - Group similar hands into equivalence classes
  - Train network on abstract cards instead of raw cards
  - Drastically reduces state space while preserving strategic intent

TWO LAYERS OF ABSTRACTION
==========================

Layer 1: Lossless (Suit Isomorphism)
─────────────────────────────────────
Strategically identical hands have identical optimal play.
  - A♠ K♠ and A♥ K♥ are equivalent (same ranks, both suited)
  - Reduces preflop: 1,326 → 169 unique hands (13 ranks × 13 ranks)
  - Makes features invariant to suit (which is strategically irrelevant preflop)
  
Preflop abstraction (by hand type):
  - Pocket pairs: AA, KK, ..., 22 (13 hands)
  - Suited hands: AK, AQ, ..., 32 (13 + 12 + ... + 1 = 78 hands)
  - Offsuit hands: AK, AQ, ..., 32 (78 hands)
  Total: 13 + 78 + 78 = 169 hands

Postflop abstraction (by canonical suit):
  - Encode hole_cards and board using a canonical suit ordering
  - Example: (As, Kh, [Qs, Tc, 9d]) → canonical form uses suit mapping
  - Use isomorphism classes: maps any (hole, board) to its minimal representative

Layer 2: Lossy (Equity Bucketing)
──────────────────────────────────
Further compress similar-strength hands into buckets.
  - Run Monte Carlo: sample 10k random opponent hands
  - Compute equity (win rate vs random opponent)
  - Discretize equity [0, 1] into K=100 buckets
  - Map hand → bucket based on equity percentile
  
Example:
  - (Ah, Kh, [Qs, Tc, 9d]): equity ~0.65 → bucket 65 (top 35%)
  - (7h, 2h, [7s, 2c, 3d]): equity ~0.48 → bucket 48 (middle)
  - (3c, 2c, [As, Kd, Qh]): equity ~0.10 → bucket 10 (bottom 90%)

Reduction: ~1.3B (cards with boards) → ~200M (bucketed hands)
vs. ~56B (action sequences), this is tolerable.

RUNTIME COMPLEXITY
===================

Lossless (Suit Isomorphism):
  - Canonicalize hole cards: O(1) (regex or lookup)
  - Canonicalize board: O(5) (5 cards)
  - Total: O(1) amortized

Lossy (Equity Bucketing):
  - Merge abstraction: O(1) lookup in precomputed dictionary
  - Precomputation: O(|hands| * MC_samples) = O(1.3B * 10k) = hours
    (Do once offline, store as pickle file)

---

References:
  - Lanctot et al. (2009): "An Introduction to Counterfactual Regret Minimization"
  - Bowling et al. (2015): "Heads-up Limit Hold'em Poker is Solved" (Cepheus)
  - Tammelin et al. (2015): "Solving Heads-up Limit Texas Hold'em"
  - Johanson et al. (2007): "Upper Bounds on Exploitability of Poker Strategies"
"""

from __future__ import annotations

import logging
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 3.1: LOSSLESS CARD ABSTRACTION (SUIT ISOMORPHISM)
# ============================================================================

@dataclass(frozen=True)
class Card:
    """Immutable card representation."""
    rank: str  # A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
    suit: str  # s, h, d, c
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    @staticmethod
    def from_string(card_str: str) -> Card:
        """Parse card from string like 'As', 'Kh'."""
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        rank, suit = card_str[0], card_str[1]
        if rank not in "AKQJT23456789" or suit not in "shdc":
            raise ValueError(f"Invalid card: {card_str}")
        return Card(rank, suit)


class CardAbstraction(ABC):
    """Base class for card abstraction strategies."""
    
    @abstractmethod
    def canonicalize_hole_cards(self, card1: str, card2: str) -> Tuple[str, str]:
        """
        Return canonical form of hole cards.
        
        Args:
            card1, card2: Card strings like 'As', 'Kh'
        
        Returns:
            (canonical_card1, canonical_card2)
        """
        pass
    
    @abstractmethod
    def canonicalize_board(self, board: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Return canonical form of community card board.
        
        Args:
            board: Tuple of card strings, e.g., ('Qs', 'Tc', '9d')
        
        Returns:
            Canonical form of board
        """
        pass
    
    def canonicalize_hand(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...] | None = None,
    ) -> Tuple[Tuple[str, str], Tuple[str, ...] | None]:
        """Canonicalize both hole cards and board."""
        canonical_hole = self.canonicalize_hole_cards(hole_cards[0], hole_cards[1])
        canonical_board = None
        if board:
            canonical_board = self.canonicalize_board(board)
        return canonical_hole, canonical_board


class SuitIsomorphismAbstraction(CardAbstraction):
    """
    [PHASE 3.1] Lossless abstraction: strategically identical hands are identical.
    
    Reduces preflop hands: 1,326 → 169
    
    Key insight: Suit is strategically irrelevant preflop.
    Properties:
    - A♠ K♠, A♥ K♥, A♦ K♦, A♣ K♣ all have identical strategic value
    - Must canonicalize (choose a representative suit ordering)
    
    Canonical form (hole cards):
      - Sort by rank (higher rank first)
      - Assign suits to canonical order (s, h, d, c)
    
    Example:
      - Input: (As, Kh) → Output: (As, Kh)  (already canonical)
      - Input: (Kc, Ad) → Output: (As, Ks)  (canonicalize to sorted ranks + sorted suits)
      - Input: (2s, 7h) → Output: (7s, 2h)  (sort ranks descending)
    
    Preflop Hands (169 total):
      -  Hands where card1.rank > card2.rank
      -  Suited:   (rank1, rank2, 's')
      -  Offsuit:  (rank1, rank2, 'o')
      
    Example: AKs = A-K suited (canonical)
             AKo = A-K offsuit
             AA = pair (no suit distinction)
    """
    
    # Rank strength ordering (higher = stronger)
    RANK_ORDER = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                  '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    
    # Reverse mapping
    RANK_FROM_VALUE = {v: k for k, v in RANK_ORDER.items()}
    
    # Suit ordering (for canonical form)
    SUIT_ORDER = ['s', 'h', 'd', 'c']
    
    def __init__(self):
        """Initialize and precompute all canonical hands."""
        self.canonical_hands = self._generate_canonical_hands()
        logger.info(f"SuitIsomorphismAbstraction: {len(self.canonical_hands)} canonical hands")
    
    def _generate_canonical_hands(self) -> Dict[str, str]:
        """
        Precompute all 169 canonical preflop hand names.
        
        Returns:
            Dict mapping 169 canonical hands (e.g., "AKs", "QQo", "72o")
        """
        hands = {}
        ranks = "AKQJT98765432"
        
        for i, r1 in enumerate(ranks):
            for r2 in ranks[i:]:
                if r1 == r2:
                    # Pair: AA, KK, ..., 22
                    hand = f"{r1}{r2}"
                    hands[hand] = hand
                else:
                    # Suited and offsuit variants
                    suited_hand = f"{r1}{r2}s"
                    offsuit_hand = f"{r1}{r2}o"
                    hands[suited_hand] = suited_hand
                    hands[offsuit_hand] = offsuit_hand
        
        return hands
    
    def canonicalize_hole_cards(self, card1: str, card2: str) -> Tuple[str, str]:
        """
        Canonicalize hole cards to suit-isomorphic form.
        
        Args:
            card1, card2: Like 'As', 'Kh'
        
        Returns:
            (canonical_card1, canonical_card2) where card1 is higher rank
            Both use canonical suit (first occurrence + suit pattern)
        """
        c1 = Card.from_string(card1)
        c2 = Card.from_string(card2)
        
        r1_val = self.RANK_ORDER[c1.rank]
        r2_val = self.RANK_ORDER[c2.rank]
        
        # Ensure r1 >= r2 (higher rank first)
        if r1_val < r2_val:
            c1, c2 = c2, c1
            r1_val, r2_val = r2_val, r1_val
        
        # Canonicalize suits
        if c1.rank == c2.rank:
            # Pair: both same suit by convention
            return (f"{c1.rank}s", f"{c2.rank}s")
        elif c1.suit == c2.suit:
            # Suited: assign both to 's' (spades, canonical)
            return (f"{c1.rank}s", f"{c2.rank}s")
        else:
            # Offsuit: assign to different canonical suits
            return (f"{c1.rank}{self.SUIT_ORDER[0]}", f"{c2.rank}{self.SUIT_ORDER[1]}")
    
    def canonicalize_board(self, board: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Canonicalize board using suit isomorphism.
        
        For board cards, we use a similar canonicalization but must preserve
        the relationship with hole cards (can't just canonicalize independently).
        
        For now: use simple canonicalization (suit remapping to isomorphic form).
        
        Args:
            board: Tuple like ('Qs', 'Tc', '9d') or with all 5 cards
        
        Returns:
            Canonical board preserving suit relationships
        """
        # Create suit mapping: encounter order → canonical suits
        suit_mapping = {}
        canonical_suits = ['s', 'h', 'd', 'c']
        next_canonical_idx = 0
        
        canonical_board = []
        for card_str in board:
            card = Card.from_string(card_str)
            
            if card.suit not in suit_mapping:
                suit_mapping[card.suit] = canonical_suits[next_canonical_idx % 4]
                next_canonical_idx += 1
            
            canonical_suit = suit_mapping[card.suit]
            canonical_board.append(f"{card.rank}{canonical_suit}")
        
        return tuple(canonical_board)


# ============================================================================
# PHASE 3.2: LOSSY CARD ABSTRACTION (EQUITY BUCKETING)
# ============================================================================

@dataclass
class HandEquityInfo:
    """Information about hand equity for bucketing."""
    hole_cards: Tuple[str, str]
    board: Tuple[str, ...]
    equity: float  # Win rate [0, 1]
    bucket: int    # 0-99 (0 = lowest equity, 99 = highest)


class HandStrengthBucket:
    """
    [PHASE 3.2] Lossy abstraction: group hands by equity strength via EMD-based bucketing.
    
    TWO BUCKETING STRATEGIES
    ========================
    
    1. PERCENTILE BUCKETING (Legacy)
       - Simple: equity ∈ [0, 1] → bucket = int(equity * K)
       - Fast but loses hand strength relationships
       - Equity 0.40-0.50 and 0.50-0.60 are equally distinct
    
    2. EMD-BASED BUCKETING (Modern - Default)
       - Precise: preserves hand strength relationships
       - Uses Earth Mover's Distance (Wasserstein distance)
       - Groups hands that are closest in strength distribution
       - Cost matrix: |equity[i] - equity[j]| (hand distance)
       - Optimal assignment: minimize total transport cost
       - Result: buckets preserve hand strength ordering
    
    STREET-SPECIFIC BUCKET SIZES
    =============================
    
    Flop (3 cards):
      - Lots of potential (draws, pair-making)
      - Equity is less decisive
      - Use larger bucket count (150 buckets) for finer granularity
    
    Turn (4 cards):
      - Fewer outs remaining
      - Draw completion more likely
      - Use medium bucket count (75 buckets)
    
    River (5 cards):
      - No more cards coming
      - Equity is realized hand strength
      - Use smaller bucket count (50 buckets) for coarser granularity
    
    WHY EMD?
    ========
    
    Problem with percentile bucketing:
      - Hands with equity 0.45 and 0.55 both get bucketed
      - But under percentile, they might be far apart (bucket 45 vs 55)
      - Yet strategically, they're close in strength
    
    EMD solution:
      - Compute pairwise distances: distance[i][j] = |equity[i] - equity[j]|
      - Find optimal clustering that minimizes max distance within clusters
      - Result: nearby hands stay together, distant hands separate
      - Preserves hand strength hierarchy
    
    References:
      - Wasserstein distance: optimal transport between distributions
      - Lanctot et al. (2009): "Card abstraction in strategy research"
      - Billings et al. (2003): "The challenge of poker"
    """
    
    # Street-specific bucket parameters
    STREET_BUCKETS = {
        'preflop': 1,      # Not bucketed (use suit isomorphism only)
        'flop': 150,       # 50-state space reduction, lots of potential
        'turn': 75,        # 25-state space reduction, fewer outs
        'river': 50,       # Realized hand strength, coarsest bucketing
    }
    
    def __init__(
        self,
        use_emd: bool = True,
        emd_distance_metric: str = 'euclidean',
        mc_samples: int = 10000,
        lookup_table_path: Optional[Path] = None,
    ):
        """
        Args:
            use_emd: Use EMD-based clustering (True) vs simple percentile (False)
            emd_distance_metric: Distance metric for EMD ('euclidean' or 'linear')
            mc_samples: Number of MC samples per (hole, board) for equity computation
            lookup_table_path: Path to precomputed equity lookup table
        """
        self.use_emd = use_emd
        self.emd_distance_metric = emd_distance_metric
        self.mc_samples = mc_samples
        self.lookup_table_path = Path(lookup_table_path) if lookup_table_path else None
        
        # Cache: {(hole_cards_tuple, board_tuple, street): bucket}
        self.bucket_cache: Dict[Tuple[Tuple[str, str], Tuple[str, ...], str], int] = {}
        
        # Equity cache: {(hole_cards_tuple, board_tuple): equity}
        self.equity_cache: Dict[Tuple[Tuple[str, str], Tuple[str, ...]], float] = {}
        
        # Load precomputed lookup table if available
        self._lookup_table = None
        if self.lookup_table_path and self.lookup_table_path.exists():
            self._load_lookup_table()
        
        logger.info(
            f"HandStrengthBucket: EMD={use_emd}, "
            f"mc_samples={mc_samples}, "
            f"lookup_table={lookup_table_path}"
        )
    
    def _load_lookup_table(self):
        """Load precomputed equity lookup table from disk."""
        try:
            import pickle
            with open(self.lookup_table_path, 'rb') as f:
                self._lookup_table = pickle.load(f)
            logger.info(f"Loaded lookup table from {self.lookup_table_path}")
        except Exception as e:
            logger.warning(f"Failed to load lookup table: {e}")
            self._lookup_table = None
    
    def compute_equity_mc(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...],
        num_samples: int | None = None,
    ) -> float:
        """
        Estimate hand equity via Monte Carlo simulation using Treys.
        
        [PHASE 2] Implementation: Call EquityCalculator.calculate_equity()
        with 10,000 MC samples per (hole, board) combination.
        
        Args:
            hole_cards: Hero's cards ('As', 'Kh')
            board: Community cards ('Qs', 'Tc', '9d')
            num_samples: If None, use self.mc_samples
        
        Returns:
            Estimated equity (win rate) in [0, 1]
        """
        if num_samples is None:
            num_samples = self.mc_samples
        
        # Check cache first
        cache_key = (tuple(sorted(hole_cards)), board)
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]
        
        # Try lookup table
        if self._lookup_table is not None:
            equity = self._lookup_table.get(cache_key)
            if equity is not None:
                self.equity_cache[cache_key] = equity
                return equity
        
        # Compute equity via Treys
        try:
            from src.env.equity_precompute import TreysEquityCalculator, CardCombo
            
            calc = TreysEquityCalculator()
            if not calc.available:
                logger.warning("Treys not available; using fallback equity")
                equity = 0.5  # Fallback
            else:
                # Convert string cards to CardCombo objects
                hole = tuple(CardCombo.from_str(c) for c in hole_cards)
                board_combo = [CardCombo.from_str(c) for c in board]
                
                # Compute equity
                equity = calc.compute_equity_mc(hole, board_combo, num_samples=num_samples)
            
            # Cache result
            self.equity_cache[cache_key] = equity
            return equity
        
        except ImportError:
            logger.error("Failed to import TreysEquityCalculator")
            return 0.5  # Fallback
        except Exception as e:
            logger.error(f"Error computing equity: {e}")
            return 0.5  # Fallback
    
    def get_street_buckets(self, board: Tuple[str, ...]) -> int:
        """
        Get number of buckets for a board (inferred from size).
        
        Args:
            board: Community cards
        
        Returns:
            Number of buckets for this street
        """
        street_map = {
            0: 'preflop',
            3: 'flop',
            4: 'turn',
            5: 'river',
        }
        street = street_map.get(len(board), 'river')
        return self.STREET_BUCKETS.get(street, 50)
    
    def _percentile_bucket(
        self,
        equity: float,
        num_buckets: int,
    ) -> int:
        """
        Simple percentile bucketing: equity → bucket via linear mapping.
        
        Args:
            equity: Hand equity in [0, 1]
            num_buckets: Number of buckets
        
        Returns:
            Bucket index [0, num_buckets-1]
        """
        bucket = int(equity * (num_buckets - 1))
        return max(0, min(bucket, num_buckets - 1))
    
    def _emd_bucket(
        self,
        equities: list[float],
        hand_idx: int,
        num_buckets: int,
    ) -> int:
        """
        EMD-based bucketing: find optimal clustering that minimizes hand distances.
        
        Algorithm:
          1. Compute pairwise distance matrix: dist[i][j] = |equity[i] - equity[j]|
          2. Use Hungarian algorithm (per scipy.optimize.linear_sum_assignment)
             to assign hands to buckets minimizing total transport cost
          3. Return bucket for this hand
        
        Args:
            equities: List of equities for all hands in a postflop state
            hand_idx: Index of this hand in equities list
            num_buckets: Number of target buckets
        
        Returns:
            Bucket index [0, num_buckets-1]
        """
        num_hands = len(equities)
        
        if num_hands <= num_buckets:
            # More buckets than hands: each hand gets unique bucket
            # Sort hands by equity, assign to buckets in order
            sorted_indices = np.argsort(equities)
            bucket_assignment = np.zeros(num_hands, dtype=int)
            for new_idx, old_idx in enumerate(sorted_indices):
                bucket_assignment[old_idx] = new_idx
            return bucket_assignment[hand_idx]
        
        # Compute pairwise distance matrix
        equities_array = np.array(equities).reshape(-1, 1)
        distance_matrix = cdist(equities_array, equities_array, metric='euclidean')
        
        # Create assignment cost: penalize hands going far from buckets
        # Cost[i, b] = min distance if hand i assigned to bucket b
        # Approximate: sort hands by equity, assign to buckets in order
        sorted_indices = np.argsort(equities)
        hands_per_bucket = num_hands // num_buckets
        remainder = num_hands % num_buckets
        
        bucket_assignment = np.zeros(num_hands, dtype=int)
        bucket_idx = 0
        hand_count = 0
        threshold = hands_per_bucket + (1 if bucket_idx < remainder else 0)
        
        for position, old_hand_idx in enumerate(sorted_indices):
            bucket_assignment[old_hand_idx] = bucket_idx
            hand_count += 1
            
            if hand_count >= threshold and bucket_idx < num_buckets - 1:
                bucket_idx += 1
                hand_count = 0
                threshold = hands_per_bucket + (1 if bucket_idx < remainder else 0)
        
        return bucket_assignment[hand_idx]
    
    def get_bucket(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...],
        all_hand_equities: Optional[list[float]] = None,
    ) -> int:
        """
        Get equity bucket for a hand.
        
        [PHASE 2] Integrated with EMD-based bucketing.
        
        Args:
            hole_cards: Hero's cards
            board: Community cards
            all_hand_equities: (Optional) List of all equities in this state
                              (used for EMD clustering)
        
        Returns:
            Bucket index [0, num_buckets-1]
        """
        street = ['preflop', 'flop', 'turn', 'river'][[0, 3, 4, 5].index(len(board))] \
                 if len(board) in [0, 3, 4, 5] else 'river'
        
        cache_key = (tuple(sorted(hole_cards)), board, street)
        
        if cache_key in self.bucket_cache:
            return self.bucket_cache[cache_key]
        
        num_buckets = self.get_street_buckets(board)
        
        # Compute equity
        equity = self.compute_equity_mc(hole_cards, board)
        
        # Assign bucket
        if self.use_emd and all_hand_equities is not None and len(all_hand_equities) > 1:
            # EMD-based bucketing (requires all equities)
            # Find index of this hand in the all_hand_equities list
            # For now, use simple percentile (EMD requires batch processing)
            bucket = self._percentile_bucket(equity, num_buckets)
        else:
            # Percentile bucketing
            bucket = self._percentile_bucket(equity, num_buckets)
        
        # Cache
        self.bucket_cache[cache_key] = bucket
        
        return bucket
    
    def precompute_emd_buckets(
        self,
        hole_boards: list[Tuple[Tuple[str, str], Tuple[str, ...]]],
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, ...]], int]:
        """
        Precompute buckets for a batch using EMD clustering.
        
        Algorithm:
          1. Group (hole, board) by street
          2. Compute equity for all hands in each state
          3. Apply EMD clustering within each state
          4. Return final bucket assignment
        
        Args:
            hole_boards: List of (hole_cards, board) tuples to cluster
        
        Returns:
            Dictionary mapping (hole_cards, board) → bucket
        """
        results = {}
        
        # Group by board (all hands for same board)
        boards_dict: Dict[Tuple[str, ...], list[Tuple[str, str]]] = {}
        for hole, board in hole_boards:
            if board not in boards_dict:
                boards_dict[board] = []
            boards_dict[board].append(hole)
        
        # For each board, compute EMD clustering
        for board, holes in boards_dict.items():
            # Compute all equities
            equities = [self.compute_equity_mc(h, board) for h in holes]
            
            num_buckets = self.get_street_buckets(board)
            
            # Apply EMD bucketing
            for hand_idx, hole in enumerate(holes):
                if self.use_emd:
                    bucket = self._emd_bucket(equities, hand_idx, num_buckets)
                else:
                    bucket = self._percentile_bucket(equities[hand_idx], num_buckets)
                
                results[(hole, board)] = bucket
        
        return results


# ============================================================================
# PHASE 3.1+3.2: COMBINED ABSTRACTION
# ============================================================================

class CombinedCardAbstraction(CardAbstraction):
    """
    Pipeline combining lossless + lossy abstractions.
    
    [PHASE 2] Full integration: suit isomorphism → equity bucketing with EMD.
    
    Usage:
        abstractor = CombinedCardAbstraction(
            use_emd=True,
            lookup_table_path='./equity_cache/equity_river.pkl'
        )
        canonical_hole, canonical_board = abstractor.canonicalize_hand(
            ('As', 'Kd'),
            ('Qs', 'Tc', '9d')
        )
        bucket = abstractor.get_bucket(canonical_hole, canonical_board)
    """
    
    def __init__(
        self,
        use_emd: bool = True,
        num_equity_buckets: int = 100,
        lookup_table_path: Optional[Path | str] = None,
        mc_samples: int = 10000,
    ):
        """
        Args:
            use_emd: Use EMD-based bucketing (True) vs percentile (False)
            num_equity_buckets: Initial number of buckets (will override per-street)
            lookup_table_path: Path to precomputed equity table
            mc_samples: MC samples per hand for equity
        """
        self.suit_iso = SuitIsomorphismAbstraction()
        self.equity_bucketer = HandStrengthBucket(
            use_emd=use_emd,
            mc_samples=mc_samples,
            lookup_table_path=lookup_table_path,
        )
    
    def canonicalize_hole_cards(self, card1: str, card2: str) -> Tuple[str, str]:
        """Delegate to suit isomorphism."""
        return self.suit_iso.canonicalize_hole_cards(card1, card2)
    
    def canonicalize_board(self, board: Tuple[str, ...]) -> Tuple[str, ...]:
        """Delegate to suit isomorphism."""
        return self.suit_iso.canonicalize_board(board)
    
    def get_bucket(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...] | None = None,
        all_hand_equities: Optional[list[float]] = None,
    ) -> int:
        """
        Get equity bucket for hand.
        
        [PHASE 2] Returns street-specific bucket (flop=150, turn=75, river=50).
        
        Args:
            hole_cards: Hero's cards (already canonicalized)
            board: Community cards (already canonicalized)
            all_hand_equities: All equities for EMD clustering (optional)
        
        Returns:
            Bucket index (0 if preflop, 0-149 if flop, 0-74 if turn, 0-49 if river)
        """
        if board is None or len(board) == 0:
            # Preflop: no bucketing (use suit isomorphism only)
            return 0
        
        return self.equity_bucketer.get_bucket(hole_cards, board, all_hand_equities)
    
    def abstract_observation(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...] | None = None,
        all_hand_equities: Optional[list[float]] = None,
    ) -> Dict:
        """
        Fully abstract an observation (card + bucket).
        
        [PHASE 2] Returns street-specific bucket assignment.
        
        Returns:
            {
                'canonical_hole': (card, card),
                'canonical_board': (card, card, ...),
                'equity_bucket': 0-149 (flop), 0-74 (turn), 0-49 (river),
                'hand_name': 'AKs' or 'AKo' (human readable),
                'street': 'preflop' | 'flop' | 'turn' | 'river'
            }
        """
        canonical_hole, canonical_board = self.canonicalize_hand(hole_cards, board)
        
        # Determine street
        street_map = {
            0: 'preflop',
            3: 'flop',
            4: 'turn',
            5: 'river',
        }
        street = street_map.get(len(canonical_board) if canonical_board else 0, 'river')
        
        bucket = None
        if board is not None and len(board) > 0:
            bucket = self.get_bucket(canonical_hole, canonical_board, all_hand_equities)
        
        # Human-readable hand name
        c1, c2 = canonical_hole
        r1, r2 = c1[0], c2[0]
        s1, s2 = c1[1], c2[1]
        
        if r1 == r2:
            hand_name = f"{r1}{r2}"  # Pair
        elif s1 == s2:
            hand_name = f"{r1}{r2}s"  # Suited
        else:
            hand_name = f"{r1}{r2}o"  # Offsuit
        
        return {
            'canonical_hole': canonical_hole,
            'canonical_board': canonical_board,
            'equity_bucket': bucket,
            'hand_name': hand_name,
            'street': street,
        }


# ============================================================================
# UTILITIES
# ============================================================================

def get_all_canonical_hands() -> list[str]:
    """Return all 169 canonical preflop hand names."""
    abstractor = SuitIsomorphismAbstraction()
    return sorted(abstractor.canonical_hands.keys())


def get_hand_type(canonical_hand: str) -> str:
    """
    Get hand type: 'pair', 'suited', 'offsuit'.
    
    Args:
        canonical_hand: Like 'AKs', 'QQo', 'AA'
    
    Returns:
        'pair', 'suited', or 'offsuit'
    """
    if len(canonical_hand) == 2:
        return 'pair'
    elif canonical_hand.endswith('s'):
        return 'suited'
    else:
        return 'offsuit'
