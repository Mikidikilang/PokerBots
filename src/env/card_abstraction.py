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
from typing import Dict, Tuple

import numpy as np
from abc import ABC, abstractmethod

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
    [PHASE 3.2] Lossy abstraction: group hands by equity percentile.
    
    Precomputation:
      For each (hole_cards, board):
        1. Run MC: sample 10k random opponent hands
        2. Compute equity (win % vs random)
        3. Get percentile rank
        4. Assign to bucket [0, K-1]
    
    Result: 1.3B (hole, board) unique states → 100 buckets per state
    Postflop: ~200M states (much more manageable)
    
    EQUITY PERCENTILE BUCKETING
    ===========================
    
    Equity [0.0, 1.0] → Bucket [0, K-1]
    
    Bucket assignment:
      bucket = int(equity * (num_buckets - 1))
      bucket = clamp(bucket, 0, num_buckets - 1)
    
    This gives:
      - bucket 0: weakest hands (equity ≈ 0%)
      - bucket 50: medium hands (equity ≈ 50%)
      - bucket 99: strongest hands (equity ≈ ~100%)
    
    Examples (with K=100):
      - equity=0.01 → bucket 1
      - equity=0.50 → bucket 50
      - equity=0.99 → bucket 99
    """
    
    def __init__(
        self,
        num_buckets: int = 100,
        mc_samples: int = 10000,
    ):
        """
        Args:
            num_buckets: Number of discretized equity buckets (e.g., 100)
            mc_samples: Number of MC samples per (hole, board) for equity computation
        """
        self.num_buckets = num_buckets
        self.mc_samples = mc_samples
        
        # Cache: {(hole_cards_tuple, board_tuple): bucket}
        self.bucket_cache: Dict[Tuple[Tuple[str, str], Tuple[str, ...]], int] = {}
        
        logger.info(
            f"HandStrengthBucket: {num_buckets} buckets, "
            f"{mc_samples} MC samples per hand"
        )
    
    def compute_equity_mc(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...],
        num_samples: int | None = None,
    ) -> float:
        """
        Estimate hand equity via Monte Carlo simulation.
        
        Args:
            hole_cards: Hero's cards
            board: Community cards
            num_samples: If None, use self.mc_samples
        
        Returns:
            Estimated equity (win rate) in [0, 1]
        
        NOTE: Placeholder implementation. Real version requires:
            - Enumerate all remaining cards
            - Sample opponent hands
            - Evaluate winner for each sample
            - Return win/(total_samples) ratio
        """
        if num_samples is None:
            num_samples = self.mc_samples
        
        # TODO: Implement real MC equity computation
        # For now: placeholder using stub equity value
        _hole_tuple = tuple(sorted([h for h in hole_cards]))
        _board_tuple = tuple(sorted([b for b in board]))
        
        # Dummy: return a placeholder based on hand strength
        # In production, this would run 10k simulations
        equity = np.random.uniform(0.2, 0.8)  # Placeholder
        
        return equity
    
    def get_bucket(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...],
    ) -> int:
        """
        Get equity bucket for a hand.
        
        Args:
            hole_cards: Hero's cards
            board: Community cards
        
        Returns:
            Bucket index [0, num_buckets-1]
        """
        cache_key = (tuple(sorted(hole_cards)), board)
        
        if cache_key in self.bucket_cache:
            return self.bucket_cache[cache_key]
        
        # Compute equity
        equity = self.compute_equity_mc(hole_cards, board)
        
        # Convert to bucket
        bucket = int(equity * (self.num_buckets - 1))
        bucket = max(0, min(bucket, self.num_buckets - 1))
        
        # Cache
        self.bucket_cache[cache_key] = bucket
        
        return bucket
    
    def precompute_buckets_batch(
        self,
        hole_boards: list[Tuple[Tuple[str, str], Tuple[str, ...]]],
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, ...]], int]:
        """
        Precompute buckets for a batch of (hole_cards, board) pairs.
        
        Args:
            hole_boards: List of (hole_cards, board) tuples
        
        Returns:
            Dictionary mapping (hole_cards, board) → bucket
        """
        results = {}
        for hole_cards, board in hole_boards:
            bucket = self.get_bucket(hole_cards, board)
            results[(hole_cards, board)] = bucket
        return results


# ============================================================================
# PHASE 3.1+3.2: COMBINED ABSTRACTION
# ============================================================================

class CombinedCardAbstraction(CardAbstraction):
    """
    Pipeline combining lossless + lossy abstractions.
    
    Usage:
        abstractor = CombinedCardAbstraction(num_equity_buckets=100)
        canonical_hole, canonical_board = abstractor.canonicalize_hand(
            ('As', 'Kd'),
            ('Qs', 'Tc', '9d')
        )
        bucket = abstractor.get_bucket(canonical_hole, canonical_board)
    """
    
    def __init__(self, num_equity_buckets: int = 100):
        """
        Args:
            num_equity_buckets: Number of buckets for lossy abstraction
        """
        self.suit_iso = SuitIsomorphismAbstraction()
        self.equity_bucketer = HandStrengthBucket(num_buckets=num_equity_buckets)
    
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
    ) -> int:
        """Get equity bucket for hand (requires board for postflop)."""
        if board is None:
            # Preflop: no bucketing needed (already abstracted to 169 hands)
            return 0
        return self.equity_bucketer.get_bucket(hole_cards, board)
    
    def abstract_observation(
        self,
        hole_cards: Tuple[str, str],
        board: Tuple[str, ...] | None = None,
    ) -> Dict[str, any]:
        """
        Fully abstract an observation (card + bucket).
        
        Returns:
            {
                'canonical_hole': (card, card),
                'canonical_board': (card, card, ...),
                'equity_bucket': 0-99 (or None if preflop),
                'hand_name': 'AKs' or 'AKo' (human readable)
            }
        """
        canonical_hole, canonical_board = self.canonicalize_hand(hole_cards, board)
        
        bucket = None
        if board is not None:
            bucket = self.get_bucket(canonical_hole, canonical_board)
        
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
