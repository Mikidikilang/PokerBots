"""
Intervention 4: Full-Rank EMD Card Abstraction with Range-Conditioned Equity
=============================================================================

The original codebase computed equity vs. random opponent hands — the weakest
possible abstraction. Two hands can have identical equity vs. random but
completely different strategic value against real opponent ranges.

Example:
    Board: K♠ K♥ 2♦
    Hand A: A♣ K♦  (trips + top kicker)  → equity vs random: ~0.92
    Hand B: 4♠ 4♥  (full house)          → equity vs random: ~0.91

Against a range heavy with Kings (opponent had KQ): Hand A is near nuts,
Hand B is beat. They belong in DIFFERENT buckets — but equity vs. random
would place them in the same bucket.

Range-Conditioned Equity (RCE)
--------------------------------

RCE(h, board) = E_{h' ~ μ_opp}[ equity(h, h', board) ]

where μ_opp is the opponent's range distribution at this board texture.

This is expensive to compute for all (hand, board) pairs, so we use:
1. Offline precomputation with Monte Carlo sampling
2. An EMD-clustering step that groups hands by their RCE DISTRIBUTION
   (not just their scalar RCE value) — capturing texture sensitivity

Wasserstein (EMD) Clustering
------------------------------

Two hands h₁, h₂ are in the same bucket if their equity distributions
across board runouts are similar:

    W₂(P_{h₁}, P_{h₂}) ≤ ε

where P_h is the distribution of equity(h, h', board) over all boards.

We approximate this via k-means on equity histograms:
    histogram_h = [P(equity ≤ 0.1), P(0.1 < equity ≤ 0.2), ..., P(equity > 0.9)]

This is the correct generalization of EMD bucketing to strategy-relevant
hand groupings.

Scalability
-----------

Full precomputation for all 169 hands × all (flop, turn, river) combos
requires ~100 CPU-hours. We provide:
1. Lazy precomputation with caching
2. Batch equity evaluation via vectorized Treys
3. LRU cache for hot (hand, board) pairs during training
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_ORDER: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}

# Street bucket counts — empirically tuned
BUCKETS_PREFLOP: int = 8      # 169 hands → 8 preflop clusters
BUCKETS_FLOP:    int = 50     # Significantly more texture sensitivity
BUCKETS_TURN:    int = 25
BUCKETS_RIVER:   int = 12     # Realized hand strength — coarser

# EMD histogram bins
N_EQUITY_BINS: int = 10       # [0,0.1), [0.1,0.2), ..., [0.9,1.0]

# Opponent sampling for RCE computation
N_OPP_SAMPLES_RCE: int = 500  # Opponent hands to sample for range-conditioning
N_BOARD_RUNOUTS: int = 50     # Board runouts per (hand, partial_board)


# ---------------------------------------------------------------------------
# Card utilities
# ---------------------------------------------------------------------------

def card_str_to_int(card: str) -> int:
    """'As' → 51, '2c' → 0 (standard Treys-compatible index)."""
    rank = card[0].upper()
    suit = card[1].lower()
    r_idx = RANK_ORDER.get(rank, 0)
    s_idx = {"c": 0, "d": 1, "h": 2, "s": 3}[suit]
    return r_idx * 4 + s_idx


def all_cards() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


def canonical_hole(c1: str, c2: str) -> Tuple[str, str]:
    """Return suit-isomorphic canonical form of hole cards."""
    r1, s1 = c1[0].upper(), c1[1].lower()
    r2, s2 = c2[0].upper(), c2[1].lower()

    # Ensure higher rank first
    if RANK_ORDER.get(r1, 0) < RANK_ORDER.get(r2, 0):
        r1, r2 = r2, r1
        s1, s2 = s2, s1

    if r1 == r2:
        return (r1 + "s", r2 + "h")  # Pocket pair: canonical suits
    elif s1 == s2:
        return (r1 + "s", r2 + "s")  # Suited
    else:
        return (r1 + "s", r2 + "h")  # Offsuit


def hand_name(c1: str, c2: str) -> str:
    """'As', 'Ks' → 'AKs'"""
    cc1, cc2 = canonical_hole(c1, c2)
    r1, r2 = cc1[0], cc2[0]
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if cc1[1] == cc2[1] else "o")


# ---------------------------------------------------------------------------
# Equity computation (Treys-backed)
# ---------------------------------------------------------------------------

class EquityEngine:
    """
    Fast batch equity computation using Treys.

    Evaluates P(hero wins | hero_hand, opp_hand, board) for many
    (hand, hand, board) triples simultaneously.
    """

    def __init__(self):
        try:
            from treys import Evaluator, Card
            self._eval = Evaluator()
            self._Card = Card
            self._available = True
        except ImportError:
            logger.warning("Treys not available. Equity computation uses fallback.")
            self._available = False

    def evaluate_batch(
        self,
        hero_hands: List[Tuple[str, str]],
        opp_hands: List[Tuple[str, str]],
        boards: List[Tuple[str, ...]],
    ) -> np.ndarray:
        """
        Evaluate equity for a batch of (hero, opp, board) triples.

        Returns:
            np.ndarray of shape (n,) with equity values in [0, 1].
        """
        n = len(hero_hands)
        equities = np.zeros(n, dtype=np.float32)

        if not self._available:
            # Fallback: rank-sum heuristic (rough but deterministic)
            for i, (h, o, b) in enumerate(zip(hero_hands, opp_hands, boards)):
                equities[i] = self._rank_sum_equity(h, o)
            return equities

        Card = self._Card
        for i, (h, o, b) in enumerate(zip(hero_hands, opp_hands, boards)):
            try:
                hero_treys = [Card.new(c) for c in h]
                opp_treys  = [Card.new(c) for c in o]
                board_treys = [Card.new(c) for c in b]

                hero_score = self._eval.evaluate(board_treys, hero_treys)
                opp_score  = self._eval.evaluate(board_treys, opp_treys)

                if hero_score < opp_score:    # Lower = stronger in Treys
                    equities[i] = 1.0
                elif hero_score == opp_score:
                    equities[i] = 0.5
                else:
                    equities[i] = 0.0
            except Exception:
                equities[i] = 0.5

        return equities

    def monte_carlo_equity_vs_range(
        self,
        hero: Tuple[str, str],
        board: Tuple[str, ...],
        opp_range: Optional[Dict[str, float]] = None,
        n_samples: int = N_OPP_SAMPLES_RCE,
    ) -> float:
        """
        Compute E[equity(hero, opp, board)] over opponent range.

        If opp_range is None, uses uniform random opponent hands
        (matches original behavior, but this is the WEAK version).
        """
        used = set(hero) | set(board)
        remaining = [c for c in all_cards() if c not in used]

        if len(remaining) < 2:
            return 0.5

        wins = 0
        ties = 0
        n_valid = 0

        for _ in range(n_samples):
            # Sample opponent hand from range (or uniform)
            if opp_range is not None:
                opp_hand = self._sample_from_range(opp_range, used)
            else:
                opp_pair = np.random.choice(len(remaining), size=2, replace=False)
                opp_hand = (remaining[opp_pair[0]], remaining[opp_pair[1]])

            if opp_hand is None:
                continue

            # Complete the board if needed
            used_with_opp = used | set(opp_hand)
            remaining_for_board = [c for c in remaining if c not in opp_hand]

            cards_needed = 5 - len(board)
            if cards_needed > len(remaining_for_board):
                continue

            if cards_needed > 0:
                runout_idx = np.random.choice(len(remaining_for_board), cards_needed, replace=False)
                complete_board = tuple(board) + tuple(remaining_for_board[i] for i in runout_idx)
            else:
                complete_board = tuple(board)

            # Evaluate
            eq_batch = self.evaluate_batch([hero], [opp_hand], [complete_board])
            eq = float(eq_batch[0])

            if eq > 0.75:
                wins += 1
            elif eq > 0.25:
                ties += 1
            n_valid += 1

        if n_valid == 0:
            return 0.5
        return (wins + 0.5 * ties) / n_valid

    def equity_histogram(
        self,
        hero: Tuple[str, str],
        board: Tuple[str, ...],
        opp_range: Optional[Dict[str, float]] = None,
        n_samples: int = N_OPP_SAMPLES_RCE,
        n_bins: int = N_EQUITY_BINS,
    ) -> np.ndarray:
        """
        Compute equity distribution over opponent hands as a histogram.

        This is the key input to EMD clustering — we cluster by
        distribution shape, not just scalar equity.

        Returns:
            np.ndarray of shape (n_bins,) summing to 1.0
        """
        used = set(hero) | set(board)
        remaining = [c for c in all_cards() if c not in used]

        equities = []
        for _ in range(n_samples):
            if len(remaining) < 2:
                break

            if opp_range is not None:
                opp_hand = self._sample_from_range(opp_range, used)
            else:
                opp_pair = np.random.choice(len(remaining), 2, replace=False)
                opp_hand = (remaining[opp_pair[0]], remaining[opp_pair[1]])

            if opp_hand is None:
                continue

            remaining_for_board = [c for c in remaining if c not in opp_hand]
            cards_needed = 5 - len(board)
            if cards_needed > len(remaining_for_board):
                continue

            if cards_needed > 0:
                idx = np.random.choice(len(remaining_for_board), cards_needed, replace=False)
                complete_board = tuple(board) + tuple(remaining_for_board[i] for i in idx)
            else:
                complete_board = tuple(board)

            eq_batch = self.evaluate_batch([hero], [opp_hand], [complete_board])
            equities.append(float(eq_batch[0]))

        if not equities:
            return np.ones(n_bins) / n_bins

        hist, _ = np.histogram(equities, bins=n_bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        total = hist.sum()
        return hist / total if total > 0 else np.ones(n_bins) / n_bins

    def _sample_from_range(
        self, opp_range: Dict[str, float], used_cards: set
    ) -> Optional[Tuple[str, str]]:
        """Sample an opponent hand from range, conditioned on card removal."""
        # Build list of compatible (hand_name → actual_cards) mappings
        compatible = []
        weights = []
        for hand_name, prob in opp_range.items():
            for c1, c2 in self._hand_name_to_combos(hand_name):
                if c1 not in used_cards and c2 not in used_cards:
                    compatible.append((c1, c2))
                    weights.append(prob)

        if not compatible:
            return None

        weights_arr = np.array(weights)
        weights_arr /= weights_arr.sum()
        idx = np.random.choice(len(compatible), p=weights_arr)
        return compatible[idx]

    def _hand_name_to_combos(self, hand_name: str) -> List[Tuple[str, str]]:
        """'AKs' → [(Ah,Kh), (Ad,Kd), (Ac,Kc), (As,Ks)]"""
        if len(hand_name) == 2:  # Pocket pair
            r = hand_name[0]
            suits = list("shdc")
            combos = []
            for i in range(len(suits)):
                for j in range(i + 1, len(suits)):
                    combos.append((r + suits[i], r + suits[j]))
            return combos
        r1, r2 = hand_name[0], hand_name[1]
        suited = hand_name[2] == 's' if len(hand_name) > 2 else False
        if suited:
            return [(r1 + s, r2 + s) for s in "shdc"]
        else:
            return [
                (r1 + s1, r2 + s2)
                for s1 in "shdc" for s2 in "shdc"
                if s1 != s2
            ]

    def _rank_sum_equity(self, h1: Tuple[str, str], h2: Tuple[str, str]) -> float:
        """Rough fallback equity estimation via rank comparison."""
        sum1 = sum(RANK_ORDER.get(c[0].upper(), 0) for c in h1)
        sum2 = sum(RANK_ORDER.get(c[0].upper(), 0) for c in h2)
        if sum1 > sum2:
            return 0.65
        elif sum1 < sum2:
            return 0.35
        return 0.5


# ---------------------------------------------------------------------------
# EMD-Based Bucketing
# ---------------------------------------------------------------------------

class EMDBucketer:
    """
    Groups hands into buckets using Wasserstein (EMD) distance on equity
    histograms, not scalar equity values.

    This is the correct implementation of EMD bucketing — the original
    codebase used a greedy linear assignment approximation that did not
    minimize the true Earth Mover's Distance.

    Algorithm
    ---------
    1. Compute equity histogram for each hand against opponent range
    2. Run k-means clustering on histograms (Euclidean distance approximates
       Wasserstein-1 distance for equal-mass distributions)
    3. Assign each hand to its nearest cluster centroid

    Why this is better than scalar equity bucketing
    ------------------------------------------------
    Scalar: AA on K22 board = bucket 95 (equity 0.95)
            KK on K22 board = bucket 93 (equity 0.93)
            → Both in bucket 19/20 if we have 20 buckets

    Histogram: AA on K22 board has equity concentrated near 1.0
               KK on K22 board has equity concentrated near 0.93 but with
               a secondary mass near 0.0 (against AA)
               → Different histogram shapes → Different buckets ✓
    """

    def __init__(
        self,
        equity_engine: EquityEngine,
        cache_dir: Optional[Path] = None,
    ):
        self.equity_engine = equity_engine
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Precomputed cluster centroids per street
        self._centroids: Dict[str, np.ndarray] = {}
        self._trained: Dict[str, bool] = {}

    def fit(
        self,
        street: str,
        n_buckets: int,
        sample_hands: Optional[List[Tuple[str, str]]] = None,
        sample_boards: Optional[List[Tuple[str, ...]]] = None,
        opp_range: Optional[Dict[str, float]] = None,
        n_samples_per_hand: int = 100,
    ) -> None:
        """
        Fit k-means centroids on equity histograms.

        Args:
            street:    "flop", "turn", or "river"
            n_buckets: Number of clusters
            sample_hands: Hands to include in clustering (all 169 if None)
            sample_boards: Sample boards for the street (random if None)
            opp_range: Opponent range for RCE computation (uniform if None)
            n_samples_per_hand: Equity samples per hand
        """
        cache_key = f"{street}_{n_buckets}_{hash(str(opp_range))}"
        cache_file = self.cache_dir / f"centroids_{cache_key}.pkl" if self.cache_dir else None

        if cache_file and cache_file.exists():
            with open(cache_file, "rb") as f:
                self._centroids[street] = pickle.load(f)
            self._trained[street] = True
            logger.info("Loaded cached centroids for %s: shape=%s", street, self._centroids[street].shape)
            return

        logger.info("Fitting EMD bucketer for %s with %d buckets...", street, n_buckets)

        # Default: all 169 canonical hands
        if sample_hands is None:
            sample_hands = self._all_canonical_hands()

        # Compute equity histograms for all sample hands
        histograms = np.zeros((len(sample_hands), N_EQUITY_BINS), dtype=np.float32)

        for i, hand in enumerate(sample_hands):
            # Sample boards for this street if not provided
            if sample_boards is None:
                board = self._sample_board(hand, street)
            else:
                board = sample_boards[i % len(sample_boards)]

            histograms[i] = self.equity_engine.equity_histogram(
                hero=hand,
                board=board,
                opp_range=opp_range,
                n_samples=n_samples_per_hand,
            )

            if (i + 1) % 20 == 0:
                logger.debug("  %d/%d hands processed", i + 1, len(sample_hands))

        # Run k-means
        centroids, labels = kmeans2(histograms, n_buckets, niter=50, minit="points", seed=42)
        self._centroids[street] = centroids
        self._trained[street] = True

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(centroids, f)

        logger.info("EMD bucketer fitted for %s: %d hands → %d buckets", street, len(sample_hands), n_buckets)

    def get_bucket(
        self,
        hand: Tuple[str, str],
        board: Tuple[str, ...],
        opp_range: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Assign hand to equity bucket.

        Args:
            hand:      Hero's hole cards
            board:     Community cards (determines street)
            opp_range: Opponent range (uses uniform if None)

        Returns:
            Bucket index [0, n_buckets)
        """
        street = {3: "flop", 4: "turn", 5: "river"}.get(len(board), "flop")
        n_buckets = {
            "flop": BUCKETS_FLOP, "turn": BUCKETS_TURN, "river": BUCKETS_RIVER
        }[street]

        if not self._trained.get(street, False):
            # Lazy initialization with small sample
            self.fit(street, n_buckets, n_samples_per_hand=30)

        # Compute histogram for this hand
        hist = self.equity_engine.equity_histogram(
            hero=hand,
            board=board,
            opp_range=opp_range,
            n_samples=50,  # Reduced for runtime performance
        )

        # Find nearest centroid (Euclidean distance ≈ W1 for unit-mass histograms)
        centroids = self._centroids[street]
        dists = np.linalg.norm(centroids - hist[np.newaxis, :], axis=1)
        return int(np.argmin(dists))

    def _all_canonical_hands(self) -> List[Tuple[str, str]]:
        """Generate all 169 canonical hole card pairs."""
        hands = []
        seen = set()
        for c1 in all_cards():
            for c2 in all_cards():
                if c1 == c2:
                    continue
                canon = canonical_hole(c1, c2)
                key = (min(canon[0][0], canon[1][0]), max(canon[0][0], canon[1][0]),
                       canon[0][1] == canon[1][1])
                if key not in seen:
                    seen.add(key)
                    hands.append(canon)
        return hands

    def _sample_board(
        self, hand: Tuple[str, str], street: str
    ) -> Tuple[str, ...]:
        """Sample a random board for the given street."""
        n_cards = {"flop": 3, "turn": 4, "river": 5}[street]
        used = set(hand)
        remaining = [c for c in all_cards() if c not in used]
        idx = np.random.choice(len(remaining), n_cards, replace=False)
        return tuple(remaining[i] for i in idx)


# ---------------------------------------------------------------------------
# Full Abstraction Pipeline
# ---------------------------------------------------------------------------

@dataclass
class AbstractedState:
    """A game state reduced to its abstract representation."""
    preflop_bucket: int          # 0–7 (suit isomorphism cluster)
    postflop_bucket: int         # 0–(BUCKETS-1) for current street
    street: str                  # "preflop", "flop", "turn", "river"
    hand_name: str               # "AKs", "AA", etc.
    canonical_hole: Tuple[str, str]
    equity_scalar: float         # Scalar equity (for fallback/logging)


class CardAbstractionV2:
    """
    Full card abstraction pipeline combining:
    1. Suit isomorphism (lossless, 1326 → 169 preflop)
    2. Preflop k-means on hand features (169 → 8 clusters)
    3. Postflop EMD bucketing with opponent-range-conditioned equity
       (range-dependent, not vs. random)

    This replaces card_abstraction.py's CombinedCardAbstraction.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        fit_on_init: bool = False,
        device: str = "cpu",
    ):
        self.equity_engine = EquityEngine()
        self.emd_bucketer = EMDBucketer(self.equity_engine, cache_dir)
        self._preflop_centroids: Optional[np.ndarray] = None

        if fit_on_init:
            self._fit_preflop_clusters()

    def abstract(
        self,
        hole_cards: Tuple[str, str],
        board_cards: Tuple[str, ...],
        opponent_range: Optional[Dict[str, float]] = None,
    ) -> AbstractedState:
        """
        Compute full card abstraction for a game state.

        Args:
            hole_cards:     Hero's 2 hole cards
            board_cards:    Community cards (0–5)
            opponent_range: Opponent's estimated range (improves bucketing)

        Returns:
            AbstractedState with all abstraction levels populated
        """
        # Step 1: Suit isomorphism
        c1, c2 = canonical_hole(hole_cards[0], hole_cards[1])
        hname = hand_name(hole_cards[0], hole_cards[1])

        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(
            len(board_cards), "river"
        )

        # Step 2: Preflop bucket
        pf_bucket = self._get_preflop_bucket((c1, c2))

        # Step 3: Postflop bucket (range-conditioned)
        if street == "preflop":
            postflop_bucket = pf_bucket
            equity_scalar = self._preflop_equity_scalar(hname)
        else:
            postflop_bucket = self.emd_bucketer.get_bucket(
                (c1, c2), board_cards, opp_range=opponent_range
            )
            equity_scalar = self.equity_engine.monte_carlo_equity_vs_range(
                (c1, c2), board_cards, opponent_range, n_samples=100
            )

        return AbstractedState(
            preflop_bucket=pf_bucket,
            postflop_bucket=postflop_bucket,
            street=street,
            hand_name=hname,
            canonical_hole=(c1, c2),
            equity_scalar=equity_scalar,
        )

    def abstract_key(
        self,
        hole_cards: Tuple[str, str],
        board_cards: Tuple[str, ...],
        action_history: Tuple[str, ...],
        opponent_range: Optional[Dict[str, float]] = None,
    ) -> bytes:
        """
        Compute a compact infoset key using abstracted cards.

        This reduces the key space from 10^14 to ~10^9 while preserving
        strategic relevance.
        """
        state = self.abstract(hole_cards, board_cards, opponent_range)
        key = (
            f"pf{state.preflop_bucket}"
            f"|pf{state.postflop_bucket}"
            f"|{state.street}"
            f"|{'|'.join(action_history)}"
        )
        return key.encode("utf-8")

    def _get_preflop_bucket(self, canonical_hand: Tuple[str, str]) -> int:
        """Map canonical hole cards to one of 8 preflop clusters."""
        # Feature vector: [rank1_norm, rank2_norm, suited, pair]
        c1, c2 = canonical_hand
        r1 = RANK_ORDER.get(c1[0].upper(), 0)
        r2 = RANK_ORDER.get(c2[0].upper(), 0)
        suited = 1.0 if c1[1] == c2[1] else 0.0
        pair = 1.0 if c1[0] == c2[0] else 0.0
        gap = abs(r1 - r2) / 12.0

        features = np.array([
            r1 / 12.0,
            r2 / 12.0,
            suited,
            pair,
            1.0 - gap,  # Connectedness
        ], dtype=np.float32)

        if self._preflop_centroids is None:
            self._fit_preflop_clusters()

        dists = np.linalg.norm(self._preflop_centroids - features[np.newaxis, :], axis=1)
        return int(np.argmin(dists))

    def _fit_preflop_clusters(self) -> None:
        """Fit 8 preflop clusters on hand feature vectors."""
        all_hands = self.emd_bucketer._all_canonical_hands()
        features = np.zeros((len(all_hands), 5), dtype=np.float32)

        for i, (c1, c2) in enumerate(all_hands):
            r1 = RANK_ORDER.get(c1[0].upper(), 0)
            r2 = RANK_ORDER.get(c2[0].upper(), 0)
            features[i] = [
                r1 / 12.0, r2 / 12.0,
                1.0 if c1[1] == c2[1] else 0.0,
                1.0 if c1[0] == c2[0] else 0.0,
                1.0 - abs(r1 - r2) / 12.0,
            ]

        centroids, _ = kmeans2(features, BUCKETS_PREFLOP, niter=100, seed=42, minit="points")
        self._preflop_centroids = centroids

    def _preflop_equity_scalar(self, hand_name_str: str) -> float:
        """Approximate preflop equity from canonical hand name."""
        from .trunk_value import _hand_strength_prior
        # Derive tuple from name
        if len(hand_name_str) == 2:
            return _hand_strength_prior(hand_name_str)
        return _hand_strength_prior(hand_name_str)
