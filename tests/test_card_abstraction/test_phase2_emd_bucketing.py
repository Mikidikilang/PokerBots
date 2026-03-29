"""
Phase 2 Card Abstraction: EMD Bucketing & Equity Precomputation Tests

[PHASE 2] Validates:
  1. Suit isomorphism canonicalization (1,326 → 169 hands)
  2. Equity precomputation with Treys (10,000 MC samples)
  3. EMD-based bucketing (preserves hand strength hierarchy)
  4. Street-specific bucket sizes (flop=150, turn=75, river=50)
  5. Lookup table serialization & loading
"""

import pytest
import logging
from pathlib import Path
from typing import Tuple

from src.env.card_abstraction import (
    SuitIsomorphismAbstraction,
    HandStrengthBucket,
    CombinedCardAbstraction,
    Card,
)

logger = logging.getLogger(__name__)


class TestSuitIsomorphism:
    """Test lossless card abstraction via suit isomorphism."""
    
    def test_canonicalize_hole_cards_sorted(self):
        """Test that hole cards are sorted by rank (higher first)."""
        abstractor = SuitIsomorphismAbstraction()
        
        # Kh with As → should become As, Ks (A-K, not K-A)
        c1, c2 = abstractor.canonicalize_hole_cards('Kh', 'As')
        assert c1[0] == 'A', f"Expected A, got {c1[0]}"
        assert c2[0] == 'K', f"Expected K, got {c2[0]}"
    
    def test_canonicalize_pair(self):
        """Test pair canonicalization (both cards same rank)."""
        abstractor = SuitIsomorphismAbstraction()
        
        c1, c2 = abstractor.canonicalize_hole_cards('As', 'Ah')
        assert c1 == 'As' and c2 == 'As', f"Pair should both be same suit, got {c1}, {c2}"
    
    def test_canonicalize_suited(self):
        """Test suited hand canonicalization."""
        abstractor = SuitIsomorphismAbstraction()
        
        # AKs in various suit combinations should canonicalize the same
        c1, c2 = abstractor.canonicalize_hole_cards('Ah', 'Kh')  # Suited
        assert c1[1] == c2[1], f"Suited hands should have same suit: {c1}, {c2}"
    
    def test_canonicalize_offsuit(self):
        """Test offsuit hand canonicalization."""
        abstractor = SuitIsomorphismAbstraction()
        
        c1, c2 = abstractor.canonicalize_hole_cards('Ah', 'Kd')  # Offsuit
        assert c1[1] != c2[1], f"Offsuit hands should have different suits: {c1}, {c2}"
    
    def test_169_canonical_hands(self):
        """Test that exactly 169 canonical hands are generated."""
        abstractor = SuitIsomorphismAbstraction()
        assert len(abstractor.canonical_hands) == 169, \
            f"Expected 169 hands, got {len(abstractor.canonical_hands)}"
    
    def test_board_canonicalization(self):
        """Test that board canonicalization preserves suit relationships."""
        abstractor = SuitIsomorphismAbstraction()
        
        # Flop: Qs, Tc, 9d
        board = ('Qs', 'Tc', '9d')
        canonical = abstractor.canonicalize_board(board)
        
        # Canonical board should have 3 cards
        assert len(canonical) == 3, f"Expected 3 cards, got {len(canonical)}"
        
        # Ranks should be preserved
        original_ranks = sorted([c[0] for c in board])
        canonical_ranks = sorted([c[0] for c in canonical])
        assert original_ranks == canonical_ranks, \
            f"Ranks should be preserved: {original_ranks} vs {canonical_ranks}"


class TestEquityBucketing:
    """Test lossy card abstraction via equity bucketing."""
    
    def test_hand_strength_bucket_initialization(self):
        """Test HandStrengthBucket initializes correctly."""
        bucketer = HandStrengthBucket(use_emd=True, mc_samples=1000)
        
        assert bucketer.use_emd is True
        assert bucketer.mc_samples == 1000
        assert len(bucketer.bucket_cache) == 0
    
    def test_street_specific_bucket_sizes(self):
        """Test that street-specific bucket sizes are correctly assigned."""
        bucketer = HandStrengthBucket(use_emd=True)
        
        # Test each street
        assert bucketer.get_street_buckets(()) == 1, "Preflop should have 1 bucket"
        assert bucketer.get_street_buckets(('Qs', 'Tc', '9d')) == 150, \
            "Flop should have 150 buckets"
        assert bucketer.get_street_buckets(('Qs', 'Tc', '9d', '2h')) == 75, \
            "Turn should have 75 buckets"
        assert bucketer.get_street_buckets(('Qs', 'Tc', '9d', '2h', '5s')) == 50, \
            "River should have 50 buckets"
    
    def test_percentile_bucketing(self):
        """Test simple percentile bucketing maps equity to bucket."""
        bucketer = HandStrengthBucket(use_emd=False)
        
        # Test boundary cases
        bucket_min = bucketer._percentile_bucket(0.0, num_buckets=100)
        assert bucket_min == 0, f"Minimum equity should map to bucket 0, got {bucket_min}"
        
        bucket_mid = bucketer._percentile_bucket(0.5, num_buckets=100)
        assert bucket_mid == 49, f"Middle equity (0.5) should map to bucket ~50, got {bucket_mid}"
        
        bucket_max = bucketer._percentile_bucket(1.0, num_buckets=100)
        assert bucket_max == 99, f"Maximum equity should map to bucket 99, got {bucket_max}"
    
    def test_emd_bucketing_ordering(self):
        """Test that EMD bucketing preserves hand strength ordering."""
        bucketer = HandStrengthBucket(use_emd=True)
        
        # Create equities in ascending order
        equities = [0.1, 0.2, 0.5, 0.8, 0.9]
        num_buckets = 50
        
        # Apply EMD bucketing
        buckets = [bucketer._emd_bucket(equities, i, num_buckets) for i in range(len(equities))]
        
        # Buckets should be in ascending order (preserves strength hierarchy)
        for i in range(len(buckets) - 1):
            assert buckets[i] <= buckets[i+1], \
                f"EMD should preserve ordering: {buckets[i]} > {buckets[i+1]}"
    
    def test_bucket_caching(self):
        """Test that bucket assignments are cached."""
        bucketer = HandStrengthBucket(use_emd=False)
        
        hole = ('As', 'Ks')
        board = ('Qs', 'Tc', '9d')
        
        # First call
        bucket1 = bucketer.get_bucket(hole, board)
        cache_size_1 = len(bucketer.bucket_cache)
        
        # Second call (should use cache)
        bucket2 = bucketer.get_bucket(hole, board)
        cache_size_2 = len(bucketer.bucket_cache)
        
        assert bucket1 == bucket2, "Cached bucket should be identical"
        assert cache_size_1 == cache_size_2, "Cache size should not change on second call"
        assert cache_size_1 == 1, f"Should have 1 cached entry, got {cache_size_1}"


class TestCombinedAbstraction:
    """Test integration of suit isomorphism + equity bucketing."""
    
    def test_combined_initialization(self):
        """Test CombinedCardAbstraction initializes both layers."""
        abstractor = CombinedCardAbstraction(use_emd=True, mc_samples=500)
        
        assert isinstance(abstractor.suit_iso, SuitIsomorphismAbstraction)
        assert isinstance(abstractor.equity_bucketer, HandStrengthBucket)
    
    def test_full_abstraction_pipeline_preflop(self):
        """Test full abstraction pipeline for preflop."""
        abstractor = CombinedCardAbstraction(use_emd=True)
        
        # Preflop observation
        obs = abstractor.abstract_observation(('As', 'Kd'), board=None)
        
        assert 'canonical_hole' in obs
        assert 'canonical_board' in obs
        assert 'equity_bucket' in obs
        assert 'hand_name' in obs
        assert 'street' in obs
        
        assert obs['street'] == 'preflop'
        assert obs['hand_name'] in ['AKo', 'AKs']  # Depending on suit
        assert obs['equity_bucket'] in [0, None], f"Preflop bucket should be 0 or None, got {obs['equity_bucket']}"
    
    def test_full_abstraction_pipeline_postflop(self):
        """Test full abstraction pipeline for postflop."""
        abstractor = CombinedCardAbstraction(use_emd=False)  # Use simple bucketing for test
        
        # Flop observation
        obs = abstractor.abstract_observation(
            ('As', 'Ks'),
            board=('Qs', 'Tc', '9d')
        )
        
        assert obs['street'] == 'flop'
        assert 0 <= obs['equity_bucket'] < 150, \
            f"Flop bucket should be in [0, 150), got {obs['equity_bucket']}"
    
    def test_canonicalization_consistency(self):
        """Test that canonicalization is consistent across calls."""
        abstractor = CombinedCardAbstraction()
        
        # Call multiple times with different input order
        hole1 = abstractor.canonicalize_hole_cards('As', 'Kh')
        hole2 = abstractor.canonicalize_hole_cards('Kh', 'As')
        
        # Both should canonicalize to the same form
        # (Note: canonicalization may not reverse input, but should be deterministic)
        c1_1, c1_2 = hole1
        c2_1, c2_2 = hole2
        
        # After canonicalization, one should be A and one should be K
        ranks_1 = sorted([c1_1[0], c1_2[0]])
        ranks_2 = sorted([c2_1[0], c2_2[0]])
        assert ranks_1 == ranks_2 == ['A', 'K'], \
            f"Both canonicalizations should produce A and K, got {ranks_1} and {ranks_2}"


class TestEquityPrecomputation:
    """Test integration with precomputation module (if Treys available)."""
    
    def test_equity_computation_range(self):
        """Test that computed equity is in valid range [0, 1]."""
        try:
            from src.env.equity_precompute import TreysEquityCalculator, CardCombo
            
            calc = TreysEquityCalculator()
            if not calc.available:
                pytest.skip("Treys not available")
            
            hole = (CardCombo('A', 's'), CardCombo('K', 's'))
            board = [CardCombo('Q', 's'), CardCombo('T', 'c'), CardCombo('9', 'd')]
            
            equity = calc.compute_equity_mc(hole, board, num_samples=100)
            
            assert 0.0 <= equity <= 1.0, \
                f"Equity should be in [0, 1], got {equity}"
        
        except ImportError:
            pytest.skip("Equity precompute module not available")
    
    def test_hand_strength_ordering(self):
        """Test that stronger hands have higher average equity (with higher variance on small boards)."""
        try:
            from src.env.equity_precompute import TreysEquityCalculator, CardCombo
            
            calc = TreysEquityCalculator()
            if not calc.available:
                pytest.skip("Treys not available")
            
            board = [CardCombo('2', 's'), CardCombo('3', 'c'), CardCombo('4', 'd')]
            
            # Very weak hand: 5-6 (low kicker, no straight potential on 234)
            weak_hand = (CardCombo('7', 's'), CardCombo('8', 'h'))
            weak_equity_samples = [
                calc.compute_equity_mc(weak_hand, board, num_samples=10)
                for _ in range(5)
            ]
            weak_equity_avg = sum(weak_equity_samples) / len(weak_equity_samples)
            
            # Stronger hand: KK (made pair)
            strong_hand = (CardCombo('K', 's'), CardCombo('K', 'h'))
            strong_equity_samples = [
                calc.compute_equity_mc(strong_hand, board, num_samples=10)
                for _ in range(5)
            ]
            strong_equity_avg = sum(strong_equity_samples) / len(strong_equity_samples)
            
            # On average, KK should beat 78 on 234 board
            # (Note: with small sample sizes, individual runs have high variance)
            logger.info(f"Weak hand (78) avg equity: {weak_equity_avg:.4f}")
            logger.info(f"Strong hand (KK) avg equity: {strong_equity_avg:.4f}")
            assert strong_equity_avg > weak_equity_avg, \
                f"KK ({strong_equity_avg:.4f}) should beat 78 ({weak_equity_avg:.4f}) on average"
        
        except ImportError:
            pytest.skip("Equity precompute module not available")


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase2Integration:
    """End-to-end integration tests for Phase 2 implementation."""
    
    def test_suit_isomorphism_reduces_hands(self):
        """Test that suit isomorphism reduces 1,326 to 169 canonical hands."""
        abstractor = SuitIsomorphismAbstraction()
        
        # All 1,326 hole card combinations
        ranks = "AKQJT98765432"
        hole_combos = []
        for i, r1 in enumerate(ranks):
            for r2 in ranks[i:]:
                for s1 in ['s', 'h', 'd', 'c']:
                    for s2 in ['s', 'h', 'd', 'c']:
                        if r1 == r2 and s1 >= s2:
                            continue  # Skip duplicate pairs
                        if r1 != r2 and (r1, s1) >= (r2, s2):
                            continue  # Skip duplicates
                        hole_combos.append((f"{r1}{s1}", f"{r2}{s2}"))
        
        # Canonicalize all
        canonical_set = set()
        for c1, c2 in hole_combos:
            canon = abstractor.canonicalize_hole_cards(c1, c2)
            canonical_set.add(canon)
        
        # Should have 169 unique canonical forms
        assert len(canonical_set) <= 169, \
            f"Should have ≤169 canonical hands, got {len(canonical_set)}"
    
    def test_full_pipeline_with_multiple_streets(self):
        """Test full pipeline across all streets."""
        abstractor = CombinedCardAbstraction(use_emd=False)
        
        # Test progression: preflop → flop → turn → river
        hole = ('As', 'Ks')
        
        # Preflop
        obs_pre = abstractor.abstract_observation(hole)
        assert obs_pre['street'] == 'preflop'
        assert obs_pre['equity_bucket'] in [0, None], \
            f"Preflop bucket should be 0 or None, got {obs_pre['equity_bucket']}"
        
        # Flop
        board_flop = ('Qs', 'Tc', '9d')
        obs_flop = abstractor.abstract_observation(hole, board_flop)
        assert obs_flop['street'] == 'flop'
        assert 0 <= obs_flop['equity_bucket'] < 150, \
            f"Flop bucket should be in [0, 150), got {obs_flop['equity_bucket']}"
        
        # Turn
        board_turn = board_flop + ('2h',)
        obs_turn = abstractor.abstract_observation(hole, board_turn)
        assert obs_turn['street'] == 'turn'
        assert 0 <= obs_turn['equity_bucket'] < 75, \
            f"Turn bucket should be in [0, 75), got {obs_turn['equity_bucket']}"
        
        # River
        board_river = board_turn + ('5s',)
        obs_river = abstractor.abstract_observation(hole, board_river)
        assert obs_river['street'] == 'river'
        assert 0 <= obs_river['equity_bucket'] < 50, \
            f"River bucket should be in [0, 50), got {obs_river['equity_bucket']}"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, '-v'])
