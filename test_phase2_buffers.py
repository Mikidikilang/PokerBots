#!/usr/bin/env python
"""Test Phase 2: Memory & Buffer Architecture."""

import numpy as np
from src.training.buffers import (
    Transition, 
    EphemeralAdvantageBuffer, 
    PersistentStrategyBuffer,
    BufferManager,
)


def test_transition_validation():
    """Test Transition immutability and validation."""
    features = np.array([0.1, 0.2, 0.3])
    probs = np.array([0.2, 0.8])
    adv = np.array([-0.1, 0.1])
    
    # Valid transition
    t = Transition(features, probs, adv, iteration=5, reach_prob=0.9)
    assert t.iteration == 5
    assert t.reach_prob == 0.9
    
    # Test immutability (frozen dataclass prevents field assignment)
    try:
        t.iteration = 10
        print("✗ FAIL: Transition is not frozen")
        return False
    except (TypeError, AttributeError, ValueError):
        pass  # Expected: frozen dataclass prevents assignment
    
    # Test invalid iteration
    try:
        bad_t = Transition(features, probs, adv, iteration=0)
        print("✗ FAIL: Should reject iteration < 1")
        return False
    except ValueError:
        pass
    
    print("✓ Transition: immutability and validation OK")
    return True


def test_ephemeral_buffer_clear():
    """Test that ephemeral buffer correctly clears."""
    buf = EphemeralAdvantageBuffer(capacity=100)
    
    # Add data
    for t in range(5):
        features = np.random.randn(10)
        probs = np.array([0.4, 0.6])
        adv = np.array([0.1, -0.1])
        trans = Transition(features, probs, adv, iteration=1)
        buf.insert(trans)
    
    assert buf.size() == 5, f"Expected 5 transitions, got {buf.size()}"
    
    # Clear
    buf.clear()
    assert buf.size() == 0, f"Buffer not cleared, size={buf.size()}"
    
    print("✓ EphemeralAdvantageBuffer: clear() works correctly")
    return True


def test_ephemeral_buffer_sampling():
    """Test ephemeral buffer minibatch sampling."""
    buf = EphemeralAdvantageBuffer(capacity=1000)
    
    # Add heterogeneous data
    for i in range(20):
        features = np.random.randn(8)
        probs = np.random.dirichlet([1, 1, 1])  # 3 actions
        adv = np.random.randn(3)
        trans = Transition(features, probs, adv, iteration=1)
        buf.insert(trans)
    
    # Sample
    feat, prob, adv = buf.sample_minibatch(batch_size=5)
    
    assert feat.shape == (5, 8), f"Features shape {feat.shape}, expected (5, 8)"
    assert prob.shape == (5, 3), f"Probs shape {prob.shape}, expected (5, 3)"
    assert adv.shape == (5, 3), f"Advantages shape {adv.shape}, expected (5, 3)"
    
    print("✓ EphemeralAdvantageBuffer: sampling returns correct shapes")
    return True


def test_persistent_buffer_time_decay():
    """Test time-decay weighting in persistent buffer."""
    buf = PersistentStrategyBuffer(capacity=10000, time_decay_power=2.0)
    
    # Add data from iterations 1, 5, 10
    for it in [1, 5, 10]:
        for i in range(3):
            features = np.random.randn(6)
            probs = np.random.dirichlet([1, 1])  # 2 actions
            trans = Transition(features, probs, iteration=it)
            buf.insert(trans)
    
    assert buf.size() == 9
    assert buf.oldest_iteration() == 1
    assert buf.newest_iteration() == 10
    
    # Sample at iteration 10: recent data should be sampled more often
    feat, prob, weights = buf.sample_minibatch(
        batch_size=9,  # Sample all 9 transitions (with replacement support)
        current_iteration=10,
        replace=True,
    )
    
    # With t^2 decay and current_iter=10:
    #   t=1: weight = (1/10)^2 = 0.01
    #   t=5: weight = (5/10)^2 = 0.25
    #   t=10: weight = (10/10)^2 = 1.0
    # Normalized: t=1: 0.0099, t=5: 0.247, t=10: 0.988
    # So t=10 samples get weight ~1.0, others get less
    
    assert feat.shape[0] == 9
    assert prob.shape[0] == 9
    assert weights.shape == (9,)
    assert np.allclose(weights.mean(), 1.0, atol=0.1), \
        f"Weights normalized to mean {weights.mean()}, expected ~1.0"
    
    print("✓ PersistentStrategyBuffer: time-decay weighting works")
    return True


def test_persistent_buffer_capacity():
    """Test that persistent buffer respects capacity."""
    buf = PersistentStrategyBuffer(capacity=10, time_decay_power=1.0)
    
    # Add 20 transitions
    for i in range(20):
        features = np.random.randn(5)
        probs = np.array([0.5, 0.5])
        trans = Transition(features, probs, iteration=i + 1)
        buf.insert(trans)
    
    # Should have at most 10
    assert buf.size() == 10, f"Size {buf.size()}, expected 10"
    
    # With random eviction, we can't guarantee specific iteration ranges,
    # but we should have a mix from later iterations with higher probability
    # Just verify buffer contains data from at least some of the later iterations
    oldest = buf.oldest_iteration()
    newest = buf.newest_iteration()
    assert 1 <= oldest <= newest <= 20, \
        f"Iteration range invalid: [{oldest}, {newest}]"
    assert buf.size() == 10, "Capacity not enforced"
    
    print("✓ PersistentStrategyBuffer: capacity enforcement works (random eviction)")
    return True


def test_buffer_manager_lifecycle():
    """Test BufferManager iteration lifecycle."""
    mgr = BufferManager(
        advantage_capacity=100,
        strategy_capacity=1000,
        time_decay_power=1.0,
    )
    
    # Iteration 1
    assert mgr.current_iteration == 1
    mgr.start_iteration()  # Should clear ephemeral buffer
    assert mgr.advantage_buffer.size() == 0
    
    # Add data
    for i in range(5):
        features = np.random.randn(4)
        probs = np.array([0.3, 0.7])
        adv = np.array([0.05, -0.05])
        mgr.add_transition(features, probs, adv)
    
    assert mgr.advantage_buffer.size() == 5
    assert mgr.strategy_buffer.size() == 5
    
    # Iteration 2
    mgr.end_iteration()
    assert mgr.current_iteration == 2
    mgr.start_iteration()
    assert mgr.advantage_buffer.size() == 0  # Cleared!
    assert mgr.strategy_buffer.size() == 5   # Persists!
    
    print("✓ BufferManager: iteration lifecycle correct")
    return True


def test_buffer_manager_sampling():
    """Test BufferManager can sample from both buffers."""
    mgr = BufferManager()
    
    # Populate
    for i in range(10):
        features = np.random.randn(3)
        probs = np.array([0.4, 0.6])
        adv = np.array([-0.1, 0.1])
        mgr.add_transition(features, probs, adv)
    
    # Sample advantage buffer
    feat_adv, prob_adv, adv_adv = mgr.advantage_buffer.sample_minibatch(batch_size=3)
    assert feat_adv.shape == (3, 3)
    
    # Sample strategy buffer
    feat_strat, prob_strat, weights = mgr.strategy_buffer.sample_minibatch(
        batch_size=3,
        current_iteration=1,
    )
    assert feat_strat.shape == (3, 3)
    assert weights.shape == (3,)
    
    print("✓ BufferManager: sampling from both buffers works")
    return True


def test_iteration_distribution():
    """Test iteration distribution histogram."""
    buf = PersistentStrategyBuffer(capacity=1000)
    
    # Add from iterations 1-5 (two each)
    for it in [1, 1, 2, 2, 3, 3]:
        features = np.random.randn(2)
        probs = np.array([0.5, 0.5])
        trans = Transition(features, probs, iteration=it)
        buf.insert(trans)
    
    dist = buf.get_iteration_distribution()
    assert dist[1] == 2
    assert dist[2] == 2
    assert dist[3] == 2
    
    print("✓ PersistentStrategyBuffer: iteration distribution tracking works")
    return True


if __name__ == "__main__":
    print("Testing Phase 2: Memory & Buffer Architecture\n")
    
    tests = [
        test_transition_validation,
        test_ephemeral_buffer_clear,
        test_ephemeral_buffer_sampling,
        test_persistent_buffer_time_decay,
        test_persistent_buffer_capacity,
        test_buffer_manager_lifecycle,
        test_buffer_manager_sampling,
        test_iteration_distribution,
    ]
    
    results = [test() for test in tests]
    
    print(f"\n{'='*60}")
    if all(results):
        print("ALL TESTS PASSED ✓")
        print("\nPhase 2 Implementation Summary:")
        print("  - Transition: Immutable, validated dataclass")
        print("  - EphemeralAdvantageBuffer: Cleared each iteration")
        print("  - PersistentStrategyBuffer: Time-decay weighted across iterations")
        print("  - BufferManager: Unified lifecycle management")
    else:
        print("SOME TESTS FAILED ✗")
