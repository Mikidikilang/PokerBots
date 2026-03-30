"""
Phase 2 LSTM History Encoder Wiring Test

Tests that the newly wired LSTMHistoryEncoder (replacing HistoryEmbedding)
correctly processes (batch, 18, 13) betting history tensors and integrates
cleanly with PokerActorCritic.

Key assumption: betting_history has shape (batch, 18, 13), NOT (batch, 234).
"""

from __future__ import annotations

import logging
import sys
import torch

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from src.model.networks import PokerActorCritic, NetworkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_lstm_history_encoder_integration():
    """Test 1: PokerActorCritic with LSTMHistoryEncoder accepts (batch, 18, 13) history."""
    print("\n" + "=" * 70)
    print("TEST 1: PokerActorCritic Forward Pass with LSTM History Encoder")
    print("=" * 70)
    
    config = NetworkConfig()
    actor_critic = PokerActorCritic(config=config)
    
    batch_size = 4
    
    # Create observation dict with (batch, 18, 13) betting history
    observation = {
        "hole_cards": torch.randn(batch_size, 52),
        "community_cards": torch.randn(batch_size, 52),
        "env_metrics": torch.randn(batch_size, 9),
        "position": torch.randn(batch_size, 6),
        "betting_history": torch.randn(batch_size, 18, 13),  # [PHASE 2] LSTM input shape
        "action_mask": torch.ones(batch_size, 10),  # All actions valid (10 actions in config)
    }
    
    # Forward pass
    action_dist, value = actor_critic.forward(observation)
    
    # Assert output shapes
    assert action_dist.batch_shape == (batch_size,), \
        f"Action dist batch shape {action_dist.batch_shape} != {(batch_size,)}"
    assert value.shape == (batch_size, 1), \
        f"Value shape {value.shape} != {(batch_size, 1)}"
    
    # Assert numeric validity
    assert not torch.isnan(value).any(), "Value contains NaN"
    assert not torch.isinf(value).any(), "Value contains inf"
    
    logger.info(f"✓ Forward pass successful: batch={batch_size}, "
                f"action_dist.shape={action_dist.batch_shape}, "
                f"value.shape={value.shape}")
    print(f"PASS: Action distribution shape {action_dist.batch_shape}, "
          f"value shape {value.shape}\n")


def test_lstm_with_action_sampling():
    """Test 2: PokerActorCritic.get_action_and_value() works with LSTM encoder."""
    print("\n" + "=" * 70)
    print("TEST 2: get_action_and_value() with LSTM History Encoder")
    print("=" * 70)
    
    config = NetworkConfig()
    actor_critic = PokerActorCritic(config=config)
    
    batch_size = 2
    
    observation = {
        "hole_cards": torch.randn(batch_size, 52),
        "community_cards": torch.randn(batch_size, 52),
        "env_metrics": torch.randn(batch_size, 9),
        "position": torch.randn(batch_size, 6),
        "betting_history": torch.randn(batch_size, 18, 13),
        "action_mask": torch.ones(batch_size, 10),  # 10 actions in config
    }
    
    action, log_prob, entropy, value = actor_critic.get_action_and_value(observation)
    
    # Assert shapes
    assert action.shape == (batch_size,), \
        f"Action shape {action.shape} != {(batch_size,)}"
    assert log_prob.shape == (batch_size,), \
        f"Log prob shape {log_prob.shape} != {(batch_size,)}"
    assert entropy.shape == (batch_size,), \
        f"Entropy shape {entropy.shape} != {(batch_size,)}"
    assert value.shape == (batch_size, 1), \
        f"Value shape {value.shape} != {(batch_size, 1)}"
    
    # Assert numeric validity
    assert not torch.isnan(log_prob).any(), "Log prob contains NaN"
    assert not torch.isnan(entropy).any(), "Entropy contains NaN"
    assert not torch.isnan(value).any(), "Value contains NaN"
    assert (entropy >= 0).all(), "Entropy should be non-negative"
    
    logger.info(f"✓ get_action_and_value() successful: "
                f"action={action.shape}, lp={log_prob.shape}, "
                f"ent={entropy.shape}, val={value.shape}")
    print(f"PASS: Action sample {action.shape}, log_prob {log_prob.shape}, "
          f"entropy {entropy.shape}, value {value.shape}\n")


def test_lstm_history_dimension_check():
    """Test 3: LSTM encoder output dimension is 256 (hidden_dim)."""
    print("\n" + "=" * 70)
    print("TEST 3: LSTM Encoder Output Dimension Check")
    print("=" * 70)
    
    config = NetworkConfig()
    actor_critic = PokerActorCritic(config=config)
    
    # Check that history encoder output is 256 (configured hidden_dim)
    expected_lstm_output = 256  # hidden_dim=256, num_layers=2
    actual_lstm_output = actor_critic.history_encoder.output_dim
    
    assert actual_lstm_output == expected_lstm_output, \
        f"LSTM output dim {actual_lstm_output} != {expected_lstm_output}"
    
    # Check that fusion dim accounts for LSTM output (256) instead of old 64
    # fusion = cards(64*2=128) + ctx(32) + lstm(256) = 416
    expected_fusion = 128 + 32 + 256
    actual_fusion = actor_critic._fusion_dim
    
    assert actual_fusion == expected_fusion, \
        f"Fusion dim {actual_fusion} != {expected_fusion}"
    
    logger.info(f"✓ LSTM encoder output dim: {actual_lstm_output}")
    logger.info(f"✓ Fusion dim: cards(128) + ctx(32) + lstm(256) = {actual_fusion}")
    print(f"PASS: LSTM output dimension {actual_lstm_output}, "
          f"fusion dimension {actual_fusion}\n")


def test_lstm_batch_dimensions():
    """Test 4: LSTM encoder handles variable batch sizes correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: LSTM Encoder Batch Dimension Handling")
    print("=" * 70)
    
    config = NetworkConfig()
    actor_critic = PokerActorCritic(config=config)
    
    # Test with different batch sizes
    for batch_size in [1, 2, 8, 16]:
        observation = {
            "hole_cards": torch.randn(batch_size, 52),
            "community_cards": torch.randn(batch_size, 52),
            "env_metrics": torch.randn(batch_size, 9),
            "position": torch.randn(batch_size, 6),
            "betting_history": torch.randn(batch_size, 18, 13),
            "action_mask": torch.ones(batch_size, 10),  # 10 actions
        }
        
        action_dist, value = actor_critic.forward(observation)
        
        assert value.shape == (batch_size, 1), \
            f"Batch {batch_size}: value shape {value.shape} != {(batch_size, 1)}"
    
    logger.info(f"✓ All batch sizes processed correctly: 1, 2, 8, 16")
    print(f"PASS: Batch sizes 1, 2, 8, 16 all successful\n")


def test_lstm_no_grad_inference():
    """Test 5: LSTM encoder works in no_grad() inference mode."""
    print("\n" + "=" * 70)
    print("TEST 5: LSTM Encoder in no_grad() Inference Mode")
    print("=" * 70)
    
    config = NetworkConfig()
    actor_critic = PokerActorCritic(config=config)
    
    batch_size = 4
    observation = {
        "hole_cards": torch.randn(batch_size, 52),
        "community_cards": torch.randn(batch_size, 52),
        "env_metrics": torch.randn(batch_size, 9),
        "position": torch.randn(batch_size, 6),
        "betting_history": torch.randn(batch_size, 18, 13),
        "action_mask": torch.ones(batch_size, 10),  # 10 actions
    }
    
    with torch.no_grad():
        action, log_prob, entropy, value = actor_critic.get_action_and_value(observation)
    
    # Verify no gradients were computed
    assert not value.requires_grad, "Value should not require grad in no_grad() mode"
    
    # Verify numeric validity
    assert not torch.isnan(value).any(), "Value contains NaN"
    assert value.shape == (batch_size, 1), f"Value shape {value.shape} != {(batch_size, 1)}"
    
    logger.info(f"✓ no_grad() inference successful: value shape {value.shape}")
    print(f"PASS: Inference mode works, value shape {value.shape}\n")


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all Phase 2 LSTM wiring tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PHASE 2: LSTM HISTORY ENCODER WIRING" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    
    tests = [
        ("PokerActorCritic Forward with LSTM", test_lstm_history_encoder_integration),
        ("get_action_and_value() with LSTM", test_lstm_with_action_sampling),
        ("LSTM Output Dimension", test_lstm_history_dimension_check),
        ("LSTM Batch Dimension Handling", test_lstm_batch_dimensions),
        ("LSTM in no_grad() Mode", test_lstm_no_grad_inference),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True))
        except AssertionError as e:
            logger.error(f"Assertion failed: {e}")
            results.append((test_name, False))
            print(f"FAIL: {e}\n")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            results.append((test_name, False))
            print(f"FAIL: Unexpected exception: {e}\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED: LSTM History Encoder is properly wired!")
        print("  - Accepts (batch, 18, 13) betting history tensors")
        print("  - Outputs (batch, 256) LSTM hidden states")
        print("  - Integrates cleanly with PokerActorCritic")
        print("  - Handles batches, inference, and gradient modes\n")
        return 0
    else:
        print(f"\n✗ {total - passed} tests failed\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
