"""
Egyseg tesztek a src/env/ modulhoz.

Tesztel: ObservationBuilder, ActionMapper, EquityCalculator
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env.features import ObservationBuilder, ObservationConfig, DECK_SIZE
from src.env.action_mapper import (
    ActionMapper, GameContext, PokerAction, ResolvedAction,
    NUM_ACTIONS, ILLEGAL_ACTION_LOGIT,
)
from src.env.equity import EquityCalculator


# =============================================================================
# ObservationConfig Tesztek
# =============================================================================

class TestObservationConfig:
    """Az ObservationConfig validacios logikajanek tesztjei."""

    def test_default_config(self) -> None:
        cfg = ObservationConfig()
        assert cfg.num_players == 6
        assert cfg.max_betting_actions == 18
        assert cfg.initial_stack_bb == 200.0

    def test_valid_player_counts(self) -> None:
        for n in [2, 3, 4, 6, 8, 9]:
            cfg = ObservationConfig(num_players=n)
            assert cfg.num_players == n

    def test_invalid_player_count_low(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            ObservationConfig(num_players=1)

    def test_invalid_player_count_high(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            ObservationConfig(num_players=10)


# =============================================================================
# ObservationBuilder Tesztek
# =============================================================================

class TestObservationBuilder:
    """Az ObservationBuilder fo muveleteinek tesztjei."""

    def test_build_output_keys(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder(ObservationConfig(num_players=6))
        obs = builder.build(sample_raw_state)
        expected_keys = {"hole_cards", "community_cards", "env_metrics",
                         "betting_history", "position", "action_mask"}
        assert set(obs.keys()) == expected_keys

    def test_hole_cards_shape(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder()
        obs = builder.build(sample_raw_state)
        assert obs["hole_cards"].shape == (52,)

    def test_hole_cards_multi_hot(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder()
        obs = builder.build(sample_raw_state)
        active = float(obs["hole_cards"].sum().item())
        assert active == 2.0  # Pontosan 2 lap

    def test_community_cards_flop(self, sample_raw_state: dict) -> None:
        """Flop: 3 kozos lap aktiv."""
        builder = ObservationBuilder()
        obs = builder.build(sample_raw_state)
        active = float(obs["community_cards"].sum().item())
        assert active == 3.0

    def test_community_cards_preflop(self, sample_preflop_state: dict) -> None:
        """Pre-flop: 0 kozos lap."""
        builder = ObservationBuilder()
        obs = builder.build(sample_preflop_state)
        active = float(obs["community_cards"].sum().item())
        assert active == 0.0

    def test_env_metrics_shape(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder(ObservationConfig(num_players=6))
        obs = builder.build(sample_raw_state)
        expected_dim = 4 + 5  # pot, stack, call, min_raise + 5 opponents
        assert obs["env_metrics"].shape == (expected_dim,)

    def test_env_metrics_normalized(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder()
        obs = builder.build(sample_raw_state)
        metrics = obs["env_metrics"]
        assert float(metrics.min().item()) >= 0.0
        assert float(metrics.max().item()) <= 1.0

    def test_betting_history_shape(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder()
        obs = builder.build(sample_raw_state)
        assert obs["betting_history"].shape == (18, 9)

    def test_position_one_hot(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder(ObservationConfig(num_players=6))
        obs = builder.build(sample_raw_state)
        pos = obs["position"]
        assert pos.shape == (6,)
        assert float(pos.sum().item()) == 1.0  # Pontosan egy 1-es

    def test_action_mask_shape(self, sample_raw_state: dict) -> None:
        builder = ObservationBuilder()
        obs = builder.build(sample_raw_state)
        assert obs["action_mask"].shape == (10,)

    def test_flatten_dimension(self) -> None:
        builder = ObservationBuilder(ObservationConfig(num_players=6))
        expected = builder.get_observation_dim()
        raw = {
            "hand": ["SA", "HK"], "public_cards": [], "pot": 100.0,
            "my_chips": 2000.0, "opponent_chips": [2000]*5, "big_blind": 10.0,
            "amount_to_call": 0.0, "min_raise": 20.0, "position": 3,
            "betting_history": [], "legal_actions": [0, 1],
        }
        obs = builder.build(raw)
        flat = builder.flatten(obs)
        assert flat.shape[0] == expected

    def test_card_roundtrip(self) -> None:
        for card in ["SA", "C2", "HT", "DK", "S7"]:
            idx = ObservationBuilder.card_str_to_index(card)
            back = ObservationBuilder.index_to_card_str(idx)
            assert back == card, f"Roundtrip failed: {card} -> {idx} -> {back}"

    def test_invalid_card_format(self) -> None:
        builder = ObservationBuilder()
        with pytest.raises(ValueError):
            builder.build({"hand": ["XY"], "public_cards": [], "pot": 0,
                          "my_chips": 0, "big_blind": 10, "amount_to_call": 0,
                          "position": 0, "legal_actions": [0]})

    def test_missing_key_raises(self) -> None:
        builder = ObservationBuilder()
        with pytest.raises(KeyError):
            builder.build({"pot": 100})  # Hianyzo "hand" kulcs


# =============================================================================
# ActionMapper Tesztek
# =============================================================================

class TestActionMapper:
    """Az ActionMapper akciofoldas es maszkolasi logikajanek tesztjei."""

    def test_fold_resolution(self) -> None:
        mapper = ActionMapper()
        ctx = GameContext(pot_size=100, my_stack=500, amount_to_call=50,
                         min_raise_amount=100, big_blind=10)
        result = mapper.resolve_action(PokerAction.FOLD, ctx)
        assert result.action == PokerAction.FOLD
        assert result.amount == 0.0

    def test_check_call_no_bet(self) -> None:
        mapper = ActionMapper()
        ctx = GameContext(pot_size=100, my_stack=500, amount_to_call=0,
                         min_raise_amount=20, big_blind=10)
        result = mapper.resolve_action(PokerAction.CHECK_CALL, ctx)
        assert result.amount == 0.0
        assert "Check" in result.description

    def test_check_call_with_bet(self) -> None:
        mapper = ActionMapper()
        ctx = GameContext(pot_size=100, my_stack=500, amount_to_call=50,
                         min_raise_amount=100, big_blind=10)
        result = mapper.resolve_action(PokerAction.CHECK_CALL, ctx)
        assert result.amount == 50.0
        assert "Call" in result.description

    def test_raise_full_pot(self) -> None:
        mapper = ActionMapper()
        ctx = GameContext(pot_size=200, my_stack=1000, amount_to_call=0,
                         min_raise_amount=20, big_blind=10)
        result = mapper.resolve_action(PokerAction.RAISE_FULL_POT, ctx)
        assert result.amount == 200.0  # 1.0x pot

    def test_stack_capping(self) -> None:
        """Ha az emeles meghaladja a stack-et, All-in-ne konvertalodik."""
        mapper = ActionMapper()
        ctx = GameContext(pot_size=800, my_stack=200, amount_to_call=50,
                         min_raise_amount=100, big_blind=10)
        result = mapper.resolve_action(PokerAction.RAISE_FULL_POT, ctx)
        assert result.action == PokerAction.ALL_IN
        assert result.amount == 200.0

    def test_all_in_resolution(self) -> None:
        mapper = ActionMapper()
        ctx = GameContext(pot_size=100, my_stack=500, amount_to_call=50,
                         min_raise_amount=100, big_blind=10)
        result = mapper.resolve_action(PokerAction.ALL_IN, ctx)
        assert result.action == PokerAction.ALL_IN
        assert result.amount == 500.0

    def test_legal_actions_always_include_fold_call(self) -> None:
        mapper = ActionMapper()
        ctx = GameContext(pot_size=100, my_stack=500, amount_to_call=50,
                         min_raise_amount=100, big_blind=10)
        legal = mapper.get_legal_actions(ctx)
        assert PokerAction.FOLD in legal
        assert PokerAction.CHECK_CALL in legal

    def test_action_mask_shape(self) -> None:
        mapper = ActionMapper()
        ctx = GameContext(pot_size=100, my_stack=500, amount_to_call=50,
                         min_raise_amount=100, big_blind=10)
        mask = mapper.get_action_mask_tensor(ctx)
        assert mask.shape == (10,)

    def test_action_mask_logit_masking(self) -> None:
        """Az illegalis akciok valoszinusege ~0 a Softmax utan."""
        import torch
        mapper = ActionMapper()
        # Legal actions: FOLD(0), CHECK_CALL(1), RAISE_3QUARTER_POT(5), ALL_IN(9)
        mask_data = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        mask = torch.tensor(mask_data)
        logits = torch.tensor(np.random.randn(10).astype(np.float32))
        masked = ActionMapper.apply_action_mask(logits, mask)

        # Softmax
        probs = torch.softmax(masked, dim=-1)
        # Illegalis indexek: 2, 3, 4, 6, 7, 8
        for idx in [2, 3, 4, 6, 7, 8]:
            assert float(probs[idx].item()) < 1e-20

    def test_action_mask_shape_mismatch_raises(self) -> None:
        import torch
        logits = torch.tensor(np.zeros(10, dtype=np.float32))
        mask = torch.tensor(np.zeros(5, dtype=np.float32))
        with pytest.raises(ValueError, match="alakja nem egyezik"):
            ActionMapper.apply_action_mask(logits, mask)

    def test_sample_action_deterministic(self) -> None:
        """A determinisztikus mod a legmagasabb valoszinusegu akciot valasztja."""
        import torch
        # Egyetlen domináns logit -> a determinisztikus valasztas egyertelmu
        data = np.array([-100]*4 + [10.0] + [-100]*4, dtype=np.float32)
        masked = torch.tensor(data)
        probs = torch.softmax(masked, dim=-1)
        # A 4-es index valoszinusege ~1.0
        assert float(probs[4].item()) > 0.99

    def test_all_action_names(self) -> None:
        mapper = ActionMapper()
        for i in range(9):
            name = mapper.action_index_to_name(i)
            assert name != f"Invalid-{i}"


# =============================================================================
# EquityCalculator Tesztek
# =============================================================================

class TestEquityCalculator:
    """Az EquityCalculator kez-ero szamitasi logikajanek tesztjei."""

    def test_preflop_hand_categories(self) -> None:
        calc = EquityCalculator()
        assert calc.get_preflop_hand_category(["SA", "HA"]) == "premium"  # AA
        assert calc.get_preflop_hand_category(["SA", "SK"]) == "premium"  # AKs
        assert calc.get_preflop_hand_category(["HJ", "DJ"]) == "strong"   # JJ
        assert calc.get_preflop_hand_category(["S5", "H5"]) == "playable" # 55
        assert calc.get_preflop_hand_category(["D7", "C2"]) == "trash"    # 72o

    def test_equity_range(self) -> None:
        calc = EquityCalculator()
        eq = calc.calculate_equity(["SA", "HA"], [], num_opponents=1, iterations=300)
        assert 0.0 <= eq <= 1.0

    def test_aa_beats_random(self) -> None:
        """AA-nak magasabb equity-je kell legyen mint 50% HU."""
        calc = EquityCalculator()
        eq = calc.calculate_equity(["SA", "HA"], [], num_opponents=1, iterations=500)
        assert eq > 0.7  # AA ~85% HU

    def test_made_hand_high_equity(self) -> None:
        """Royal flush board-dal kozel 100% equity."""
        calc = EquityCalculator()
        eq = calc.calculate_equity(
            ["SA", "SK"], ["ST", "SJ", "SQ"],
            num_opponents=1, iterations=300,
        )
        assert eq > 0.9

    def test_invalid_hole_cards_count(self) -> None:
        calc = EquityCalculator()
        with pytest.raises(ValueError, match="Pontosan 2"):
            calc.calculate_equity(["SA"], [])

    def test_too_many_community_cards(self) -> None:
        calc = EquityCalculator()
        with pytest.raises(ValueError, match="Legfeljebb 5"):
            calc.calculate_equity(["SA", "HK"], ["C2"]*6)

    def test_hand_strength_requires_board(self) -> None:
        calc = EquityCalculator()
        with pytest.raises(ValueError, match="egal"):
            calc.evaluate_hand_strength(["SA", "HK"], ["C2", "D3"])

    def test_hand_strength_ordering(self) -> None:
        """Az erosebb keznek alacsonyabb (jobb) score-ja van."""
        calc = EquityCalculator()
        pair_score = calc.evaluate_hand_strength(["SA", "HA"], ["C2", "D7", "S9"])
        high_score = calc.evaluate_hand_strength(["HK", "DQ"], ["C2", "D7", "S9"])
        assert pair_score < high_score  # Par erosebb mint high card
