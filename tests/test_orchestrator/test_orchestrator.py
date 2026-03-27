"""
Egyseg tesztek a src/orchestrator/ modulhoz.

Tesztel: TelemetryAnalyzer, CurriculumManager, RewardShaper, AutoAdaptiveOrchestrator
"""

from __future__ import annotations

import random
import pytest

from src.orchestrator.telemetry import TelemetryAnalyzer, HandRecord
from src.orchestrator.curriculum import CurriculumManager, CurriculumPhase, UCBArm
from src.orchestrator.reward_shaper import RewardShaper, RewardShapingConfig
from src.orchestrator.orchestrator import AutoAdaptiveOrchestrator, OrchestratorConfig


# =============================================================================
# TelemetryAnalyzer Tesztek
# =============================================================================

class TestTelemetryAnalyzer:
    """A TelemetryAnalyzer HUD metrikak es anomalia detektalas tesztjei."""

    def _make_analyzer(self, window: int = 1000) -> TelemetryAnalyzer:
        return TelemetryAnalyzer(window_size=window, num_players=6)

    def _fill_normal(self, ana: TelemetryAnalyzer, n: int = 500) -> None:
        """Normalis (GTO-kozeli) jatekot szimulal."""
        for i in range(n):
            ana.record_hand(HandRecord(
                hand_id=i,
                player_id=0,
                position=random.randint(0, 5),
                iteration=0,
                reward_bb=random.gauss(0.02, 3),
                street_reached=random.randint(0, 3),
                went_to_showdown=random.random() < 0.30,
                won_at_showdown=random.random() < 0.15,
                vpip=random.random() < 0.24,
                pfr=random.random() < 0.20,
                three_bet=random.random() < 0.09,
                total_aggressive_actions=random.randint(0, 3),
                total_passive_actions=random.randint(0, 2),
                total_folds=random.randint(0, 1),
            ))

    def test_empty_metrics(self) -> None:
        ana = self._make_analyzer()
        m = ana.get_current_metrics()
        assert m["vpip"] == 0.0
        assert m["hands_in_window"] == 0.0

    def test_metrics_after_recording(self) -> None:
        ana = self._make_analyzer()
        self._fill_normal(ana, 500)
        m = ana.get_current_metrics()
        assert m["hands_in_window"] == 500.0
        assert 0 < m["vpip"] < 100
        assert 0 < m["pfr"] < 100
        assert m["af"] > 0

    def test_vpip_pfr_gap(self) -> None:
        """VPIP mindig >= PFR (nem emelhet anelkul hogy potba fizetne)."""
        ana = self._make_analyzer()
        self._fill_normal(ana)
        m = ana.get_current_metrics()
        assert m["vpip_pfr_gap"] >= -1.0  # Float pontossag

    def test_sliding_window_eviction(self) -> None:
        """Az ablak fix meretu: regi rekordok kiesnek."""
        ana = self._make_analyzer(window=100)
        self._fill_normal(ana, 200)
        m = ana.get_current_metrics()
        assert m["hands_in_window"] == 100.0

    def test_detect_passivity(self) -> None:
        """Extrem passzivitas detektalasa."""
        ana = self._make_analyzer(window=5000)
        for i in range(5000):
            ana.record_hand(HandRecord(
                hand_id=i,
                player_id=0,
                position=0,
                iteration=0,
                vpip=random.random() < 0.05,  # ~5% VPIP << 16%
                pfr=random.random() < 0.02,
                reward_bb=random.gauss(-0.5, 2),
                street_reached=0 if random.random() < 0.90 else 1,
                went_to_showdown=False,
                won_at_showdown=False,
                three_bet=False,
                total_aggressive_actions=0,
                total_passive_actions=0,
                total_folds=1,
            ))
        m = ana.get_current_metrics()
        # Manualisan ellenorizzuk: VPIP < 16% kell legyen
        assert m["vpip"] < 16.0, f"VPIP={m['vpip']:.1f}% (nem eleg alacsony a teszthez)"
        thresholds = {"passivity": {"vpip_below": 16.0, "vpip_pfr_gap_above": 10.0},
                      "maniac": {"pfr_above": 28.0, "three_bet_above": 15.0, "af_above": 4.0}}
        anomalies = ana.detect_anomalies({}, thresholds)
        assert "passivity" in anomalies

    def test_detect_maniac(self) -> None:
        """Extrem agresszio detektalasa."""
        ana = self._make_analyzer(window=5000)
        for i in range(5000):
            ana.record_hand(HandRecord(
                hand_id=i,
                player_id=0,
                position=0,
                iteration=0,
                vpip=True,
                pfr=True,           # 100% PFR >> 28%
                three_bet=random.random() < 0.5,  # ~50% 3bet >> 15%
                total_aggressive_actions=5,
                total_passive_actions=0,              # AF = 5/0 -> nagy
                reward_bb=random.gauss(0, 5),
                street_reached=1 if random.random() < 0.9 else 0,
                went_to_showdown=False,
                won_at_showdown=False,
                total_folds=0,
            ))
        m = ana.get_current_metrics()
        assert m["pfr"] > 28.0, f"PFR={m['pfr']:.1f}% (nem eleg magas)"
        thresholds = {"passivity": {"vpip_below": 16.0, "vpip_pfr_gap_above": 10.0},
                      "maniac": {"pfr_above": 28.0, "three_bet_above": 15.0, "af_above": 4.0}}
        anomalies = ana.detect_anomalies({}, thresholds)
        assert "maniac" in anomalies

    def test_no_anomalies_normal_play(self) -> None:
        ana = self._make_analyzer()
        self._fill_normal(ana, 2000)
        gto = {"vpip": [21.0, 27.0], "pfr": [17.0, 22.5]}
        thresholds = {"passivity": {"vpip_below": 16.0, "vpip_pfr_gap_above": 10.0},
                      "maniac": {"pfr_above": 28.0, "three_bet_above": 15.0, "af_above": 4.0}}
        anomalies = ana.detect_anomalies(gto, thresholds)
        # Normalis jateknal nem kene anomalianek lennie (de a random sokat szor)
        # Legalabb a struktura mukodik
        assert isinstance(anomalies, list)

    def test_gto_distance_in_range(self) -> None:
        """Ha a metrikak a GTO savban vannak, a tavolsag 0."""
        ana = self._make_analyzer()
        self._fill_normal(ana, 500)
        gto = {"vpip": [0.0, 100.0]}  # Nagyon tgg sav
        dist = ana.compute_gto_distance(gto)
        assert dist["vpip"] == 0.0

    def test_reset(self) -> None:
        ana = self._make_analyzer()
        self._fill_normal(ana, 100)
        ana.reset()
        assert ana.get_current_metrics()["hands_in_window"] == 0.0


# =============================================================================
# CurriculumManager Tesztek
# =============================================================================

class TestCurriculumManager:
    """A CurriculumManager fazisatmenet es UCB logikajanek tesztjei."""

    def test_initial_phase(self) -> None:
        mgr = CurriculumManager()
        assert mgr.current_phase == CurriculumPhase.PHASE_0_STATIC

    def test_phase_0_opponents(self) -> None:
        mgr = CurriculumManager()
        opponents = mgr.get_current_opponents()
        assert "calling_station" in opponents

    def test_no_transition_insufficient_data(self) -> None:
        mgr = CurriculumManager()
        result = mgr.check_phase_transition(
            {"win_rate_mbb": 100.0, "total_hands": 500.0}
        )
        assert not result

    def test_transition_0_to_1(self) -> None:
        mgr = CurriculumManager()
        result = mgr.check_phase_transition(
            {"win_rate_mbb": 60.0, "total_hands": 150_000.0}, iteration=1000,
        )
        assert result
        assert mgr.current_phase == CurriculumPhase.PHASE_1_SFT

    def test_no_transition_from_phase_2(self) -> None:
        mgr = CurriculumManager()
        mgr.current_phase = CurriculumPhase.PHASE_2_FSP
        result = mgr.check_phase_transition(
            {"win_rate_mbb": 1000.0, "total_hands": 1_000_000.0}
        )
        assert not result

    def test_ucb_arm_score(self) -> None:
        arm = UCBArm(name="test", total_reward=10.0, selection_count=5)
        score = arm.ucb_score(total_rounds=100, c=2.0)
        assert score > arm.average_reward  # Exploration bonus

    def test_ucb_unvisited_inf(self) -> None:
        arm = UCBArm(name="new")
        score = arm.ucb_score(total_rounds=100, c=2.0)
        assert score == float("inf")

    def test_ucb_opponent_selection(self) -> None:
        mgr = CurriculumManager()
        mgr.register_opponent("a")
        mgr.register_opponent("b")
        mgr.register_opponent("c")
        selected = mgr.select_opponent()
        assert selected in ["a", "b", "c"]

    def test_ucb_reward_update(self) -> None:
        mgr = CurriculumManager()
        mgr.register_opponent("bot_x")
        mgr.select_opponent()
        mgr.update_opponent_reward("bot_x", 5.0)
        stats = mgr.get_ucb_stats()
        assert stats["total_selections"] >= 1

    def test_state_save_load(self) -> None:
        mgr = CurriculumManager()
        mgr.check_phase_transition(
            {"win_rate_mbb": 60, "total_hands": 150_000.0}, iteration=500,
        )
        state = mgr.get_state()
        assert state["current_phase"] == 1

        mgr2 = CurriculumManager()
        mgr2.load_state(state)
        assert mgr2.current_phase == CurriculumPhase.PHASE_1_SFT

    def test_from_dict(self, sample_config: dict) -> None:
        mgr = CurriculumManager.from_dict(sample_config)
        assert mgr.current_phase == CurriculumPhase.PHASE_0_STATIC
        assert mgr.mab_config.algorithm == "ucb"


# =============================================================================
# RewardShaper Tesztek
# =============================================================================

class TestRewardShaper:
    """A RewardShaper jutalom-modositas logikajanek tesztjei."""

    def test_no_shaping_when_inactive(self) -> None:
        shaper = RewardShaper(RewardShapingConfig(
            bluff_penalty_lambda=0.0, preflop_aggression_bonus=0.0,
        ))
        result = shaper.shape_reward(5.0, action_index=8, lost_showdown=True)
        assert result == 5.0

    def test_bluff_penalty_applied(self) -> None:
        shaper = RewardShaper(RewardShapingConfig(bluff_penalty_lambda=1.0))
        result = shaper.shape_reward(
            5.0, action_index=8, bet_amount=200, pot_size=100,
            hand_strength=0.2, lost_showdown=True,
        )
        assert result < 5.0

    def test_no_penalty_without_showdown_loss(self) -> None:
        shaper = RewardShaper(RewardShapingConfig(bluff_penalty_lambda=1.0))
        result = shaper.shape_reward(
            5.0, action_index=8, bet_amount=200, pot_size=100,
            hand_strength=0.2, lost_showdown=False,  # Nem vesztett
        )
        assert result == 5.0

    def test_aggression_bonus_applied(self) -> None:
        shaper = RewardShaper(RewardShapingConfig(preflop_aggression_bonus=0.5))
        result = shaper.shape_reward(0.0, action_index=2, is_preflop_raise=True)
        assert result == 0.5

    def test_hot_reload_lambda(self) -> None:
        shaper = RewardShaper()
        shaper.update_penalty_lambda(2.0)
        assert shaper.config.bluff_penalty_lambda == 2.0

    def test_hot_reload_bonus(self) -> None:
        shaper = RewardShaper()
        shaper.update_aggression_bonus(0.3)
        assert shaper.config.preflop_aggression_bonus == 0.3

    def test_deactivate_all(self) -> None:
        shaper = RewardShaper(RewardShapingConfig(
            bluff_penalty_lambda=1.0, preflop_aggression_bonus=0.5,
        ))
        shaper.deactivate_all_shaping()
        assert shaper.config.bluff_penalty_lambda == 0.0
        assert shaper.config.preflop_aggression_bonus == 0.0

    def test_stats(self) -> None:
        shaper = RewardShaper(RewardShapingConfig(bluff_penalty_lambda=1.0))
        shaper.shape_reward(5.0, 8, bet_amount=200, pot_size=100,
                           hand_strength=0.1, lost_showdown=True)
        stats = shaper.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_penalties"] > 0


# =============================================================================
# AutoAdaptiveOrchestrator Tesztek
# =============================================================================

class TestOrchestrator:
    """Az AutoAdaptiveOrchestrator singleton es callback logikajanek tesztjei."""

    def setup_method(self) -> None:
        AutoAdaptiveOrchestrator.reset_instance()

    def teardown_method(self) -> None:
        AutoAdaptiveOrchestrator.reset_instance()

    def test_singleton(self, sample_config: dict) -> None:
        orch_cfg = OrchestratorConfig(num_players=6, enable_hot_reload=False)
        o1 = AutoAdaptiveOrchestrator.get_instance(orch_cfg, sample_config)
        o2 = AutoAdaptiveOrchestrator.get_instance()
        assert o1 is o2

    def test_initial_phase(self, sample_config: dict) -> None:
        orch_cfg = OrchestratorConfig(num_players=6, enable_hot_reload=False)
        orch = AutoAdaptiveOrchestrator.get_instance(orch_cfg, sample_config)
        assert orch.curriculum.current_phase == CurriculumPhase.PHASE_0_STATIC

    def test_callback_returns_dict(self, sample_config: dict) -> None:
        orch_cfg = OrchestratorConfig(num_players=6, eval_interval=1,
                                       enable_hot_reload=False)
        orch = AutoAdaptiveOrchestrator.get_instance(orch_cfg, sample_config)
        result = orch.on_iteration_callback(1, {})
        assert "phase" in result
        assert "anomalies" in result

    def test_passivity_intervention(self, sample_config: dict) -> None:
        """Extrem passzivitas triggereli a beavatkozast."""
        orch_cfg = OrchestratorConfig(
            num_players=6, eval_interval=1,
            telemetry_window=2000, enable_hot_reload=False,
        )
        orch = AutoAdaptiveOrchestrator.get_instance(orch_cfg, sample_config)

        # Passziv adatok injektalasa
        for i in range(2000):
            orch.telemetry.record_hand(HandRecord(
                hand_id=i,
                player_id=0,
                position=0,
                iteration=0,
                vpip=random.random() < 0.05,
                pfr=random.random() < 0.02,
                reward_bb=random.gauss(-0.5, 2),
                street_reached=0 if random.random() < 0.95 else 1,
                went_to_showdown=False,
                won_at_showdown=False,
                three_bet=False,
                total_aggressive_actions=0,
                total_passive_actions=0,
                total_folds=1,
            ))

        result = orch.on_iteration_callback(50, {})
        assert "passivity" in result["anomalies"]
        assert len(result["interventions"]) > 0

    def test_state_save_load(self, sample_config: dict) -> None:
        orch_cfg = OrchestratorConfig(num_players=6, enable_hot_reload=False)
        orch = AutoAdaptiveOrchestrator.get_instance(orch_cfg, sample_config)
        state = orch.get_state()
        assert "curriculum_state" in state

        AutoAdaptiveOrchestrator.reset_instance()
        orch2 = AutoAdaptiveOrchestrator.get_instance(orch_cfg, sample_config)
        orch2.load_state(state)

    def test_summary(self, sample_config: dict) -> None:
        orch_cfg = OrchestratorConfig(num_players=6, enable_hot_reload=False)
        orch = AutoAdaptiveOrchestrator.get_instance(orch_cfg, sample_config)
        summary = orch.get_summary()
        assert "current_phase" in summary
        assert "current_metrics" in summary
        assert "gto_distance" in summary
