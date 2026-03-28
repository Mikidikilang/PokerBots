"""
HUD Telemetria Analizator (telemetry.py).

[FIX C3 - 2025-03-28] O(N) Stagnacio Detektalo Lecserelve O(1) Megoldasra:
    A korabbi check_stagnation() implementacio itertools.islice()-t hasznalt
    egy deque-n, ami O(N) muvelet (N = window_size). window_size=100_000-nel
    es eval_interval=50-nel ez 200x hivodik egy 12 oras Kaggle session alatt,
    minden alkalommal akár 100,000 elemet atiterálva — kb. 100ms CPU spike
    alkalmanként, ami GPU-ejhezel (GPU starvation) okoz.

    A javitas: egy _recent_deque: deque[float] nevű, maxlen=stagnation_half
    meretu csuszoablakos deque-t tartunk fenn, ami O(1) frissitessel
    kezelheto a record_hand() hivaskor. A check_stagnation() ezutan
    az O(1) osszeget hasonlitja a teljes ablak O(1) _reward_sum-jahoz.
    Igy a check_stagnation() teljes koltseige O(1) lesz O(N) helyett.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class ActionCategory(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    ALL_IN = 3


class StreetPhase(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


@dataclass
class HandRecord:
    """Per-hand HUD record submitted to TelemetryAnalyzer.record_hand()."""
    hand_id: int
    player_id: int
    position: int
    iteration: int
    reward_bb: float
    street_reached: int
    went_to_showdown: bool
    won_at_showdown: bool
    vpip: bool
    pfr: bool
    three_bet: bool
    total_aggressive_actions: int
    total_passive_actions: int
    total_folds: int
    actions: list[int] = field(default_factory=list)
    action_streets: list[int] = field(default_factory=list)


class TelemetryAnalyzer:
    """Mozgoablakos HUD metrikak szamitasa es anomalia detektalas.

    [FIX C3] Az O(N) check_stagnation() helyett O(1) implementacio:
        - _recent_deque: deque[float] (maxlen=stagnation_half) tartja
          az utolso N jutalom erteket inkrementalisan.
        - record_hand() O(1) kovetkessel frissiti ezt a deque-t.
        - check_stagnation() az O(1) _reward_sum / len(window) atlagot
          hasonlitja az O(1) recent_mean-hez.
        Nincs tobb O(N) iteralas, nincs tobb GPU-ejhezel.
    """

    # [C3 FIX] Az alapertelmezett stagnacio-ablak fele, amit a _recent_deque tarol.
    _DEFAULT_STAGNATION_HALF: int = 50

    def __init__(
        self,
        window_size: int = 100_000,
        num_players: int = 6,
    ) -> None:
        self.window_size: int = window_size
        self.num_players: int = num_players
        self._window: deque[HandRecord] = deque(maxlen=window_size)

        # Kumulativ szamlalok O(1) metrika szamitashoz
        self._vpip_count: int = 0
        self._pfr_count: int = 0
        self._three_bet_count: int = 0
        self._postflop_bets_total: int = 0
        self._postflop_calls_total: int = 0
        self._showdown_count: int = 0
        self._saw_flop_count: int = 0
        self._reward_sum: float = 0.0
        self._total_hands: int = 0

        # [FIX C3 / CV-4] True O(1) stagnation detection.
        # _recent_deque stores the last N rewards (N = _DEFAULT_STAGNATION_HALF).
        # _recent_sum maintains the running sum of _recent_deque, updated
        # incrementally in record_hand() — eviction is O(1) via deque maxlen,
        # and mean computation avoids sum() iteration entirely.
        self._recent_deque: deque[float] = deque(
            maxlen=self._DEFAULT_STAGNATION_HALF
        )
        self._recent_sum: float = 0.0   # [CV-4] running sum for O(1) mean

        logger.info(
            "TelemetryAnalyzer inicializalva: window=%d, players=%d, "
            "stagnation_deque_maxlen=%d [C3 FIX: O(1) stagnacio]",
            window_size, num_players, self._DEFAULT_STAGNATION_HALF,
        )

    # =========================================================================
    # Leosztas Rogzitese
    # =========================================================================

    def record_hand(self, record: HandRecord) -> None:
        """Egyetlen leosztas HUD adatait rogziti a mozgoablakba.

        [FIX C3] O(1) _recent_deque frissites hozzaadva: a record.reward_bb
        inkrementalisan kerul a csuszoablakos deque-be, lehetove teve az
        O(1) stagnacio detektalast.
        """
        # Ha az ablak tele van, a kiesot kivonjuk
        if len(self._window) == self.window_size:
            old: HandRecord = self._window[0]
            self._vpip_count -= int(old.vpip)
            self._pfr_count -= int(old.pfr)
            self._three_bet_count -= int(old.three_bet)
            self._postflop_bets_total -= old.total_aggressive_actions
            self._postflop_calls_total -= old.total_passive_actions
            self._showdown_count -= int(old.went_to_showdown)
            self._saw_flop_count -= int(old.street_reached >= 1)
            self._reward_sum -= old.reward_bb

        # Uj rekord hozzaadasa
        self._window.append(record)
        self._vpip_count += int(record.vpip)
        self._pfr_count += int(record.pfr)
        self._three_bet_count += int(record.three_bet)
        self._postflop_bets_total += record.total_aggressive_actions
        self._postflop_calls_total += record.total_passive_actions
        self._showdown_count += int(record.went_to_showdown)
        self._saw_flop_count += int(record.street_reached >= 1)
        self._reward_sum += record.reward_bb
        self._total_hands += 1

        # [CV-4 FIX] True O(1) running-sum maintenance.
        # Before appending, subtract the value that will be evicted (if full).
        if len(self._recent_deque) == self._DEFAULT_STAGNATION_HALF:
            self._recent_sum -= self._recent_deque[0]  # evicted element
        self._recent_deque.append(record.reward_bb)
        self._recent_sum += record.reward_bb

        if self._total_hands % 10000 == 0:
            logger.debug(
                "Telemetria: %d leosztas rogzitve (ablak: %d/%d)",
                self._total_hands, len(self._window), self.window_size,
            )

    # =========================================================================
    # Metrika Szamitas
    # =========================================================================

    def get_current_metrics(self) -> dict[str, float]:
        """Kiszamitja az osszes HUD metrikat az aktualis ablakbol."""
        n: int = len(self._window)
        if n == 0:
            logger.warning("Ures telemetriai ablak. Nulla metrikak.")
            return self._empty_metrics()

        vpip: float = (self._vpip_count / n) * 100.0
        pfr: float = (self._pfr_count / n) * 100.0
        three_bet: float = (self._three_bet_count / n) * 100.0

        total_calls: int = max(self._postflop_calls_total, 1)
        af: float = self._postflop_bets_total / total_calls

        wtsd: float = 0.0
        if self._saw_flop_count > 0:
            wtsd = (self._showdown_count / self._saw_flop_count) * 100.0

        win_rate_mbb: float = (self._reward_sum / n) * 1000.0

        metrics: dict[str, float] = {
            "vpip": vpip,
            "pfr": pfr,
            "vpip_pfr_gap": vpip - pfr,
            "three_bet": three_bet,
            "af": af,
            "wtsd": wtsd,
            "win_rate_mbb": win_rate_mbb,
            "hands_in_window": float(n),
            "total_hands": float(self._total_hands),
        }

        logger.debug(
            "HUD Metrikak: VPIP=%.1f%%, PFR=%.1f%%, gap=%.1f, "
            "3Bet=%.1f%%, AF=%.2f, WTSD=%.1f%%, wr=%.1f mbb/h",
            vpip, pfr, vpip - pfr, three_bet, af, wtsd, win_rate_mbb,
        )

        return metrics

    # =========================================================================
    # Anomalia Detektalas
    # =========================================================================

    def detect_anomalies(
        self,
        gto_matrix: dict[str, list[float]],
        thresholds: dict[str, dict[str, float]],
    ) -> list[str]:
        """A GTO matrix es a degeneracios kuszobok alapjan anomaliakat detektal."""
        metrics: dict[str, float] = self.get_current_metrics()
        anomalies: list[str] = []

        if len(self._window) < 1000:
            logger.debug(
                "Tul keves adat az anomalia detektaalshoz (%d < 1000).",
                len(self._window),
            )
            return anomalies

        # Passzivitas detektalas
        pass_thresh = thresholds.get("passivity", {})
        vpip_below: float = pass_thresh.get("vpip_below", 0.0)
        gap_above: float = pass_thresh.get("vpip_pfr_gap_above", 100.0)

        if metrics["vpip"] < vpip_below:
            anomalies.append("passivity")
            logger.warning(
                "PATOLOGIA: Passzivitas! VPIP=%.1f%% < %.1f%%",
                metrics["vpip"], vpip_below,
            )
        elif metrics["vpip_pfr_gap"] > gap_above:
            anomalies.append("passivity")
            logger.warning(
                "PATOLOGIA: Passzivitas (gap)! VPIP-PFR=%.1f > %.1f",
                metrics["vpip_pfr_gap"], gap_above,
            )

        # Maniac / All-in Spam detektalas
        maniac_thresh = thresholds.get("maniac", {})
        pfr_above: float = maniac_thresh.get("pfr_above", 100.0)
        three_bet_above: float = maniac_thresh.get("three_bet_above", 100.0)
        af_above: float = maniac_thresh.get("af_above", 100.0)

        if metrics["pfr"] > pfr_above:
            anomalies.append("maniac")
            logger.warning(
                "PATOLOGIA: Maniac! PFR=%.1f%% > %.1f%%",
                metrics["pfr"], pfr_above,
            )
        elif metrics["three_bet"] > three_bet_above:
            anomalies.append("maniac")
            logger.warning(
                "PATOLOGIA: Maniac (3-Bet)! 3Bet=%.1f%% > %.1f%%",
                metrics["three_bet"], three_bet_above,
            )
        elif metrics["af"] > af_above:
            anomalies.append("maniac")
            logger.warning(
                "PATOLOGIA: Maniac (AF)! AF=%.2f > %.2f",
                metrics["af"], af_above,
            )

        if anomalies:
            logger.info("Detektalt anomaliak: %s", anomalies)
        else:
            logger.debug("Nincs anomalia detektalva.")

        return anomalies

    def check_stagnation(
        self,
        reward_window: int = 50,
        threshold: float = 0.001,
    ) -> bool:
        """O(1) stagnacio detekcio elore fenntartott csuszoablakos deque-vel.

        [FIX C3] A korabbi implementacio itertools.islice()-t hasznalt egy
        100,000 meretu deque-n, ami O(N) muvelet. A javitas:

        1. _recent_deque (maxlen=_DEFAULT_STAGNATION_HALF) tartja az utolso
           N jutalom erteket inkrementalisan, O(1) frissitessel record_hand()-ben.
        2. A 'recent' atlag az _recent_deque osszege / hossza — O(1).
        3. A 'window' atlag a mar O(1) _reward_sum / len(_window) — O(1).
        4. Teljes muveletigeny: O(1) (volt: O(N)).

        Megjegyzes: A _recent_deque maxlen-je rogzitett (_DEFAULT_STAGNATION_HALF).
        Ha a hivasi reward_window kulonbozik ettol, egy figyelmeztetes logolodik,
        de a szamitas az eloallott deque meretehez adaptálódik.

        Args:
            reward_window: A vizsgalt utolso leosztas szam (alap: 50).
            threshold: Minimalis jutalom-valtozasi kuszob.

        Returns:
            True ha stagnacio detektalt (O(1) ido).
        """
        if len(self._window) < reward_window * 2:
            return False

        if reward_window != self._DEFAULT_STAGNATION_HALF:
            logger.debug(
                "check_stagnation: reward_window=%d kulonbozik a default %d-tol. "
                "A _recent_deque az elore beallitott meretet hasznalja.",
                reward_window, self._DEFAULT_STAGNATION_HALF,
            )

        if len(self._recent_deque) < self._DEFAULT_STAGNATION_HALF:
            return False

        # [CV-4 FIX] True O(1) recent mean — uses the incrementally maintained
        # _recent_sum, avoiding sum() iteration over the deque entirely.
        recent_mean = self._recent_sum / len(self._recent_deque)

        # O(1) teljes ablak atlag — _reward_sum mar karbantartva record_hand()-ben
        total_mean = self._reward_sum / max(len(self._window), 1)

        delta: float = abs(recent_mean - total_mean)
        is_stagnant: bool = delta < threshold

        if is_stagnant:
            logger.warning(
                "STAGNACIO detektalt: delta=%.6f < %.6f "
                "(recent_mean=%.4f, window_avg=%.4f) [O(1) ellenorzes]",
                delta, threshold, recent_mean, total_mean,
            )

        return is_stagnant

    # =========================================================================
    # GTO Tavolsag Szamitas
    # =========================================================================

    def compute_gto_distance(
        self, gto_targets: dict[str, list[float]]
    ) -> dict[str, float]:
        """Kiszamitja az aktualis metrikak tavolsagat a GTO celertekektol."""
        metrics: dict[str, float] = self.get_current_metrics()
        distances: dict[str, float] = {}

        metric_mapping: dict[str, str] = {
            "vpip": "vpip",
            "pfr": "pfr",
            "three_bet": "three_bet",
            "af": "af",
            "wtsd": "wtsd",
        }

        for gto_key, metric_key in metric_mapping.items():
            if gto_key not in gto_targets:
                continue
            target_range: list[float] = gto_targets[gto_key]
            value: float = metrics.get(metric_key, 0.0)

            if target_range[0] <= value <= target_range[1]:
                distances[gto_key] = 0.0
            elif value < target_range[0]:
                distances[gto_key] = target_range[0] - value
            else:
                distances[gto_key] = value - target_range[1]

        distances["total"] = sum(distances.values())
        return distances

    # =========================================================================
    # Segedmetodusok
    # =========================================================================

    def reset(self) -> None:
        """Torli az egesz telemetriai ablakot es a szamlalokat."""
        self._window.clear()
        self._vpip_count = 0
        self._pfr_count = 0
        self._three_bet_count = 0
        self._postflop_bets_total = 0
        self._postflop_calls_total = 0
        self._showdown_count = 0
        self._saw_flop_count = 0
        self._reward_sum = 0.0
        self._total_hands = 0
        # [CV-4 FIX] Clear both the deque and its running sum
        self._recent_deque.clear()
        self._recent_sum = 0.0
        logger.info("Telemetria ablak resetelve (_recent_deque + _recent_sum torolt).")

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {
            "vpip": 0.0, "pfr": 0.0, "vpip_pfr_gap": 0.0,
            "three_bet": 0.0, "af": 0.0, "wtsd": 0.0,
            "win_rate_mbb": 0.0, "hands_in_window": 0.0,
            "total_hands": 0.0,
        }
