"""
HUD Telemetria Analizator (telemetry.py).

A halozat dontesi mintazatait empirikus poker statisztikakka, ugynevezett
Heads-Up Display (HUD) metrikakka konvertalja. Ezek a metrikak szolgalnak
az iranyelv-eloszlas (policy distribution) matematikai proxyjakent.

Ot kritikus metrika mozgoablakos (sliding window) atlagat szamitja ki:

    1. VPIP (Voluntarily Put In Pot): Pre-flop potba helyezes %
    2. PFR (Preflop Raise): Pre-flop emeles %
    3. 3-Bet %: Ujra-emeles gyakorisaga
    4. AF (Aggression Factor): (bet+raise) / call arany post-flop
    5. WTSD (Went to Showdown): Showdown-ig eljutas %

A mozgoablak merete konfiguralhato (tipikusan 100k leosztas).

Hivatkozasok:
    - Specifikacio: telemetry.py — HUD metrikak szamitasa
    - Orchestrator Data: GTO Matrix, degeneracios kuszobok
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Akcio Kategorizalas
# =============================================================================

class ActionCategory(IntEnum):
    """Akcio kategoriak a HUD statisztikak szamitasahoz."""

    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2        # Barmilyen emeles (min-raise, pot-raise, stb.)
    ALL_IN = 3


class StreetPhase(IntEnum):
    """A leosztas fazisai."""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


# =============================================================================
# Egyes Leosztas Rekordja
# =============================================================================

@dataclass
class HandRecord:
    """Per-hand HUD record submitted to TelemetryAnalyzer.record_hand().

    All boolean fields are computed from the raw action sequence so that
    TelemetryAnalyzer only needs to aggregate, never re-derive.

    Streets are encoded as integers:
        0 = Preflop, 1 = Flop, 2 = Turn, 3 = River
    """
    # ── Identification ─────────────────────────────────────────────
    hand_id: int                  # monotonically increasing hand counter
    player_id: int                # seat index of the learning agent
    position: int                 # 0-based position (0=BTN/SB in HU)
    iteration: int                # training iteration this hand belongs to

    # ── Outcome ────────────────────────────────────────────────────
    reward_bb: float              # chip delta / big_blind (signed)
    street_reached: int           # 0-3; last street with any action
    went_to_showdown: bool        # True if river was dealt and reached SD
    won_at_showdown: bool         # True if reward_bb > 0 at showdown

    # ── VPIP / PFR / 3-Bet (preflop aggressiveness) ───────────────
    vpip: bool                    # voluntarily put money in pot preflop
    pfr: bool                     # made at least one preflop raise
    three_bet: bool               # re-raised over an existing preflop raise

    # ── Aggression Factor (AF = aggressive / passive) ─────────────
    total_aggressive_actions: int  # bets + raises across all streets
    total_passive_actions: int     # calls across all streets
    total_folds: int               # folds

    # ── Raw action sequence (for future per-street breakdown) ──────
    actions: list[int] = field(default_factory=list)
    action_streets: list[int] = field(default_factory=list)


# =============================================================================
# Fo Telemetria Analizator
# =============================================================================

class TelemetryAnalyzer:
    """Mozgoablakos HUD metrikak szamitasa es anomalia detektalas.

    A telemetria a leosztas-szintu rekordokat aggregalja egy
    konfiguralhato meretu csuszoablakban, es kiszamitja a
    GTO-relevans statisztikakat.

    Example:
        >>> analyzer = TelemetryAnalyzer(window_size=100_000, num_players=6)
        >>> for hand in hands:
        ...     analyzer.record_hand(hand)
        >>> metrics = analyzer.get_current_metrics()
        >>> anomalies = analyzer.detect_anomalies(gto_matrix, thresholds)

    Attributes:
        window_size: A mozgoablak merete (leosztas szam).
        num_players: Az asztal merete.
        _window: A mozgoablak (deque).
    """

    def __init__(
        self,
        window_size: int = 100_000,
        num_players: int = 6,
    ) -> None:
        """Inicializalja a telemetria analizatort.

        Args:
            window_size: Mozgoablak merete leosztas szamban.
            num_players: Az asztal jatekos szama (a GTO matrix kivalasztasahoz).
        """
        self.window_size: int = window_size
        self.num_players: int = num_players
        self._window: deque[HandRecord] = deque(maxlen=window_size)

        # Kumulativ szamlalok a gyors szamitashoz
        self._vpip_count: int = 0
        self._pfr_count: int = 0
        self._three_bet_count: int = 0
        self._postflop_bets_total: int = 0
        self._postflop_calls_total: int = 0
        self._showdown_count: int = 0
        self._saw_flop_count: int = 0
        self._reward_sum: float = 0.0
        self._total_hands: int = 0

        logger.info(
            "TelemetryAnalyzer inicializalva: window=%d, players=%d",
            window_size, num_players,
        )

    # =========================================================================
    # Leosztas Rogzitese
    # =========================================================================

    def record_hand(self, record: HandRecord) -> None:
        """Egyetlen leosztas HUD adatait rogziti a mozgoablakba.

        Ha az ablak megtelt, a legrgebbi rekord kiesik es a szamlalok
        korrigalodnak (O(1) muvelet).

        Args:
            record: A leosztas adatai.
        """
        # Ha az ablak tele van, a kiesot kivonjuk
        if len(self._window) == self.window_size:
            old: HandRecord = self._window[0]
            self._vpip_count -= int(old.voluntarily_put_in_pot)
            self._pfr_count -= int(old.preflop_raised)
            self._three_bet_count -= int(old.three_betted)
            self._postflop_bets_total -= old.postflop_bets
            self._postflop_calls_total -= old.postflop_calls
            self._showdown_count -= int(old.went_to_showdown)
            self._saw_flop_count -= int(old.saw_flop)
            self._reward_sum -= old.reward

        # Uj rekord hozzaadasa
        self._window.append(record)
        self._vpip_count += int(record.voluntarily_put_in_pot)
        self._pfr_count += int(record.preflop_raised)
        self._three_bet_count += int(record.three_betted)
        self._postflop_bets_total += record.postflop_bets
        self._postflop_calls_total += record.postflop_calls
        self._showdown_count += int(record.went_to_showdown)
        self._saw_flop_count += int(record.saw_flop)
        self._reward_sum += record.reward
        self._total_hands += 1

        if self._total_hands % 10000 == 0:
            logger.debug(
                "Telemetria: %d leosztas rogzitve (ablak: %d/%d)",
                self._total_hands, len(self._window), self.window_size,
            )

    def record_from_actions(
        self,
        actions: list[tuple[int, int]],
        reward: float,
        reached_showdown: bool,
    ) -> None:
        """Nyers akcio-listabol epitkezo gyors rogzito.

        Args:
            actions: (action_index, street_phase) parok listaja.
            reward: A leosztas jutalma.
            reached_showdown: Eljutott-e a showdown-ig.
        """
        record = HandRecord(reward=reward, went_to_showdown=reached_showdown)
        record.actions_taken = actions

        for action_idx, street in actions:
            if street == StreetPhase.PREFLOP:
                if action_idx >= 2:  # Barmilyen emeles
                    record.voluntarily_put_in_pot = True
                    record.preflop_raised = True
                elif action_idx == 1:  # Call
                    record.voluntarily_put_in_pot = True
                # 3-Bet: ha volt mar emeles es ujra emelunk
                # Egyszerusitett heurisztika: ha tobbszor emelt pre-flop
                if action_idx >= 2 and record.preflop_raised:
                    record.three_betted = True
                    record.preflop_raised = True  # Elso emeles is

            elif street >= StreetPhase.FLOP:
                record.saw_flop = True
                if action_idx >= 2:
                    record.postflop_bets += 1
                elif action_idx == 1:
                    record.postflop_calls += 1

        self.record_hand(record)

    # =========================================================================
    # Metrika Szamitas
    # =========================================================================

    def get_current_metrics(self) -> dict[str, float]:
        """Kiszamitja az osszes HUD metrikat az aktualis ablakbol.

        Returns:
            Dict a kovetkezo kulcsokkal:
                - vpip: VPIP szazalekban
                - pfr: PFR szazalekban
                - vpip_pfr_gap: VPIP - PFR kulonbseg
                - three_bet: 3-Bet szazalekban
                - af: Aggression Factor (ratio)
                - wtsd: Went To Showdown szazalekban
                - win_rate_mbb: Nyeresi rata milli-big-blinds/hand
                - hands_in_window: Az ablakban levo leosztas szam
                - total_hands: Osszes rogzitett leosztas
        """
        n: int = len(self._window)
        if n == 0:
            logger.warning("Ures telemetriai ablak. Nulla metrikak.")
            return self._empty_metrics()

        vpip: float = (self._vpip_count / n) * 100.0
        pfr: float = (self._pfr_count / n) * 100.0
        three_bet: float = (self._three_bet_count / n) * 100.0

        # AF: (bets + raises) / calls, calls=0 eseten nagy ertek
        total_calls: int = max(self._postflop_calls_total, 1)
        af: float = self._postflop_bets_total / total_calls

        # WTSD: showdown / saw_flop
        wtsd: float = 0.0
        if self._saw_flop_count > 0:
            wtsd = (self._showdown_count / self._saw_flop_count) * 100.0

        # Win rate: mbb/hand (1 BB = 1000 mbb)
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
        """A GTO matrix es a degeneracios kuszobok alapjan anomaliakat detektal.

        Az ellenorzes a config.yaml-ben definialt asztalmerethez tartozo
        GTO celertekek es kuszobok alapjan tortenik.

        Args:
            gto_matrix: A GTO celertekek szotara (pl. {"vpip": [21.0, 27.0], ...}).
            thresholds: A degeneracios kuszobok (passivity + maniac).

        Returns:
            Detektalt anomaliak neveinek listaja:
                - "passivity": Tulzott passzivitas
                - "maniac": Hiper-agresszio / All-in Spam
                - "stagnation": Jutalom stagnacio
                - "entropy_collapse": Policy entropia osszeomlasa
        """
        metrics: dict[str, float] = self.get_current_metrics()
        anomalies: list[str] = []

        if len(self._window) < 1000:
            logger.debug("Tul keves adat az anomalia detektaalshoz (%d < 1000).", len(self._window))
            return anomalies

        # --- Passzivitas detektalas ---
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

        # --- Maniac / All-in Spam detektalas ---
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
        """Ellenorzi, hogy a jutalom stagnacio-ban van-e.

        Az utolso `reward_window` leosztas jutalmat vizsgalja:
        ha a mozgoatlag valtozasa kisebb mint a threshold, stagnacio.

        Args:
            reward_window: A vizsgalt utolso leosztas szam.
            threshold: A minimalis valtozasi kuoszob.

        Returns:
            True ha stagnacio detektalt.
        """
        if len(self._window) < reward_window * 2:
            return False

        recent = list(self._window)
        half = reward_window
        recent_rewards = [r.reward for r in recent[-half:]]
        older_rewards = [r.reward for r in recent[-2 * half:-half]]

        recent_mean: float = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        older_mean: float = sum(older_rewards) / len(older_rewards) if older_rewards else 0.0
        delta: float = abs(recent_mean - older_mean)

        is_stagnant: bool = delta < threshold

        if is_stagnant:
            logger.warning(
                "STAGNACIO detektalt: delta=%.6f < %.6f (recent=%.4f, older=%.4f)",
                delta, threshold, recent_mean, older_mean,
            )

        return is_stagnant

    # =========================================================================
    # GTO Tavolsag Szamitas
    # =========================================================================

    def compute_gto_distance(
        self, gto_targets: dict[str, list[float]]
    ) -> dict[str, float]:
        """Kiszamitja az aktualis metrikak tavolsagat a GTO celertekektol.

        Minden metrikara: ha az ertek a [min, max] savon belul van,
        a tavolsag 0. Egyebkent a legkozelebbi hatartol valo tavolsag.

        Args:
            gto_targets: GTO celertekek szotara {"vpip": [min, max], ...}.

        Returns:
            Dict metrikankenti tavolsagokkal.
        """
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
        logger.info("Telemetria ablak resetelve.")

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        """Ures metrika szotarat ad vissza."""
        return {
            "vpip": 0.0, "pfr": 0.0, "vpip_pfr_gap": 0.0,
            "three_bet": 0.0, "af": 0.0, "wtsd": 0.0,
            "win_rate_mbb": 0.0, "hands_in_window": 0.0,
            "total_hands": 0.0,
        }
