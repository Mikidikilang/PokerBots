"""
Equity Kalkulátor (equity.py).

Ez a modul a pre-flop és post-flop kézerő (hand equity) valószínűségek
matematikai kalkulációját valósítja meg. Az equity a játékos nyerési
valószínűsége egy adott kártyakombinációval az ellenfelek lehetséges
tartományaival (ranges) szemben, figyelembe véve a hátralevő közös
lapok összes lehetséges kimenetelét.

Két fő módszer:
    1. Monte Carlo szimuláció: Véletlenszerű mintavételezéssel becsüli
       az equityt. Gyors, de sztochasztikus zajjal rendelkezik.
    2. Enumeráció (Exhaustive): Az összes lehetséges board runout
       kiértékelése. Pontos, de kombinatorikusan robbanó.

A modul a Treys könyvtárat használja a gyors 5/7 kártyás kézértékeléshez.

Architektúra szerződés:
    - Bemenet: Kártyajelölések (str formátum: "AS", "KH", stb.)
    - Kimenet: Nyerési valószínűség [0.0, 1.0] tartományban
    - A reward_shaper.py és a telemetry.py modulok használják

Hivatkozások:
    - Treys: https://github.com/ihendley/treys
    - Specifikáció: equity.py — Pre-flop és post-flop nyerési valószínűségek
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Konstansok
# =============================================================================

RANK_MAP_TREYS: dict[str, str] = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8",
    "9": "9", "T": "T", "J": "J", "Q": "Q", "K": "K", "A": "A",
}
"""Érték leképezés a belső formátumból a Treys formátumba."""

SUIT_MAP_TREYS: dict[str, str] = {"S": "s", "H": "h", "D": "d", "C": "c"}
"""Szín leképezés a belső formátumból a Treys formátumba (kisbetű)."""

DEFAULT_MC_ITERATIONS: int = 10_000
"""Alapértelmezett Monte Carlo iterációszám."""


# =============================================================================
# Kártya Konverziós Segédfüggvények
# =============================================================================

def _card_to_treys_str(card: str) -> str:
    """A belső kártyaformátumot Treys-kompatibilis formátumba konvertálja.

    A belső formátum: "SR" (pl. "SA" = Ász Pikk, SuitRank format)
    A Treys formátum: "Rs" (pl. "As" = Ász Pikk, kisbetűs szín)

    Args:
        card: Kétkarakteres kártyajelölés SuitRank belső formátumban.

    Returns:
        A kártya Treys-kompatibilis jelölése.

    Raises:
        ValueError: Ha a kártya formátuma érvénytelen.
    """
    card = card.strip().upper()
    if len(card) != 2:
        raise ValueError(f"Érvénytelen kártyaformátum: '{card}'")

    suit_char: str = card[0]
    rank_char: str = card[1]

    if rank_char not in RANK_MAP_TREYS:
        raise ValueError(f"Ismeretlen kártyaérték: '{rank_char}'")
    if suit_char not in SUIT_MAP_TREYS:
        raise ValueError(f"Ismeretlen kártyaszín: '{suit_char}'")

    return RANK_MAP_TREYS[rank_char] + SUIT_MAP_TREYS[suit_char]


def _build_full_deck() -> list[str]:
    """Felépíti a teljes 52 lapos paklit a belső SuitRank formátumban.

    Returns:
        52 elemű lista kétkarakteres kártyajelölésekkel (pl. "SA", "HK", stb.).
    """
    ranks: str = "23456789TJQKA"
    suits: str = "SHDC"
    return [s + r for s in suits for r in ranks]


# =============================================================================
# Fő EquityCalculator Osztály
# =============================================================================

class EquityCalculator:
    """Pre-flop és post-flop nyerési valószínűségek kalkulátora.

    Az equity kiszámítása Monte Carlo szimulációval történik: a
    kalkulátor véletlenszerűen kiosztja a hiányzó közös lapokat,
    értékeli a kezeket, és a nyerési arányból becsüli a valószínűséget.

    A Treys könyvtár opcionális: ha nem elérhető, egy egyszerűsített
    belső kézrangsorolás lép életbe (kevésbé pontos, de működőképes).

    Example:
        >>> calc = EquityCalculator()
        >>> equity = calc.calculate_equity(
        ...     hole_cards=["SA", "SK"],
        ...     community_cards=["ST", "SJ", "SQ"],
        ...     num_opponents=1,
        ...     iterations=10000,
        ... )
        >>> print(f"Equity: {equity:.2%}")  # ~95%+

    Attributes:
        _evaluator: A Treys Evaluator példány (ha elérhető).
        _treys_available: True ha a Treys könyvtár importálható.
    """

    def __init__(self) -> None:
        """Inicializálja az EquityCalculator-t és megpróbálja betölteni a Treys-t."""
        self._treys_available: bool = False
        self._evaluator: object | None = None
        self._treys_card_class: type | None = None

        try:
            from treys import Evaluator, Card  # type: ignore[import-untyped]
            self._evaluator = Evaluator()
            self._treys_card_class = Card
            self._treys_available = True
            logger.info("EquityCalculator inicializálva Treys evaluator-ral.")
        except ImportError:
            logger.warning(
                "A 'treys' könyvtár nem elérhető. Egyszerűsített belső evaluator "
                "kerül alkalmazásra. A pontosabb equity számításhoz telepítsd: "
                "pip install treys"
            )

    # =========================================================================
    # Publikus API
    # =========================================================================

    def calculate_equity(
        self,
        hole_cards: list[str],
        community_cards: list[str] | None = None,
        num_opponents: int = 1,
        iterations: int = DEFAULT_MC_ITERATIONS,
    ) -> float:
        """Kiszámítja a saját kézzel elért nyerési valószínűséget Monte Carlo módszerrel.

        Az algoritmus véletlenszerűen kiosztja az ellenfelek kártyáit és a
        hiányzó közös lapokat, majd a Treys evaluator-ral kiértékeli az
        összes kezet. A nyerési arány adja az equity becslést.

        Args:
            hole_cards: Saját 2 lap (pl. ["SA", "HK"]).
            community_cards: A már kiosztott közös lapok (0-5 elem).
                            None vagy üres lista = pre-flop.
            num_opponents: Az aktív ellenfelek száma (1-8).
            iterations: Monte Carlo iterációk száma. Magasabb = pontosabb.

        Returns:
            Nyerési valószínűség a [0.0, 1.0] tartományban.
            Döntetlen esetén a fél pot is beleszámít (equity share).

        Raises:
            ValueError: Ha a hole_cards nem pontosan 2 kártyát tartalmaz.
            ValueError: Ha a community_cards több mint 5 kártyát tartalmaz.
        """
        if len(hole_cards) != 2:
            raise ValueError(
                f"Pontosan 2 saját lap szükséges, de {len(hole_cards)} érkezett."
            )

        community: list[str] = community_cards or []
        if len(community) > 5:
            raise ValueError(
                f"Legfeljebb 5 közös lap adható meg, de {len(community)} érkezett."
            )

        if not 1 <= num_opponents <= 8:
            raise ValueError(
                f"num_opponents értéke {num_opponents}, de 1 és 8 között kell legyen."
            )

        logger.debug(
            "Equity számítás: hole=%s, community=%s, opponents=%d, iterations=%d",
            hole_cards, community, num_opponents, iterations,
        )

        if self._treys_available:
            equity: float = self._monte_carlo_treys(
                hole_cards, community, num_opponents, iterations
            )
        else:
            equity = self._monte_carlo_simple(
                hole_cards, community, num_opponents, iterations
            )

        logger.info(
            "Equity eredmény: %.4f (%.1f%%) | %s vs %d ellenfél, board=%s",
            equity, equity * 100, hole_cards, num_opponents, community,
        )

        return equity

    def evaluate_hand_strength(
        self,
        hole_cards: list[str],
        community_cards: list[str],
    ) -> int:
        """Egy konkrét kézkombináció abszolút erejét értékeli.

        A Treys evaluator-nál alacsonyabb szám = erősebb kéz (1 = Royal Flush).

        Args:
            hole_cards: Saját 2 lap.
            community_cards: Közös lapok (3-5 elem szükséges).

        Returns:
            Kézrangsor szám (1 = legerősebb, 7462 = leggyengébb Treys-ben).
            Ha a Treys nem elérhető, egy heurisztikus pontszám [0, 7462].

        Raises:
            ValueError: Ha nincs legalább 3 közös lap.
        """
        if len(community_cards) < 3:
            raise ValueError(
                f"Legalább 3 közös lap szükséges az értékeléshez, "
                f"de csak {len(community_cards)} érkezett."
            )

        if self._treys_available:
            return self._evaluate_treys(hole_cards, community_cards)
        else:
            return self._evaluate_simple(hole_cards, community_cards)

    # =========================================================================
    # Treys Alapú Implementáció
    # =========================================================================

    def _monte_carlo_treys(
        self,
        hole_cards: list[str],
        community_cards: list[str],
        num_opponents: int,
        iterations: int,
    ) -> float:
        """Monte Carlo equity számítás a Treys evaluator-ral.

        Args:
            hole_cards: Saját lapok.
            community_cards: Kiosztott közös lapok.
            num_opponents: Ellenfelek száma.
            iterations: Szimulációk száma.

        Returns:
            Becsült equity [0.0, 1.0].
        """
        assert self._treys_card_class is not None
        assert self._evaluator is not None

        Card = self._treys_card_class

        # Konvertálás Treys formátumba
        try:
            hero_hand: list[int] = [
                Card.new(_card_to_treys_str(c)) for c in hole_cards
            ]
            board_cards: list[int] = [
                Card.new(_card_to_treys_str(c)) for c in community_cards
            ]
        except (KeyError, ValueError) as exc:
            logger.error("Kártyakonverziós hiba: %s", exc)
            raise ValueError(f"Érvénytelen kártya a bemeneten: {exc}") from exc

        # Megmaradt pakli (kizárva a már kiosztott lapokat)
        used_cards: set[int] = set(hero_hand) | set(board_cards)
        full_deck: list[int] = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]
        remaining_deck: list[int] = [c for c in full_deck if c not in used_cards]

        cards_needed_for_board: int = 5 - len(board_cards)
        cards_needed_per_opponent: int = 2
        total_cards_needed: int = cards_needed_for_board + (num_opponents * cards_needed_per_opponent)

        if len(remaining_deck) < total_cards_needed:
            logger.error(
                "Nincs elég kártya a szimulációhoz: %d szükséges, %d elérhető",
                total_cards_needed, len(remaining_deck),
            )
            return 0.5  # Fallback: 50% equity

        wins: float = 0.0
        ties: float = 0.0

        for i in range(iterations):
            random.shuffle(remaining_deck)
            idx: int = 0

            # Board kiegészítés
            sim_board: list[int] = list(board_cards)
            sim_board.extend(remaining_deck[idx: idx + cards_needed_for_board])
            idx += cards_needed_for_board

            # Hero kéz értékelés
            hero_score: int = self._evaluator.evaluate(sim_board, hero_hand)  # type: ignore[union-attr]

            # Ellenfelek értékelése
            hero_wins: bool = True
            is_tie: bool = False

            for _ in range(num_opponents):
                opp_hand: list[int] = remaining_deck[idx: idx + 2]
                idx += 2
                opp_score: int = self._evaluator.evaluate(sim_board, opp_hand)  # type: ignore[union-attr]

                if opp_score < hero_score:
                    # Treys: alacsonyabb szám = erősebb kéz
                    hero_wins = False
                    is_tie = False
                    break
                elif opp_score == hero_score:
                    is_tie = True

            if hero_wins and not is_tie:
                wins += 1.0
            elif is_tie:
                ties += 1.0

        equity: float = (wins + ties * 0.5) / iterations

        logger.debug(
            "MC Treys: %d iteráció → wins=%.0f, ties=%.0f, equity=%.4f",
            iterations, wins, ties, equity,
        )

        return equity

    def _evaluate_treys(
        self, hole_cards: list[str], community_cards: list[str]
    ) -> int:
        """Egy konkrét kéz értékelése a Treys evaluator-ral.

        Args:
            hole_cards: Saját lapok (2 db).
            community_cards: Közös lapok (3-5 db).

        Returns:
            Treys hand rank (1 = legerősebb, 7462 = leggyengébb).
        """
        assert self._treys_card_class is not None
        assert self._evaluator is not None

        Card = self._treys_card_class

        hand: list[int] = [Card.new(_card_to_treys_str(c)) for c in hole_cards]
        board: list[int] = [Card.new(_card_to_treys_str(c)) for c in community_cards]

        score: int = self._evaluator.evaluate(board, hand)  # type: ignore[union-attr]

        logger.debug(
            "Treys kézértékelés: %s + %s → score=%d",
            hole_cards, community_cards, score,
        )
        return score

    # =========================================================================
    # Egyszerűsített Fallback Implementáció (Treys nélkül)
    # =========================================================================

    def _monte_carlo_simple(
        self,
        hole_cards: list[str],
        community_cards: list[str],
        num_opponents: int,
        iterations: int,
    ) -> float:
        """Egyszerűsített Monte Carlo equity becslés Treys nélkül.

        Ez a fallback implementáció egy heurisztikus kézerő-rangsorolást
        használ a pontos Treys evaluator helyett. Kevésbé pontos, de
        nem igényel külső függőséget.

        Args:
            hole_cards: Saját lapok.
            community_cards: Kiosztott közös lapok.
            num_opponents: Ellenfelek száma.
            iterations: Szimulációk száma.

        Returns:
            Becsült equity [0.0, 1.0].
        """
        logger.debug("Egyszerűsített MC equity számítás (Treys nem elérhető)")

        full_deck: list[str] = _build_full_deck()
        used: set[str] = set(hole_cards) | set(community_cards)
        remaining: list[str] = [c for c in full_deck if c not in used]

        cards_needed_for_board: int = 5 - len(community_cards)

        wins: float = 0.0
        ties: float = 0.0

        for _ in range(iterations):
            random.shuffle(remaining)
            idx: int = 0

            sim_board: list[str] = list(community_cards)
            sim_board.extend(remaining[idx: idx + cards_needed_for_board])
            idx += cards_needed_for_board

            hero_score: int = self._evaluate_simple(hole_cards, sim_board)

            hero_wins: bool = True
            is_tie: bool = False

            for _ in range(num_opponents):
                opp_hand: list[str] = remaining[idx: idx + 2]
                idx += 2
                opp_score: int = self._evaluate_simple(opp_hand, sim_board)

                if opp_score < hero_score:
                    hero_wins = False
                    break
                elif opp_score == hero_score:
                    is_tie = True

            if hero_wins and not is_tie:
                wins += 1.0
            elif is_tie:
                ties += 1.0

        return (wins + ties * 0.5) / iterations

    @staticmethod
    def _evaluate_simple(
        hole_cards: list[str],
        community_cards: list[str],
    ) -> int:
        """Heurisztikus kézerő-rangsorolás a Treys könyvtár nélkül.

        Egyszerűsített pontozási rendszer: a legmagasabb pár/kicker
        kombinációk alapján rangsorol. NEM implementálja a teljes
        póker kézrangsorolást (pl. flush, straight detektálás korlátozott).

        FONTOS: Ez egy fallback — produkciós használathoz a Treys
        könyvtár telepítése javasolt.

        Args:
            hole_cards: Saját lapok (2 db) SuitRank formátumban.
            community_cards: Közös lapok (3-5 db) SuitRank formátumban.

        Returns:
            Heurisztikus pontszám (alacsonyabb = erősebb, konzisztens a Treys-szel).
        """
        rank_order: str = "23456789TJQKA"
        all_cards: list[str] = hole_cards + community_cards

        # Értékek kinyerése (0-12 index, magasabb = erősebb)
        # SuitRank format: card[0] is suit, card[1] is rank
        ranks: list[int] = []
        suits: list[str] = []
        for card in all_cards:
            card = card.strip().upper()
            if len(card) >= 2 and card[1] in rank_order:
                ranks.append(rank_order.index(card[1]))
                suits.append(card[0])

        if not ranks:
            return 7462  # Leggyengébb kéz

        # Rangszámolás
        rank_counts: dict[int, int] = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        counts: list[tuple[int, int]] = sorted(
            rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True
        )

        # Egyszerűsített pontozás (inverz — alacsonyabb = jobb)
        best_count: int = counts[0][1]
        best_rank: int = counts[0][0]
        second_count: int = counts[1][1] if len(counts) > 1 else 0
        second_rank: int = counts[1][0] if len(counts) > 1 else 0

        # Flush detektálás (egyszerűsített)
        suit_counts: dict[str, int] = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        has_flush: bool = any(c >= 5 for c in suit_counts.values())

        # Straight detektálás (egyszerűsített)
        unique_ranks: list[int] = sorted(set(ranks))
        has_straight: bool = False
        if len(unique_ranks) >= 5:
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i + 4] - unique_ranks[i] == 4:
                    has_straight = True
                    break
            # A-2-3-4-5 (wheel) speciális eset
            if {12, 0, 1, 2, 3}.issubset(set(ranks)):
                has_straight = True

        # Pontszám kiszámítása (alacsonyabb = erősebb)
        if has_straight and has_flush:
            score = 100 - best_rank  # Straight Flush
        elif best_count == 4:
            score = 200 - best_rank  # Quads
        elif best_count == 3 and second_count >= 2:
            score = 300 - best_rank  # Full House
        elif has_flush:
            score = 400 - best_rank  # Flush
        elif has_straight:
            score = 500 - best_rank  # Straight
        elif best_count == 3:
            score = 600 - best_rank  # Trips
        elif best_count == 2 and second_count == 2:
            score = 700 - max(best_rank, second_rank)  # Two Pair
        elif best_count == 2:
            score = 800 - best_rank  # One Pair
        else:
            score = 900 - best_rank  # High Card

        return max(1, score)

    # =========================================================================
    # Segédmetódusok
    # =========================================================================

    def is_treys_available(self) -> bool:
        """Visszaadja, hogy a Treys evaluator elérhető-e.

        Returns:
            True, ha a Treys könyvtár sikeresen importálva lett.
        """
        return self._treys_available

    @staticmethod
    def get_preflop_hand_category(hole_cards: list[str]) -> str:
        """Meghatározza a pre-flop kézkategóriát (tier rendszer).

        A kézkategóriák a standard póker pre-flop chartok alapján:
            - "premium": AA, KK, QQ, AKs
            - "strong": JJ, TT, AQs, AKo, AQo
            - "playable": 99-22, suited connectors, broadways
            - "marginal": egyéb
            - "trash": nagyon gyenge kezek

        Args:
            hole_cards: A 2 saját lap SuitRank formátumban (pl. ["SA", "HK"]).

        Returns:
            A kézkategória neve szövegesen.
        """
        if len(hole_cards) != 2:
            return "unknown"

        rank_order: str = "23456789TJQKA"

        c1: str = hole_cards[0].strip().upper()
        c2: str = hole_cards[1].strip().upper()

        # SuitRank format: card[0] is suit, card[1] is rank
        r1_idx: int = rank_order.index(c1[1]) if len(c1) > 1 and c1[1] in rank_order else -1
        r2_idx: int = rank_order.index(c2[1]) if len(c2) > 1 and c2[1] in rank_order else -1
        suited: bool = c1[0] == c2[0]

        high: int = max(r1_idx, r2_idx)
        low: int = min(r1_idx, r2_idx)
        is_pair: bool = r1_idx == r2_idx

        # Premium: AA, KK, QQ, AKs
        if is_pair and high >= 10:  # QQ+
            return "premium"
        if high == 12 and low == 11 and suited:  # AKs
            return "premium"

        # Strong: JJ, TT, AQs, AKo, AQo
        if is_pair and high >= 8:  # TT, JJ
            return "strong"
        if high == 12 and low >= 10:  # AQ+
            return "strong"

        # Playable: 99-22, suited broadways, suited connectors
        if is_pair:
            return "playable"
        if suited and high >= 8 and low >= 7:  # Suited broadways
            return "playable"
        if suited and abs(high - low) == 1 and low >= 4:  # Suited connectors 56s+
            return "playable"

        # Marginal
        if high >= 10 and low >= 6:
            return "marginal"
        if suited and high >= 10:
            return "marginal"

        return "trash"
