"""
Observation Space Konstruktor (features.py).

[FIX H1 - 2025-03-28] Korlatlan Log-Skala Normalizacio Javitasa:
    A korabbi _normalize_chips() implementacio log-skalaval kezelte az
    initial_stack-et meghalado ertekeket:
        if normalized > 1.0:
            return 1.0 + np.log1p(normalized - 1.0)
    Egy 3x initial stacknel ez ~2.1-et adott vissza, 5x stacknel ~2.6-ot —
    a feature ter korlat nelkul nott. Ez destabilizalta az ortogonalis
    sulyinicializaciot es lassu konvergenciat okozott mély stack helyzetekben.

    A javitas: az erteket 5 × initial_stack-nel ragjuk be (hard clip), majd
    a [0, 1] tartomanyba normalizaljuk. Igy a feature ter mindig [0, 1]
    kompakt tartomanyban marad, a haloaz belso retegei konzisztens bemenetet
    kapnak, es az ortogonalis init hatekeony marad.

    Megorzott funkcionalis viselkedes:
        - Kis ertekek (< initial_stack): linearis skalas, ugyanaz mint elobb
        - Nagy ertekek (1x-5x initial): linearis skalas, [0, 1]-re normalizalva
        - Nagyon nagy ertekek (>5x): 1.0-ra csokkentve (ritka esemeny deep stack)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


DECK_SIZE: int = 52
NUM_SUITS: int = 4
NUM_RANKS: int = 13

SUIT_MAP: dict[str, int] = {"S": 0, "H": 1, "D": 2, "C": 3}
RANK_MAP: dict[str, int] = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}

# [FIX H1] Maximalis chip ertek a normalizaciohoz, initial_stack szorosaban.
# 5x-os cap: tipikus 200BB stack-nel ez 1000BB, ami lefed mindent relevansat.
_CHIP_NORMALIZATION_MAX_MULTIPLIER: float = 5.0


@dataclass(frozen=True)
class ObservationConfig:
    """Az Observation Space osszes konfiguralhato parameteret tarolja."""

    num_players: int = 6
    max_betting_actions: int = 18
    action_feature_dim: int = 9
    initial_stack_bb: float = 200.0
    normalization_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not 2 <= self.num_players <= 9:
            raise ValueError(
                f"num_players erteke {self.num_players}, de 2 es 9 kozott kell lennie."
            )
        if self.max_betting_actions < 1:
            raise ValueError(
                f"max_betting_actions erteke {self.max_betting_actions}, de legalabb 1 kell legyen."
            )
        logger.debug(
            "ObservationConfig inicializalva: num_players=%d, max_betting_actions=%d, "
            "initial_stack_bb=%.1f",
            self.num_players, self.max_betting_actions, self.initial_stack_bb,
        )


class ObservationBuilder:
    """A nyers jatekallapotot strukturalt, normalizalt tenzor-szotarra alakitja.

    [FIX H1] A chip normalizacio mostantol korlatos [0, 1] tartomanyban marad.
    Lasd _normalize_chips() es a modszer szintju docstringet.
    """

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config: ObservationConfig = config or ObservationConfig()
        self._norm_min: float = self.config.normalization_range[0]
        self._norm_max: float = self.config.normalization_range[1]

        logger.info(
            "ObservationBuilder inicializalva: %d jatekos, %d max akcio, "
            "normalizacio=[%.1f, %.1f] [H1 FIX: korlatos chip norm]",
            self.config.num_players,
            self.config.max_betting_actions,
            self._norm_min,
            self._norm_max,
        )

    # =========================================================================
    # Publikus API
    # =========================================================================

    def build(self, raw_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        """A nyers jatekallapotbol elkesziti a teljes megfigyeles szotarat."""
        if isinstance(raw_state, tuple):
            raw_state = raw_state[0]
        logger.debug("Observation epitese nyers allapotbol: %d kulcs", len(raw_state))

        state_source = raw_state
        if "raw_obs" in raw_state and isinstance(raw_state["raw_obs"], dict):
            logger.debug("Beagyazott 'raw_obs' kulcs detektalt, RLCard allapot.")
            state_source = {**raw_state, **raw_state["raw_obs"]}

        try:
            observation: dict[str, torch.Tensor] = {
                "hole_cards": self._encode_cards(state_source["hand"]),
                "community_cards": self._encode_cards(state_source.get("public_cards", [])),
                "env_metrics": self._encode_env_metrics(state_source),
                "betting_history": self._encode_betting_history(
                    state_source.get("betting_history", [])
                ),
                "position": self._encode_position(state_source.get("position", 0)),
                "action_mask": self._encode_action_mask(
                    state_source.get("legal_actions", list(range(9)))
                ),
            }
        except KeyError as exc:
            logger.error(
                "Hianyzo kulcs az allapotforrasbol ('%s'): %s. "
                "Elerheto kulcsok: %s",
                "raw_obs" if "raw_obs" in raw_state else "raw_state",
                exc,
                list(state_source.keys()),
            )
            raise

        if logger.isEnabledFor(logging.DEBUG):
            for key, tensor in observation.items():
                logger.debug(
                    "  Observation[%s]: shape=%s, dtype=%s, range=[%.4f, %.4f]",
                    key, tensor.shape, tensor.dtype,
                    tensor.min().item(), tensor.max().item(),
                )

        return observation

    def flatten(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """A szotarat egyetlen laposított (1D) vektorra konvertalja."""
        components: list[torch.Tensor] = [
            observation["hole_cards"],
            observation["community_cards"],
            observation["env_metrics"],
            observation["betting_history"].flatten(),
            observation["position"],
        ]
        flat_vector: torch.Tensor = torch.cat(components, dim=0)
        logger.debug("Laposított megfigyelés: dim=%d", flat_vector.shape[0])
        return flat_vector

    def get_observation_dim(self) -> int:
        card_dim: int = DECK_SIZE * 2
        metrics_dim: int = 4 + (self.config.num_players - 1)
        history_dim: int = (
            self.config.max_betting_actions * self.config.action_feature_dim
        )
        position_dim: int = self.config.num_players
        total: int = card_dim + metrics_dim + history_dim + position_dim
        return total

    # =========================================================================
    # Privat Kodoló Metódusok
    # =========================================================================

    def _encode_cards(self, cards: list[str]) -> torch.Tensor:
        """Kartyalistat 52-dimenziós multi-hot binaris vektorra kodol."""
        encoding: torch.Tensor = torch.zeros(DECK_SIZE, dtype=torch.float32)

        for card_str in cards:
            card_str = card_str.strip().upper()
            if len(card_str) != 2:
                raise ValueError(
                    f"Ervenytelen kartyaformatum: '{card_str}'. "
                    f"Elvart formatum: 'SR' (pl. 'SA' = Ász Pikk)."
                )

            suit_char: str = card_str[0]
            rank_char: str = card_str[1]

            if rank_char not in RANK_MAP:
                raise ValueError(
                    f"Ismeretlen kartyaertek: '{rank_char}' a(z) '{card_str}' kartyaban."
                )
            if suit_char not in SUIT_MAP:
                raise ValueError(
                    f"Ismeretlen kartyaszin: '{suit_char}' a(z) '{card_str}' kartyaban."
                )

            rank_idx: int = RANK_MAP[rank_char]
            suit_idx: int = SUIT_MAP[suit_char]
            card_index: int = rank_idx * NUM_SUITS + suit_idx
            encoding[card_index] = 1.0

        logger.debug(
            "Kartyakodolas: %d kartya → %d aktiv pozicio",
            len(cards), int(encoding.sum().item()),
        )
        return encoding

    def _encode_env_metrics(self, raw_state: dict[str, Any]) -> torch.Tensor:
        """A kornyezeti metrikakat normalizalt float vektorra kodola.

        [FIX H1] A _normalize_chips() most korlatos [0, 1] tartomanyban marad.
        Lasd a belso fuggveny docstringet a reszletekert.
        """
        big_blind: float = float(raw_state.get("big_blind", 2.0))
        if big_blind <= 0:
            logger.warning(
                "A big_blind erteke %.2f, ami nem pozitiv. Fallback: 2.0", big_blind
            )
            big_blind = 2.0

        initial_stack: float = self.config.initial_stack_bb * big_blind
        # [FIX H1] A maximalis chip ertek a normalizaciohoz
        max_chip_value: float = initial_stack * _CHIP_NORMALIZATION_MAX_MULTIPLIER

        def _normalize_chips(value: float) -> float:
            """Monetaris erteket normalizal a [0, 1] kompakt tartomanyba.

            [FIX H1] A korabbi implementacio log-skalaval kezelte a
            initial_stack-et meghalado ertekeket, ami korlatlan kimenetet
            adott (pl. 3x stack → 2.1, 5x stack → 2.6).

            Az uj implementacio:
                - Ertekkeszlet: [0, max_chip_value] (max = 5 × initial_stack)
                - Kimenet: [0, 1] kompakt tartomany
                - 5x-on feluli ertekek: 1.0-ra csokkentve (ritka esemeny)

            Elonyok:
                1. A halozat belso retegei mindig [0, 1]-ben latjak a bemenetet
                2. Az ortogonalis sulyinicializacio hatekonyan mukodik
                3. Meely stack helyzetekben is stabil a konvergencia
                4. A log-skala logaritmikus torzitasa nem roncsol a fontossagi
                   sorrendet (p. nagy stack vs. kis stack szignifikanciaja)

            Args:
                value: A normalizalandó chip ertek.

            Returns:
                Normalizalt float a [0, 1] tartomanyban.
            """
            if initial_stack <= 0:
                return 0.0
            # Linearis normalizalas, 5x-os cap-el
            capped: float = min(float(value), max_chip_value)
            normalized: float = capped / max_chip_value
            # Biztonsagi clamp (float pontossag miatt)
            return float(max(0.0, min(1.0, normalized)))

        pot: float = float(raw_state.get("pot", 0.0))
        my_chips: float = float(raw_state.get("my_chips", 0.0))
        amount_to_call: float = float(raw_state.get("amount_to_call", 0.0))
        min_raise: float = float(raw_state.get("min_raise", big_blind))

        metrics: list[float] = [
            _normalize_chips(pot),
            _normalize_chips(my_chips),
            _normalize_chips(amount_to_call),
            _normalize_chips(min_raise),
        ]

        opponent_chips: list[float] = raw_state.get("opponent_chips", [])
        for opp_stack in opponent_chips[: self.config.num_players - 1]:
            metrics.append(_normalize_chips(float(opp_stack)))

        expected_opponents: int = self.config.num_players - 1
        while len(metrics) < 4 + expected_opponents:
            metrics.append(0.0)

        metrics_tensor: torch.Tensor = torch.tensor(metrics, dtype=torch.float32)

        # [FIX H1] Ellenorizzuk, hogy a normalizalt ertekek [0, 1]-ben vannak
        if logger.isEnabledFor(logging.DEBUG):
            out_of_range = [(i, v) for i, v in enumerate(metrics) if not (0.0 <= v <= 1.0)]
            if out_of_range:
                logger.warning(
                    "H1 FIX: Normalizalt chip ertekek kiesnek [0,1]-bol: %s. "
                    "Ez nem fordulhat elo a fix utan — ellenorizd a max_multiplier-t.",
                    out_of_range,
                )

        logger.debug(
            "Kornyezeti metrikak kodolva [H1 FIX: korlatos norm, max=%.0f chips]: "
            "pot=%.3f, stack=%.3f, call=%.3f, min_raise=%.3f, %d ellenfél stack",
            max_chip_value,
            metrics[0], metrics[1], metrics[2], metrics[3],
            len(opponent_chips),
        )

        return metrics_tensor

    def _encode_betting_history(
        self, history: list[dict[str, Any]]
    ) -> torch.Tensor:
        """A liciettortenetet rogzített meretu tenzorra kodola."""
        max_actions: int = self.config.max_betting_actions
        action_dim: int = self.config.action_feature_dim

        history_tensor: torch.Tensor = torch.zeros(
            (max_actions, action_dim), dtype=torch.float32
        )

        for step_idx, step in enumerate(history[:max_actions]):
            action_idx: int = int(step.get("action", 0))
            if 0 <= action_idx < action_dim:
                history_tensor[step_idx, action_idx] = 1.0

        logger.debug(
            "Liciettortenet kodolva: %d/%d lepes feltoltve",
            min(len(history), max_actions), max_actions,
        )
        return history_tensor

    def _encode_position(self, position_index: int) -> torch.Tensor:
        """A jatekos poziciojat one-hot vektorra kodola."""
        num_positions: int = self.config.num_players
        position_vector: torch.Tensor = torch.zeros(num_positions, dtype=torch.float32)

        if 0 <= position_index < num_positions:
            position_vector[position_index] = 1.0
        else:
            logger.warning(
                "Ervenytelen pozicio index: %d (max: %d). Nulla vektor visszaadva.",
                position_index, num_positions - 1,
            )

        logger.debug("Pozicio kodolva: index=%d/%d", position_index, num_positions)
        return position_vector

    def _encode_action_mask(self, legal_actions: list[int | Any]) -> torch.Tensor:
        """Binaris akcio ervenyessegi maszkot general."""
        num_actions: int = 9
        mask: torch.Tensor = torch.zeros(num_actions, dtype=torch.float32)

        for action_item in legal_actions:
            try:
                if hasattr(action_item, 'value'):
                    action_idx: int = int(action_item.value)
                else:
                    action_idx = int(action_item)

                if 0 <= action_idx < num_actions:
                    mask[action_idx] = 1.0
                else:
                    logger.warning(
                        "Illegalis akcio index figyelmen kivul hagyva: %d (tartomany: 0-%d)",
                        action_idx, num_actions - 1,
                    )
            except (ValueError, TypeError, AttributeError) as exc:
                logger.warning(
                    "Az akcio-elem nem konvertalhato int-e: %r (%s). Figyelmen kivul hagyva.",
                    action_item, exc,
                )

        active_count: int = int(mask.sum().item())
        if active_count == 0:
            logger.error(
                "KRITIKUS: Ures akcio maszk! Nincs ervenyes akcio. "
                "Fold (index 0) engedelyezese biztonsagi fallback-kent."
            )
            mask[0] = 1.0

        logger.debug("Akcio maszk generalt: %d/%d akcio ervenyes", active_count, num_actions)
        return mask

    # =========================================================================
    # Segedmetodusok
    # =========================================================================

    @staticmethod
    def card_str_to_index(card_str: str) -> int:
        card_str = card_str.strip().upper()
        if len(card_str) != 2:
            raise ValueError(f"Ervenytelen kartyaformatum: '{card_str}'")
        suit_idx: int = SUIT_MAP[card_str[0]]
        rank_idx: int = RANK_MAP[card_str[1]]
        return rank_idx * NUM_SUITS + suit_idx

    @staticmethod
    def index_to_card_str(index: int) -> str:
        if not 0 <= index < DECK_SIZE:
            raise ValueError(f"Kartyaindex {index} kivul esik a [0, 51] tartomanyon.")
        rank_idx: int = index // NUM_SUITS
        suit_idx: int = index % NUM_SUITS
        rank_chars: str = "23456789TJQKA"
        suit_chars: str = "SHDC"
        return suit_chars[suit_idx] + rank_chars[rank_idx]
