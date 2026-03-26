"""
Observation Space Konstruktor (features.py).

Ez a modul felelős a nyers játékmotor (RLCard/PettingZoo) állapotadatainak
transzformálásáért egy strukturált, normalizált tenzoriális formátumba,
amelyet a PPO Actor-Critic hálózat közvetlenül feldolgozhat.

A megfigyelési tér (Observation Space) a következő almodulokból épül fel:
    1. Kártyakódolás: 52-dim multi-hot × 2 (hole cards + community cards) = 104 dim
    2. Környezeti metrikák: Normalizált zsetonállások, pot méret, pozíció stb.
    3. Licittörténet: Rögzített méretű tenzor (max_actions × akció_jellemzők)
    4. Pozíció: One-hot kódolt vektor

Az összesített kimenet egy ~250-300 dimenziós laposított (flat) vektor.

Architektúra szerződés:
    - Bemenet: Nyers játékállapot szótár a játékmotorból
    - Kimenet: Dict[str, torch.Tensor] formátumú megfigyelési szótár
    - A hálózat beágyazó rétegei felelősek az egyesítésért

Hivatkozások:
    - Specifikáció: Állapottér és Akciótér Tervezése szekció
    - PettingZoo NLHE: https://pettingzoo.farama.org/environments/classic/texas_holdem_no_limit/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Konstansok
# =============================================================================

DECK_SIZE: int = 52
"""A francia kártyapakli mérete."""

NUM_SUITS: int = 4
"""Színek száma (pikk, kőr, káró, treff)."""

NUM_RANKS: int = 13
"""Értékek száma (2-A)."""

SUIT_MAP: dict[str, int] = {"S": 0, "H": 1, "D": 2, "C": 3}
"""Szín → index leképezés (Spade, Heart, Diamond, Club)."""

RANK_MAP: dict[str, int] = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}
"""Érték → index leképezés (2-A)."""


# =============================================================================
# Konfigurációs Adatosztály
# =============================================================================

@dataclass(frozen=True)
class ObservationConfig:
    """Az Observation Space összes konfigurálható paraméterét tárolja.

    Attributes:
        num_players: Az asztal játékosainak száma (2-9).
        max_betting_actions: Maximális licitkörök száma egy leosztásban.
        action_feature_dim: Akciótípusok + normalizált méretek dimenziója.
        initial_stack_bb: Induló zsetonkészlet Big Blind egységben.
        normalization_range: A normalizált értékek céltartománya [min, max].
    """

    num_players: int = 6
    max_betting_actions: int = 18
    action_feature_dim: int = 9
    initial_stack_bb: float = 200.0
    normalization_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        """Validálja a konfigurációs paramétereket inicializáláskor."""
        if not 2 <= self.num_players <= 9:
            raise ValueError(
                f"num_players értéke {self.num_players}, de 2 és 9 között kell lennie."
            )
        if self.max_betting_actions < 1:
            raise ValueError(
                f"max_betting_actions értéke {self.max_betting_actions}, de legalább 1 kell legyen."
            )
        logger.debug(
            "ObservationConfig inicializálva: num_players=%d, max_betting_actions=%d, "
            "initial_stack_bb=%.1f",
            self.num_players, self.max_betting_actions, self.initial_stack_bb,
        )


# =============================================================================
# Fő Observation Builder Osztály
# =============================================================================

class ObservationBuilder:
    """A nyers játékállapotot strukturált, normalizált tenzor-szótárrá alakítja.

    Ez az osztály a POMDP megfigyelési függvényt implementálja: a játékmotor
    nyers kimenetéből egy Dict[str, Tensor] formátumú bemenetet generál
    a neurális hálózat számára.

    A kimeneti szótár kulcsai:
        - ``hole_cards``: (52,) multi-hot bináris vektor a saját lapokhoz
        - ``community_cards``: (52,) multi-hot bináris vektor a közös lapokhoz
        - ``env_metrics``: (N,) normalizált környezeti metrikák
        - ``betting_history``: (max_actions, action_dim) licittörténet tenzor
        - ``position``: (num_players,) one-hot pozíció vektor
        - ``action_mask``: (9,) bináris akció érvényességi maszk

    Example:
        >>> config = ObservationConfig(num_players=6)
        >>> builder = ObservationBuilder(config)
        >>> raw_state = {"hand": ["AS", "KH"], "public_cards": ["TC", "JD", "QS"],
        ...              "pot": 150, "my_chips": 1800, "big_blind": 10, ...}
        >>> obs = builder.build(raw_state)
        >>> flat = builder.flatten(obs)
        >>> flat.shape  # ~250-300 dimenziós vektor
    """

    def __init__(self, config: ObservationConfig | None = None) -> None:
        """Inicializálja az ObservationBuilder-t a megadott konfigurációval.

        Args:
            config: Az observation space paraméterei. Ha None, az alapértelmezett
                    ObservationConfig kerül alkalmazásra.
        """
        self.config: ObservationConfig = config or ObservationConfig()
        self._norm_min: float = self.config.normalization_range[0]
        self._norm_max: float = self.config.normalization_range[1]

        logger.info(
            "ObservationBuilder inicializálva: %d játékos, %d max akció, "
            "normalizáció=[%.1f, %.1f]",
            self.config.num_players,
            self.config.max_betting_actions,
            self._norm_min,
            self._norm_max,
        )

    # =========================================================================
    # Publikus API
    # =========================================================================

    def build(self, raw_state: dict[str, Any]) -> dict[str, torch.Tensor]:
        """A nyers játékállapotból elkészíti a teljes megfigyelési szótárat.

        Args:
            raw_state: A játékmotor nyers kimeneti szótára. Elvárt kulcsok:
                - ``hand``: List[str] — saját lapok (pl. ["AS", "KH"])
                - ``public_cards``: List[str] — közös lapok (pl. ["TC", "JD", "QS"])
                - ``pot``: float — aktuális pot méret (abszolút chip)
                - ``my_chips``: float — saját zsetonállás (abszolút chip)
                - ``opponent_chips``: List[float] — ellenfelek zsetonállásai
                - ``big_blind``: float — nagyvak méret
                - ``amount_to_call``: float — megadandó tét (abszolút chip)
                - ``position``: int — saját pozíció index (0-tól)
                - ``betting_history``: List[dict] — licitlépések történelme
                - ``legal_actions``: List[int] — legális akció indexek

        Returns:
            Dict[str, torch.Tensor] formátumú megfigyelési szótár.

        Raises:
            KeyError: Ha a raw_state-ből hiányzik egy kötelező kulcs.
            ValueError: Ha a kártyaformátum érvénytelen.
        """
        if isinstance(raw_state, tuple):
            raw_state = raw_state[0]
        logger.debug("Observation építése nyers állapotból: %d kulcs", len(raw_state))

        try:
            observation: dict[str, torch.Tensor] = {
                "hole_cards": self._encode_cards(raw_state["hand"]),
                "community_cards": self._encode_cards(raw_state.get("public_cards", [])),
                "env_metrics": self._encode_env_metrics(raw_state),
                "betting_history": self._encode_betting_history(
                    raw_state.get("betting_history", [])
                ),
                "position": self._encode_position(raw_state.get("position", 0)),
                "action_mask": self._encode_action_mask(
                    raw_state.get("legal_actions", list(range(9)))
                ),
            }
        except KeyError as exc:
            logger.error("Hiányzó kulcs a nyers állapotból: %s", exc)
            raise

        # Dimenzió ellenőrzés (DEBUG szinten)
        if logger.isEnabledFor(logging.DEBUG):
            for key, tensor in observation.items():
                logger.debug(
                    "  Observation[%s]: shape=%s, dtype=%s, range=[%.4f, %.4f]",
                    key, tensor.shape, tensor.dtype,
                    tensor.min().item(), tensor.max().item(),
                )

        return observation

    def flatten(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """A szótárat egyetlen laposított (1D) vektorrá konvertálja.

        A flatten sorrendje determinisztikus és a hálózat beágyazó rétegének
        bemeneti dimenziójával konzisztens:
        hole_cards(52) + community_cards(52) + env_metrics(N) +
        betting_history(flat) + position(P)

        Args:
            observation: A build() metódus kimenete.

        Returns:
            Egydimenziós torch.Tensor (~250-300 elem).
        """
        components: list[torch.Tensor] = [
            observation["hole_cards"],
            observation["community_cards"],
            observation["env_metrics"],
            observation["betting_history"].flatten(),
            observation["position"],
        ]

        flat_vector: torch.Tensor = torch.cat(components, dim=0)

        logger.debug(
            "Laposított megfigyelés: dim=%d (hole=52, community=52, metrics=%d, "
            "history=%d, position=%d)",
            flat_vector.shape[0],
            observation["env_metrics"].shape[0],
            observation["betting_history"].numel(),
            observation["position"].shape[0],
        )

        return flat_vector

    def get_observation_dim(self) -> int:
        """Kiszámítja a laposított megfigyelési vektor teljes dimenzióját.

        Returns:
            A vektor elemeinek összesített száma (int).
        """
        card_dim: int = DECK_SIZE * 2  # hole + community
        # env_metrics: pot, my_chips, amount_to_call, min_raise, + opponent stacks
        metrics_dim: int = 4 + (self.config.num_players - 1)
        history_dim: int = (
            self.config.max_betting_actions * self.config.action_feature_dim
        )
        position_dim: int = self.config.num_players

        total: int = card_dim + metrics_dim + history_dim + position_dim

        logger.debug(
            "Observation dimenzió: cards=%d, metrics=%d, history=%d, position=%d → total=%d",
            card_dim, metrics_dim, history_dim, position_dim, total,
        )
        return total

    # =========================================================================
    # Privát Kódoló Metódusok
    # =========================================================================

    def _encode_cards(self, cards: list[str]) -> torch.Tensor:
        """Kártyalistát 52-dimenziós multi-hot bináris vektorrá kódol.

        Minden kártya egy egyedi indexet kap a [0, 51] tartományban:
            index = rank_index * NUM_SUITS + suit_index

        A kimenet egy 52-elemű float32 vektor, ahol az aktív kártyák
        pozícióján 1.0, máshol 0.0 áll.

        Args:
            cards: Kártyajelölések listája (pl. ["AS", "KH", "TC"]).
                   Üres lista esetén csupa nulla vektor.

        Returns:
            (52,) alakú torch.Tensor (float32).

        Raises:
            ValueError: Ha egy kártya formátuma érvénytelen.
        """
        encoding: torch.Tensor = torch.zeros(DECK_SIZE, dtype=torch.float32)

        for card_str in cards:
            card_str = card_str.strip().upper()
            if len(card_str) != 2:
                raise ValueError(
                    f"Érvénytelen kártyaformátum: '{card_str}'. "
                    f"Elvárt formátum: 'RS' (pl. 'AS' = Ász Pikk)."
                )

            rank_char: str = card_str[0]
            suit_char: str = card_str[1]

            if rank_char not in RANK_MAP:
                raise ValueError(
                    f"Ismeretlen kártyaérték: '{rank_char}' a(z) '{card_str}' kártyában. "
                    f"Érvényes értékek: {list(RANK_MAP.keys())}"
                )
            if suit_char not in SUIT_MAP:
                raise ValueError(
                    f"Ismeretlen kártyaszín: '{suit_char}' a(z) '{card_str}' kártyában. "
                    f"Érvényes színek: {list(SUIT_MAP.keys())}"
                )

            rank_idx: int = RANK_MAP[rank_char]
            suit_idx: int = SUIT_MAP[suit_char]
            card_index: int = rank_idx * NUM_SUITS + suit_idx

            encoding[card_index] = 1.0

        logger.debug(
            "Kártyakódolás: %d kártya → %d aktív pozíció a 52-dim vektorban",
            len(cards), int(encoding.sum().item()),
        )

        return encoding

    def _encode_env_metrics(self, raw_state: dict[str, Any]) -> torch.Tensor:
        """A környezeti metrikákat normalizált float vektorrá kódolja.

        Normalizáció: Minden monetáris érték a nagyvak (Big Blind) összegéhez
        viszonyítva, majd a [0, 1] tartományba szorítva.

        A metrikák sorrendje:
            [pot_norm, my_chips_norm, amount_to_call_norm, min_raise_norm,
             opp_1_chips_norm, opp_2_chips_norm, ...]

        Args:
            raw_state: A nyers játékállapot szótár.

        Returns:
            (4 + num_opponents,) alakú normalizált torch.Tensor.
        """
        big_blind: float = float(raw_state.get("big_blind", 2.0))
        if big_blind <= 0:
            logger.warning(
                "A big_blind értéke %.2f, ami nem pozitív. Fallback: 2.0", big_blind
            )
            big_blind = 2.0

        initial_stack: float = self.config.initial_stack_bb * big_blind

        # Normalizáló segédfüggvény: chip → [0, 1] tartomány
        def _normalize_chips(value: float) -> float:
            """Monetáris értéket normalizál az induló stack-hez viszonyítva."""
            normalized: float = value / initial_stack if initial_stack > 0 else 0.0
            return float(np.clip(normalized, self._norm_min, self._norm_max))

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

        # Ellenfelek zsetonállásainak normalizálása
        opponent_chips: list[float] = raw_state.get("opponent_chips", [])
        for opp_stack in opponent_chips[: self.config.num_players - 1]:
            metrics.append(_normalize_chips(float(opp_stack)))

        # Padding: ha kevesebb ellenfél van, mint a konfiguráció elvárja
        expected_opponents: int = self.config.num_players - 1
        while len(metrics) < 4 + expected_opponents:
            metrics.append(0.0)

        metrics_tensor: torch.Tensor = torch.tensor(metrics, dtype=torch.float32)

        logger.debug(
            "Környezeti metrikák kódolva: pot=%.3f, stack=%.3f, call=%.3f, "
            "min_raise=%.3f, %d ellenfél stack",
            metrics[0], metrics[1], metrics[2], metrics[3],
            len(opponent_chips),
        )

        return metrics_tensor

    def _encode_betting_history(
        self, history: list[dict[str, Any]]
    ) -> torch.Tensor:
        """A licittörténetet rögzített méretű tenzorrá kódolja.

        Minden licitlépés egy sorként jelenik meg a kimeneti mátrixban:
            [action_one_hot(9-dim)] ahol az akció indexe alapján
            a megfelelő pozíció 1.0.

        A tenzor zero-paddinget alkalmaz a maximális méretig.

        Args:
            history: Licitlépések listája. Minden elem egy szótár:
                     {"action": int, "amount": float, "player": int}

        Returns:
            (max_betting_actions, action_feature_dim) alakú torch.Tensor.
        """
        max_actions: int = self.config.max_betting_actions
        action_dim: int = self.config.action_feature_dim

        history_tensor: torch.Tensor = torch.zeros(
            (max_actions, action_dim), dtype=torch.float32
        )

        for step_idx, step in enumerate(history[:max_actions]):
            action_idx: int = int(step.get("action", 0))
            if 0 <= action_idx < action_dim:
                history_tensor[step_idx, action_idx] = 1.0

            # Opcionális: normalizált tét méret hozzáadása
            # (ha az action_dim > num_actions, az extra dimenziók használhatók)

        logger.debug(
            "Licittörténet kódolva: %d/%d lépés feltöltve (zero-padding: %d sor)",
            min(len(history), max_actions),
            max_actions,
            max(0, max_actions - len(history)),
        )

        return history_tensor

    def _encode_position(self, position_index: int) -> torch.Tensor:
        """A játékos pozícióját one-hot vektorrá kódolja.

        A pozíciók indexelése (6-Max példa):
            0=SB, 1=BB, 2=UTG, 3=HJ, 4=CO, 5=BTN

        Args:
            position_index: A játékos pozíciójának indexe (0-tól).

        Returns:
            (num_players,) alakú one-hot torch.Tensor.
        """
        num_positions: int = self.config.num_players
        position_vector: torch.Tensor = torch.zeros(num_positions, dtype=torch.float32)

        if 0 <= position_index < num_positions:
            position_vector[position_index] = 1.0
        else:
            logger.warning(
                "Érvénytelen pozíció index: %d (max: %d). Nulla vektor visszaadva.",
                position_index, num_positions - 1,
            )

        logger.debug("Pozíció kódolva: index=%d/%d", position_index, num_positions)
        return position_vector

    def _encode_action_mask(self, legal_actions: list[int]) -> torch.Tensor:
        """Bináris akció érvényességi maszkot generál.

        Az érvényes akciók indexeinél 1.0, az érvénytelen (illegális)
        akcióknál 0.0 áll. Ezt a maszkot a hálózat utolsó rétegén,
        a Softmax előtt alkalmazzuk logit maszkolásra.

        Args:
            legal_actions: Az érvényes akciók indexeinek listája (0-8).

        Returns:
            (9,) alakú bináris torch.Tensor.
        """
        num_actions: int = 9
        mask: torch.Tensor = torch.zeros(num_actions, dtype=torch.float32)

        for action_idx in legal_actions:
            if 0 <= action_idx < num_actions:
                mask[action_idx] = 1.0
            else:
                logger.warning(
                    "Illegális akció index figyelmen kívül hagyva: %d (tartomány: 0-%d)",
                    action_idx, num_actions - 1,
                )

        active_count: int = int(mask.sum().item())
        if active_count == 0:
            logger.error(
                "KRITIKUS: Üres akció maszk! Nincs érvényes akció. "
                "Fold (index 0) engedélyezése biztonsági fallback-ként."
            )
            mask[0] = 1.0  # Fold mindig legális

        logger.debug(
            "Akció maszk generálva: %d/%d akció érvényes",
            active_count, num_actions,
        )
        return mask

    # =========================================================================
    # Segédmetódusok
    # =========================================================================

    @staticmethod
    def card_str_to_index(card_str: str) -> int:
        """Egyetlen kártyajelölést numerikus indexé alakít.

        Args:
            card_str: Kétkarakteres kártyajelölés (pl. "AS" = Ász Pikk).

        Returns:
            A kártya indexe a [0, 51] tartományban.

        Raises:
            ValueError: Ha a formátum érvénytelen.
        """
        card_str = card_str.strip().upper()
        if len(card_str) != 2:
            raise ValueError(f"Érvénytelen kártyaformátum: '{card_str}'")
        rank_idx: int = RANK_MAP[card_str[0]]
        suit_idx: int = SUIT_MAP[card_str[1]]
        return rank_idx * NUM_SUITS + suit_idx

    @staticmethod
    def index_to_card_str(index: int) -> str:
        """Numerikus kártyaindexet kártyajelöléssé alakít.

        Args:
            index: A kártya indexe a [0, 51] tartományban.

        Returns:
            Kétkarakteres kártyajelölés (pl. "AS").

        Raises:
            ValueError: Ha az index kívül esik a [0, 51] tartományon.
        """
        if not 0 <= index < DECK_SIZE:
            raise ValueError(f"Kártyaindex {index} kívül esik a [0, 51] tartományon.")
        rank_idx: int = index // NUM_SUITS
        suit_idx: int = index % NUM_SUITS
        rank_chars: str = "23456789TJQKA"
        suit_chars: str = "SHDC"
        return rank_chars[rank_idx] + suit_chars[suit_idx]
