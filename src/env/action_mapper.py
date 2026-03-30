"""
Diszkretizált Akciótér Kezelő (action_mapper.py).

A No-Limit Texas Hold'em definíció szerint folytonos akciótérrel rendelkezik,
mivel bármilyen összeg emelhető a minimális emelés és az all-in között.
A gyakorlatban és a csúcsmodell AI-k (Libratus, Pluribus) esetében a folytonos
akciótér instabil, nehezen optimalizálható stratégiákat eredményez.

Ez a modul egy 12-dimenziós diszkretizált akcióteret implementál, amely:
    1. A hálózat Softmax kimeneti indexét szemantikai póker akcióvá fordítja
    2. Kiszámítja a pot-relatív tétméreteket chip értékekben
    3. Érvényesíti a legális akciók maszkját a Softmax logitok szintjén
    4. Kezeli az edge case-eket (nem elegendő stack, minimum raise szabály)
    5. Különösen kezeli a CHECK és CALL akciókat (Deep CFR feltételezés)

Akció Index Tábla (12 diszkrét akció):
    0 = Fold (Dobás)
    1 = Check (Passz — csak ha nincs bet)
    2 = Call (Megadás — szükséges ha van bet)
    3 = Min-Raise (Legkisebb legális emelés)
    4 = Raise 0.25x Pot (25% pot — early position sizing)
    5 = Raise 0.33x Pot (GTO range bet / block bet)
    6 = Raise 0.5x Pot
    7 = Raise 0.75x Pot
    8 = Raise 1.0x Pot
    9 = Raise 1.5x Pot (Overbet)
    10 = Raise 2.0x Pot (Deep Overbet)
    11 = All-in (Teljes stack betolás)

Hivatkozások:
    - Specifikáció: Akciótér szekció (9 diszkrét akció)
    - Action Masking: torch.where + torch.finfo(dtype).min (AMP-safe)
    - Libratus/Pluribus bet sizing buckets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Akció Enumeráció
# =============================================================================

class PokerAction(IntEnum):
    """A diszkretizált akciótér 12 lehetséges lépésének enumerációja.

    Minden akció egy egyértelmű indexet kap, amelyet a Softmax kimeneti
    réteg valószínűségi eloszlásként ad vissza.

    CHECK és CALL szeparálása kritikus Deep CFR-hez:
    - CHECK (1): passzív akció, ha amount_to_call = 0
    - CALL (2): passzív akció, ha amount_to_call > 0

    Ez lehetővé teszi a hálózatnak, hogy megtanuljon check-raise dinamikát
    és pontos pozíciós alkalmazkodást.

    Action space (12 buckets — checkpoint-breaking change):
        0  Fold
        1  Check  ← NEW: separated from CALL for check-raise learning
        2  Call   ← NEW: separated from CHECK
        3  Min-Raise
        4  Raise 0.25x Pot
        5  Raise 0.33x Pot  (GTO range bet / block bet)
        6  Raise 0.50x Pot
        7  Raise 0.75x Pot
        8  Raise 1.0x Pot
        9  Raise 1.5x Pot   (overbet)
        10 Raise 2.0x Pot   (deep overbet)
        11 All-in
    """

    FOLD = 0
    CHECK = 1
    CALL = 2
    MIN_RAISE = 3
    RAISE_QUARTER_POT = 4              # 25% pot early position sizing
    RAISE_THIRD_POT = 5                # 33% pot block/range bet
    RAISE_HALF_POT = 6
    RAISE_THREE_QUARTER_POT = 7
    RAISE_FULL_POT = 8
    RAISE_1_5X_POT = 9
    RAISE_2X_POT = 10
    ALL_IN = 11


# Pot-relatív szorzók az emelési akciókhoz
_RAISE_MULTIPLIERS: dict[PokerAction, float] = {
    PokerAction.RAISE_QUARTER_POT: 0.25,           # NEW — 25% pot early position sizing
    PokerAction.RAISE_THIRD_POT: 0.33,             # block bet / range bet
    PokerAction.RAISE_HALF_POT: 0.50,
    PokerAction.RAISE_THREE_QUARTER_POT: 0.75,
    PokerAction.RAISE_FULL_POT: 1.00,
    PokerAction.RAISE_1_5X_POT: 1.50,
    PokerAction.RAISE_2X_POT: 2.00,
}

NUM_ACTIONS: int = 12
"""Az akciótér teljes dimenziója (12 — CHECK és CALL szeparálva Deep CFR-hez)."""

ILLEGAL_ACTION_LOGIT: float = -1.0e8
"""Az illegális akciók logitjaihoz hozzáadott extrém negatív szám.

.. deprecated::
    Használd a :func:`get_safe_mask_value` függvényt, amely dtype-aware
    és biztonságos float16 (AMP) környezetben is.
"""


def get_safe_mask_value(dtype: torch.dtype = torch.float32) -> float:
    """Visszaadja a dtype-specifikus biztonságos maszkolási értéket.

    A ``torch.finfo(dtype).min`` értéket használja, amely garantáltan
    a legkisebb véges szám az adott típusban, elkerülve a NaN propagációt
    Automatic Mixed Precision (AMP) float16 környezetben.

    +-----------+------------------+-----------------------------------+
    | dtype     | finfo.min        | Megjegyzés                        |
    +===========+==================+===================================+
    | float32   | ~-3.4e38         | Bőven biztonságos                 |
    +-----------+------------------+-----------------------------------+
    | float16   | ~-65504          | -1e8 itt NaN-t okozna!            |
    +-----------+------------------+-----------------------------------+
    | bfloat16  | ~-3.39e38        | Biztonságos                       |
    +-----------+------------------+-----------------------------------+

    Args:
        dtype: A logit tenzor adattípusa.

    Returns:
        A biztonságos minimális véges érték az adott dtype-hoz.
    """
    return torch.finfo(dtype).min


# =============================================================================
# Adatstruktúrák
# =============================================================================

# =============================================================================
# Adatstruktúrák
# =============================================================================

@dataclass(frozen=True)
class BetSizingConfig:
    """Street-specific bet sizing configuration.
    
    Defines fractional pot multipliers for each street (preflop, flop, turn, river).
    This allows the strategy to adapt bet sizing based on the current game state.
    
    Attributes:
        preflop: List of bet size multipliers for preflop (e.g., [0.33, 0.5, 0.75, 1.0]).
        flop: List of bet size multipliers for flop.
        turn: List of bet size multipliers for turn.
        river: List of bet size multipliers for river.
    """
    preflop: list[float] = None
    flop: list[float] = None
    turn: list[float] = None
    river: list[float] = None
    
    def __post_init__(self):
        """Set default configurations if not provided."""
        if self.preflop is None:
            object.__setattr__(self, 'preflop', [0.33, 0.5, 0.75, 1.0])
        if self.flop is None:
            object.__setattr__(self, 'flop', [0.33, 0.5, 0.75, 1.0])
        if self.turn is None:
            object.__setattr__(self, 'turn', [0.5, 0.75, 1.0, 1.5])
        if self.river is None:
            object.__setattr__(self, 'river', [0.5, 0.75, 1.0, 1.5])
    
    def get_multipliers(self, street: int) -> list[float]:
        """Get bet size multipliers for a specific street.
        
        Args:
            street: Street index (0=preflop, 1=flop, 2=turn, 3=river).
        
        Returns:
            List of fractional pot multipliers.
        """
        streets = [self.preflop, self.flop, self.turn, self.river]
        if street < 0 or street >= len(streets):
            return self.preflop  # Default to preflop
        return streets[street]


class ResolvedAction(NamedTuple):
    """Az akció-feloldás eredménye: a szemantikai akció és a pontos chip összeg.

    Attributes:
        action: A végrehajtandó akció típusa.
        amount: A tétméret abszolút chip értékben. Fold és Check esetén 0.
        description: Emberi olvashatóságú leírás a lépésről.
    """

    action: PokerAction
    amount: float
    description: str


@dataclass(frozen=True)
class GameContext:
    """Az aktuális játékszituáció kontextusa az akció-feloldáshoz.

    Attributes:
        pot_size: Az aktuális pot mérete (abszolút chip).
        my_stack: A saját zsetonállás (abszolút chip).
        amount_to_call: A megadandó tét összege (chip).
        min_raise_amount: A legkisebb legális emelési méret (chip).
        big_blind: A nagyvak mérete (chip).
        street: Az aktuális street indexe (0=preflop, 1=flop, 2=turn, 3=river).
    """

    pot_size: float
    my_stack: float
    amount_to_call: float
    min_raise_amount: float
    big_blind: float
    street: int = 0

    def __post_init__(self) -> None:
        """Validálja a kontextus értékeit."""
        if self.pot_size < 0:
            raise ValueError(f"pot_size nem lehet negatív: {self.pot_size}")
        if self.my_stack < 0:
            raise ValueError(f"my_stack nem lehet negatív: {self.my_stack}")
        if self.big_blind <= 0:
            raise ValueError(f"big_blind pozitívnak kell lennie: {self.big_blind}")
        if self.street < 0 or self.street > 3:
            raise ValueError(f"street 0-3 között kell lennie: {self.street}")


# =============================================================================
# Fő ActionMapper Osztály
# =============================================================================

class ActionMapper:
    """A hálózat diszkrét akció-indexeit konkrét póker lépésekre fordítja.

    Ez az osztály felelős a hálózat kimeneti rétegén a Softmax által
    generált valószínűségi eloszlás indexeinek szemantikai és matematikai
    feloldásáért, valamint az illegális akciók szűréséért.

    Az Action Masking mechanizmus biztosítja, hogy a hálózat soha ne
    válasszon érvénytelen lépést, anélkül hogy a backward pass
    összeomlana.

    Example:
        >>> mapper = ActionMapper()
        >>> ctx = GameContext(pot_size=100, my_stack=500,
        ...                  amount_to_call=50, min_raise_amount=100, big_blind=10)
        >>> legal = mapper.get_legal_actions(ctx)
        >>> masked_logits = mapper.apply_action_mask(logits, legal)
        >>> action_idx = torch.argmax(masked_logits).item()
        >>> resolved = mapper.resolve_action(PokerAction(action_idx), ctx)
    """

    def __init__(self) -> None:
        """Inicializálja az ActionMapper-t."""
        self._action_names: dict[PokerAction, str] = {
            PokerAction.FOLD: "Fold",
            PokerAction.CHECK: "Check",
            PokerAction.CALL: "Call",
            PokerAction.MIN_RAISE: "Min-Raise",
            PokerAction.RAISE_QUARTER_POT: "Raise 0.25x Pot",
            PokerAction.RAISE_THIRD_POT: "Raise 0.33x Pot",
            PokerAction.RAISE_HALF_POT: "Raise 0.5x Pot",
            PokerAction.RAISE_THREE_QUARTER_POT: "Raise 0.75x Pot",
            PokerAction.RAISE_FULL_POT: "Raise 1.0x Pot",
            PokerAction.RAISE_1_5X_POT: "Raise 1.5x Pot",
            PokerAction.RAISE_2X_POT: "Raise 2.0x Pot",
            PokerAction.ALL_IN: "All-in",
        }
        self.bet_sizing_config: BetSizingConfig = BetSizingConfig()
        logger.info(
            "ActionMapper inicializálva: %d diszkrét akció, "
            "illegális logit maszk=%.0e, "
            "street-specific bet sizing enabled",
            NUM_ACTIONS, ILLEGAL_ACTION_LOGIT,
        )

    # =========================================================================
    # Akció Feloldás (Resolution)
    # =========================================================================

    def resolve_action(
        self, action: PokerAction, context: GameContext
    ) -> ResolvedAction:
        """Egy diszkrét akció-indexet konkrét chip értékű póker akcióvá old fel.

        Az ezutáni metódus figyelembe veszi a játékszituáció kontextusát 
        (pot méret, stack méret, minimális emelés, street) és kiszámítja 
        a pontos tétméretet a street-specifikus bet sizing szabályok alapján.

        Ha az emelés összege meghaladja a játékos stackjét, automatikusan
        All-in-né konvertálódik (stack capping).
        
        Args:
            action: A hálózat által választott akció (PokerAction enum).
            context: Az aktuális játékszituáció (street információval).

        Returns:
            ResolvedAction nevesített tuple a végrehajtási részletekkel.
        """
        logger.debug(
            "Akció feloldása: %s (street=%d) | pot=%.0f, stack=%.0f, call=%.0f",
            action.name, context.street, context.pot_size, context.my_stack, context.amount_to_call,
        )

        if action == PokerAction.FOLD:
            return ResolvedAction(
                action=PokerAction.FOLD,
                amount=0.0,
                description="Fold — A játékos feladja a leosztást.",
            )

        if action == PokerAction.CHECK:
            if context.amount_to_call == 0:
                return ResolvedAction(
                    action=PokerAction.CHECK,
                    amount=0.0,
                    description="Check — A játékos passzív lépést választ.",
                )
            else:
                logger.debug(
                    "CHECK kérés amount_to_call > 0 mellett → CALL-ra visszaváltás"
                )
                call_amount: float = min(context.amount_to_call, context.my_stack)
                return ResolvedAction(
                    action=PokerAction.CALL,
                    amount=call_amount,
                    description=f"Call — {call_amount:.0f} chip",
                )

        if action == PokerAction.CALL:
            call_amount: float = min(context.amount_to_call, context.my_stack)
            if call_amount == 0:
                return ResolvedAction(
                    action=PokerAction.CHECK,
                    amount=0.0,
                    description="Check — A játékos passzív lépést választ.",
                )
            return ResolvedAction(
                action=PokerAction.CALL,
                amount=call_amount,
                description=f"Call — {call_amount:.0f} chip",
            )

        if action == PokerAction.ALL_IN:
            return ResolvedAction(
                action=PokerAction.ALL_IN,
                amount=context.my_stack,
                description=f"All-in — {context.my_stack:.0f} chip (teljes stack)",
            )

        # Emelési akciók: street-specifikus bet sizing verwendet
        raise_amount = self._calculate_raise_amount(action, context)

        # Minimum raise floor
        raise_amount = max(raise_amount, context.min_raise_amount)

        # Stack capping
        if raise_amount >= context.my_stack:
            logger.debug(
                "Stack capping: kívánt emelés=%.0f >= stack=%.0f → All-in",
                raise_amount, context.my_stack,
            )
            return ResolvedAction(
                action=PokerAction.ALL_IN,
                amount=context.my_stack,
                description=(
                    f"All-in (stack cap) — {context.my_stack:.0f} chip "
                    f"(kívánt: {self._action_names.get(action, '?')})"
                ),
            )

        action_name: str = self._action_names.get(action, f"Action-{action.value}")
        return ResolvedAction(
            action=action,
            amount=raise_amount,
            description=f"{action_name} — {raise_amount:.0f} chip",
        )
    
    def _calculate_raise_amount(self, action: PokerAction, context: GameContext) -> float:
        """Calculate raise amount using street-specific bet sizing.
        
        Args:
            action: The raise action type (MIN_RAISE or one of the RAISE_*_POT actions).
            context: The current game context with street information.
        
        Returns:
            The calculated raise amount in chips.
        """
        if action == PokerAction.MIN_RAISE:
            return context.min_raise_amount
        
        # Map action to street-specific multiplier
        street_multipliers = self.bet_sizing_config.get_multipliers(context.street)
        multiplier = 0.5  # Default fallback
        
        # Map action index to street-specific multiplier position
        if action == PokerAction.RAISE_QUARTER_POT:
            multiplier = street_multipliers[0] if len(street_multipliers) > 0 else 0.33
        elif action == PokerAction.RAISE_THIRD_POT:
            multiplier = street_multipliers[0] if len(street_multipliers) > 0 else 0.33
        elif action == PokerAction.RAISE_HALF_POT:
            idx = min(1, len(street_multipliers) - 1)
            multiplier = street_multipliers[idx] if len(street_multipliers) > idx else 0.5
        elif action == PokerAction.RAISE_THREE_QUARTER_POT:
            idx = min(2, len(street_multipliers) - 1)
            multiplier = street_multipliers[idx] if len(street_multipliers) > idx else 0.75
        elif action == PokerAction.RAISE_FULL_POT:
            idx = min(len(street_multipliers) - 1, 2)
            multiplier = street_multipliers[idx] if len(street_multipliers) > idx else 1.0
        elif action == PokerAction.RAISE_1_5X_POT:
            idx = min(len(street_multipliers) - 1, 3)
            multiplier = street_multipliers[idx] if len(street_multipliers) > idx else 1.5
        elif action == PokerAction.RAISE_2X_POT:
            multiplier = street_multipliers[-1] if street_multipliers else 1.5
        
        # Calculate total raise amount
        raise_amount = context.amount_to_call + multiplier * (context.pot_size + context.amount_to_call)
        return raise_amount

    # =========================================================================
    # Legális Akciók és Maszkolás
    # =========================================================================

    def get_legal_actions(self, context: GameContext) -> list[PokerAction]:
        """Meghatározza a jelenlegi játékszituációban legális akciók listáját.

        Szabályok:
            - Fold: Mindig legális
            - Check: Legális ha amount_to_call == 0
            - Call: Legális ha amount_to_call > 0
            - Emelések: Csak ha a stack elegendő a minimális emeléshez
            - All-in: Mindig legális, ha van chip a stackben

        Args:
            context: Az aktuális játékszituáció.

        Returns:
            A legális PokerAction értékek listája.
        """
        legal: list[PokerAction] = []

        # Fold mindig legális
        legal.append(PokerAction.FOLD)
        
        # CHECK vagy CALL: kontextustól függő
        if context.amount_to_call == 0:
            legal.append(PokerAction.CHECK)
        else:
            legal.append(PokerAction.CALL)

        # Emelések: csak ha van elég chip a minimális emeléshez
        remaining_after_call: float = context.my_stack - context.amount_to_call
        if remaining_after_call > 0:
            # Min-Raise
            if remaining_after_call >= context.min_raise_amount:
                legal.append(PokerAction.MIN_RAISE)

            # Pot-relatív emelések
            for action, multiplier in _RAISE_MULTIPLIERS.items():
                target_raise: float = context.pot_size * multiplier
                if remaining_after_call >= target_raise:
                    legal.append(action)

            # All-in: mindig legális ha van bármennyi chip
            if context.my_stack > 0:
                legal.append(PokerAction.ALL_IN)

        logger.debug(
            "Legális akciók: %d/%d — %s",
            len(legal), NUM_ACTIONS,
            [a.name for a in legal],
        )
        return legal

    def get_action_mask_tensor(self, context: GameContext) -> torch.Tensor:
        """Bináris akció maszkot generál torch.Tensor formátumban.

        Az érvényes akció pozíciókon 1.0, az érvénytelen pozíciókon 0.0.

        Args:
            context: Az aktuális játékszituáció.

        Returns:
            (10,) alakú bináris torch.Tensor (float32).
        """
        legal_actions: list[PokerAction] = self.get_legal_actions(context)
        mask: torch.Tensor = torch.zeros(NUM_ACTIONS, dtype=torch.float32)

        for action in legal_actions:
            mask[action.value] = 1.0

        return mask

    @staticmethod
    def apply_action_mask(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """A Softmax ELŐTT alkalmazza az akció maszkot a logit vektorra.

        AMP-SAFE implementáció: ``torch.where`` + ``torch.finfo(dtype).min``
        használatával, amely float16/bfloat16 környezetben is biztonságos.
        A korábbi additív maszkolás (-1e8) float16-ban NaN-t okozna,
        mivel a float16 min ~ -65504.

        Args:
            logits: A hálózat nyers kimeneti logitjai (10,) vagy (batch, 10).
            action_mask: Bináris maszk (1.0 = legális, 0.0 = illegális).
                         Azonos alakú a logits-szal.

        Returns:
            A maszkolt logit vektor, ugyanolyan alakban mint a bemenet.

        Raises:
            ValueError: Ha a logits és action_mask alakja nem egyezik.
        """
        if logits.shape != action_mask.shape:
            raise ValueError(
                f"A logits ({logits.shape}) és action_mask ({action_mask.shape}) "
                f"alakja nem egyezik."
            )

        # AMP-safe: dtype-specifikus minimális véges érték
        mask_value: float = get_safe_mask_value(logits.dtype)
        masked_logits: torch.Tensor = torch.where(
            action_mask.bool(), logits, torch.tensor(mask_value, dtype=logits.dtype)
        )

        if logger.isEnabledFor(logging.DEBUG):
            num_legal: int = int(action_mask.sum().item()) if action_mask.dim() == 1 else -1
            logger.debug(
                "Akció maszkolás alkalmazva: %d legális akció, "
                "logit tartomány: [%.2f, %.2f] → maszkolt: [%.2f, %.2f], "
                "dtype=%s, mask_value=%.2e",
                num_legal,
                logits.min().item(), logits.max().item(),
                masked_logits[action_mask.bool()].min().item()
                if action_mask.any() else float("nan"),
                masked_logits[action_mask.bool()].max().item()
                if action_mask.any() else float("nan"),
                logits.dtype, mask_value,
            )

        return masked_logits

    # =========================================================================
    # Segédmetódusok
    # =========================================================================

    def action_index_to_name(self, index: int) -> str:
        """Akció indexet emberi olvashatóságú névvé alakít.

        Args:
            index: Akció index (0-9).

        Returns:
            Az akció neve szövegesen.
        """
        try:
            action = PokerAction(index)
            return self._action_names.get(action, f"Unknown-{index}")
        except ValueError:
            return f"Invalid-{index}"

    @staticmethod
    def sample_action(
        masked_logits: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[int, float]:
        """Mintavételez egy akciót a maszkolt logitokból.

        Args:
            masked_logits: A maszkolt logit vektor (9,).
            deterministic: Ha True, a legmagasabb valószínűségű akciót választja
                          (greedy). Ha False, sztochasztikus mintavétel.

        Returns:
            Tuple: (kiválasztott_akció_index, az akció log-valószínűsége).
        """
        probs: torch.Tensor = torch.softmax(masked_logits, dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)

        if deterministic:
            action_idx: int = int(torch.argmax(probs, dim=-1).item())
        else:
            action_tensor: torch.Tensor = distribution.sample()
            action_idx = int(action_tensor.item())

        log_prob: float = float(distribution.log_prob(torch.tensor(action_idx)).item())

        logger.debug(
            "Akció mintavételezés: index=%d (%s), log_prob=%.4f, deterministic=%s",
            action_idx,
            PokerAction(action_idx).name if 0 <= action_idx < NUM_ACTIONS else "?",
            log_prob,
            deterministic,
        )

        return action_idx, log_prob
