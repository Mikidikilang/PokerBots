"""
Dinamikus Jutalom Formazo (reward_shaper.py).

A nyers chip-EV jutalmat futasidejuleg modositja a patologias
viselkedesek korrigalasara. Ket fo beavatkozasi mechanizmus:

    1. All-in Spam Buntetes:
       R_mod = R_base - lambda * bluff_intensity * I(lost_showdown)
       A sikertelen bloffok es indokolatlan All-in lepesek buntetese.

    2. Passzivitas Bonus:
       R_mod = R_base + bonus * I(preflop_raise)
       Enyhe pozitiv jutalom a pre-flop emelesekert.

A lambda es bonus parameterek HOT-RELOADABLE: az Orchestrator
futasidejuleg modositja oket a config.yaml-en keresztul.

Hivatkozasok:
    - Specifikacio: reward_shaper.py — dinamikus EV modositasok
    - Orchestrator Data: Reward Shaping mechanika
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RewardShapingConfig:
    """A jutalom formazas konfiguracioja.

    Attributes:
        bluff_penalty_lambda: Bloff bunteto szorzo (0.0=inaktiv). HOT-RELOADABLE.
        bluff_intensity_threshold: Bloff intenzitasi kuszob (tet/pot arany).
        preflop_aggression_bonus: Bonus pre-flop emelesekert. HOT-RELOADABLE.
        entropy_boost_factor: Entropia koefficienshez szorzo stagnacio eseten.
        stagnation_window: Stagnacio detektalasi ablak merete (iteracio).
        stagnation_threshold: Minimalis jutalom-valtozasi kuszob.
    """

    bluff_penalty_lambda: float = 0.0
    bluff_intensity_threshold: float = 0.7
    preflop_aggression_bonus: float = 0.0
    entropy_boost_factor: float = 2.0
    stagnation_window: int = 50
    stagnation_threshold: float = 0.001

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> RewardShapingConfig:
        """YAML config szotarbol peldanyosit.

        Args:
            cfg: Teljes YAML konfiguracio.

        Returns:
            RewardShapingConfig peldany.
        """
        rs = cfg.get("reward_shaping", {})
        return cls(
            bluff_penalty_lambda=rs.get("bluff_penalty_lambda", 0.0),
            bluff_intensity_threshold=rs.get("bluff_intensity_threshold", 0.7),
            preflop_aggression_bonus=rs.get("preflop_aggression_bonus", 0.0),
            entropy_boost_factor=rs.get("entropy_boost_factor", 2.0),
            stagnation_window=rs.get("stagnation_window", 50),
            stagnation_threshold=rs.get("stagnation_threshold", 0.001),
        )


class RewardShaper:
    """Futasidejuleg modositja a kornyezet altal adott jutalmat.

    A RewardShaper a nyers chip-EV jutalomhoz hozzaadja vagy kivonja
    a patologia-korrigalo tagokat. A parameterek (lambda, bonus) az
    Orchestrator altal dinamikusan modosithatoak.

    Example:
        >>> shaper = RewardShaper(RewardShapingConfig())
        >>> modified = shaper.shape_reward(
        ...     base_reward=5.0,
        ...     action_index=8, bet_amount=200, pot_size=100,
        ...     hand_strength=0.3, lost_showdown=True,
        ...     is_preflop_raise=False, street=0,
        ... )

    Attributes:
        config: A jutalom formazas konfiguracioja.
    """

    def __init__(self, config: RewardShapingConfig | None = None) -> None:
        """Inicializalja a RewardShaper-t.

        Args:
            config: Konfiguracio. Alapertelmezett ha None.
        """
        self.config: RewardShapingConfig = config or RewardShapingConfig()
        self._total_penalties: float = 0.0
        self._total_bonuses: float = 0.0
        self._total_calls: int = 0

        logger.info(
            "RewardShaper inicializalva: lambda=%.4f, bonus=%.4f, "
            "threshold=%.2f, entropy_boost=%.1fx",
            self.config.bluff_penalty_lambda,
            self.config.preflop_aggression_bonus,
            self.config.bluff_intensity_threshold,
            self.config.entropy_boost_factor,
        )

    # =========================================================================
    # Fo Jutalom Formatas
    # =========================================================================

    def shape_reward(
        self,
        base_reward: float,
        action_index: int,
        bet_amount: float = 0.0,
        pot_size: float = 1.0,
        hand_strength: float = 0.5,
        lost_showdown: bool = False,
        is_preflop_raise: bool = False,
        street: int = 0,
    ) -> float:
        """A nyers jutalmat modositja a patologia-korrigalo tagokkal.

        A formula:
            R_mod = R_base
                  - lambda * bluff_intensity * I(lost_showdown)
                  + bonus * I(preflop_raise)

        Args:
            base_reward: A kornyezet altal adott nyers jutalom (chip EV).
            action_index: Az agens altal valasztott akcio (0-8).
            bet_amount: A tet merete absolut chip ertekben.
            pot_size: A pot aktualis merete (a bloff intenzitas szamitashoz).
            hand_strength: A kez ereje [0.0, 1.0] (equity proxy).
            lost_showdown: True ha az agens vesztett a showdown-ban.
            is_preflop_raise: True ha pre-flop emelest hajtott vegre.
            street: A leosztas fazisa (0=preflop, 1=flop, stb.).

        Returns:
            A modositott jutalom ertek.
        """
        self._total_calls += 1
        modified_reward: float = base_reward
        penalty: float = 0.0
        bonus: float = 0.0

        # === All-in Spam Buntetes ===
        if self.config.bluff_penalty_lambda > 0.0 and lost_showdown:
            bluff_intensity: float = self._compute_bluff_intensity(
                action_index, bet_amount, pot_size, hand_strength
            )

            if bluff_intensity > self.config.bluff_intensity_threshold:
                penalty = self.config.bluff_penalty_lambda * bluff_intensity
                modified_reward -= penalty
                self._total_penalties += penalty

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Bloff buntetes: intensity=%.3f, lambda=%.4f, "
                        "penalty=%.4f, reward: %.4f -> %.4f",
                        bluff_intensity, self.config.bluff_penalty_lambda,
                        penalty, base_reward, modified_reward,
                    )

        # === Passzivitas Bonus ===
        if self.config.preflop_aggression_bonus > 0.0 and is_preflop_raise:
            bonus = self.config.preflop_aggression_bonus
            modified_reward += bonus
            self._total_bonuses += bonus

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Agresszio bonus: +%.4f, reward: %.4f -> %.4f",
                    bonus, base_reward, modified_reward,
                )

        return modified_reward

    # =========================================================================
    # Bloff Intenzitas Szamitas
    # =========================================================================

    @staticmethod
    def _compute_bluff_intensity(
        action_index: int,
        bet_amount: float,
        pot_size: float,
        hand_strength: float,
    ) -> float:
        """Kiszamitja a bloff intenzitasat.

        A bloff intenzitas a tet/pot arany es a kez gyengesegenek szorzata.
        Magas intenzitas = nagy tet gyenge kezzel.

        Args:
            action_index: Az akcio indexe (0-8).
            bet_amount: A tet merete.
            pot_size: A pot merete.
            hand_strength: A kez ereje [0.0, 1.0].

        Returns:
            Bloff intenzitas [0.0, inf).
        """
        if pot_size <= 0:
            pot_size = 1.0

        bet_ratio: float = bet_amount / pot_size
        weakness: float = max(0.0, 1.0 - hand_strength)

        # All-in (index 8) extra sulyozast kap
        if action_index == 8:
            bet_ratio = max(bet_ratio, 2.0)

        intensity: float = bet_ratio * weakness
        return intensity

    # =========================================================================
    # Hot-Reload Parameterek
    # =========================================================================

    def update_penalty_lambda(self, new_lambda: float) -> None:
        """Futasidejuleg modositja a bloff bunteto szorzot.

        Az Orchestrator hivja meg All-in Spam detektalasakor.

        Args:
            new_lambda: Az uj lambda ertek.
        """
        old: float = self.config.bluff_penalty_lambda
        self.config.bluff_penalty_lambda = new_lambda
        logger.info("Bloff lambda frissitve: %.4f -> %.4f", old, new_lambda)

    def update_aggression_bonus(self, new_bonus: float) -> None:
        """Futasidejuleg modositja a passzivitas bonuszt.

        Az Orchestrator hivja meg Passzivitas detektalasakor.

        Args:
            new_bonus: Az uj bonus ertek.
        """
        old: float = self.config.preflop_aggression_bonus
        self.config.preflop_aggression_bonus = new_bonus
        logger.info("Agresszio bonus frissitve: %.4f -> %.4f", old, new_bonus)

    def deactivate_all_shaping(self) -> None:
        """Kikapcsol minden jutalom modositast (visszaall a nyers EV-re)."""
        self.config.bluff_penalty_lambda = 0.0
        self.config.preflop_aggression_bonus = 0.0
        logger.info("Reward Shaping deaktivalva: nyers chip-EV mod.")

    # =========================================================================
    # Statisztikak
    # =========================================================================

    def get_stats(self) -> dict[str, float]:
        """Visszaadja a reward shaping osszesitett statisztikait.

        Returns:
            Dict a kovetkezo kulcsokkal:
                - total_penalties: Osszes kiosztott buntetes
                - total_bonuses: Osszes kiosztott bonus
                - total_calls: Osszes shape_reward hivas szama
                - avg_penalty: Atlagos buntetes hivasankent
                - avg_bonus: Atlagos bonus hivasankent
                - current_lambda: Aktualis bunteto szorzo
                - current_bonus: Aktualis bonus ertek
        """
        n: int = max(self._total_calls, 1)
        return {
            "total_penalties": self._total_penalties,
            "total_bonuses": self._total_bonuses,
            "total_calls": float(self._total_calls),
            "avg_penalty": self._total_penalties / n,
            "avg_bonus": self._total_bonuses / n,
            "current_lambda": self.config.bluff_penalty_lambda,
            "current_bonus": self.config.preflop_aggression_bonus,
        }
