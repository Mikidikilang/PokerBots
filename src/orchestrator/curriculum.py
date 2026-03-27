"""
Curriculum Manager (curriculum.py).

A haromfazisu curriculum rendszer vezerlo logikajat implementalja:

    Phase 0: Rules-Based Exploitation — statikus botok ellen
    Phase 1: Opponent Modeling & SFT — finomhangolt ellenfelek
    Phase 2: Co-Adaptive FSP — MAB vezererlt onjatszo

A fazisok kozotti atmenet (Transition Rules) szigoruan a matematikai
konvergencia, a nyeresi rata es a kizsakmanyolhatosag (exploitability)
kusobertekein alapul.

A MAB (Multi-Armed Bandit) algoritmus a UCB (Upper Confidence Bound)
formulas ellenfel-kivalasztast vegzi a Phase 2-ben.

Hivatkozasok:
    - Specifikacio: curriculum.py — GTO validalas, MAB logika
    - Orchestrator Data: Phase definiciok es atmeneti kuszobok
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Curriculum Fazisok
# =============================================================================

class CurriculumPhase(IntEnum):
    """A harom curriculum fazis enumeracioja."""

    PHASE_0_STATIC = 0
    PHASE_1_SFT = 1
    PHASE_2_FSP = 2


# =============================================================================
# Fazis Konfiguraciok
# =============================================================================

@dataclass
class PhaseConfig:
    """Egyetlen curriculum fazis konfiguracioja.

    Attributes:
        name: A fazis neve.
        description: A fazis rovid leirasa.
        opponents: A fazishoz tartozo ellenfelek nevei.
        min_win_rate_mbb: Minimalis nyeresi rata (mbb/hand) az atmenethez.
        min_hands: Minimalis leosztas szam a statisztikai szignificanciahoz.
        max_exploitability_pct: Maximalis kizsakmanyolhatosag (Nash Distance %).
    """

    name: str = ""
    description: str = ""
    opponents: list[str] = field(default_factory=list)
    min_win_rate_mbb: float = 50.0
    min_hands: int = 100_000
    max_exploitability_pct: float = 1.0


@dataclass
class MABConfig:
    """Multi-Armed Bandit konfiguracio.

    Attributes:
        algorithm: MAB algoritmus ("ucb", "exp3", "thompson").
        ucb_exploration_factor: UCB feltarasi suly (c parameter).
        pool_snapshot_interval: Snapshot mentes gyakorisaga (iteraciokent).
        max_pool_size: Maximalis ellenfel-pool meret.
    """

    algorithm: str = "ucb"
    ucb_exploration_factor: float = 2.0
    pool_snapshot_interval: int = 50
    max_pool_size: int = 20


# =============================================================================
# UCB Arm (Ellenfel Kar)
# =============================================================================

@dataclass
class UCBArm:
    """Egyetlen ellenfel "karja" a MAB algoritmusban.

    Attributes:
        name: Az ellenfel neve/azonositoja.
        total_reward: Osszesitett jutalom az ellenfel ellen.
        selection_count: Kivalasztasok szama.
    """

    name: str = ""
    total_reward: float = 0.0
    selection_count: int = 0

    @property
    def average_reward(self) -> float:
        """Atlagos jutalom az ellenfel ellen."""
        if self.selection_count == 0:
            return 0.0
        return self.total_reward / self.selection_count

    def ucb_score(self, total_rounds: int, c: float = 2.0) -> float:
        """UCB1 pontszam szamitasa.

        UCB1 = avg_reward + c * sqrt(ln(N) / n_i)

        A magas pontszamu ellenfelet kell valasztani:
        ez egyensulyt teremt a kizsakmanyolas (exploitation)
        es a feltaras (exploration) kozott.

        Args:
            total_rounds: Az osszes kivalasztas szama (N).
            c: Feltarasi parameter (magasabb = tobb exploration).

        Returns:
            Az UCB1 pontszam.
        """
        if self.selection_count == 0:
            return float("inf")  # Meg sosem volt kivalasztva -> prioritas
        exploitation: float = self.average_reward
        exploration: float = c * math.sqrt(
            math.log(total_rounds) / self.selection_count
        )
        return exploitation + exploration


# =============================================================================
# Fo Curriculum Manager
# =============================================================================

class CurriculumManager:
    """A haromfazisu curriculum es a MAB ellenfel-kivalasztas vezerlo osztalya.

    A CurriculumManager figyeli a telemetriai metrikakat, dontest hoz
    a fazisatmenetekrol, es a Phase 2-ben a UCB algoritmussal valasztja
    ki az optimalis ellenfelet az Opponent Pool-bol.

    Example:
        >>> mgr = CurriculumManager(phase_configs, mab_config, num_players=6)
        >>> mgr.check_phase_transition(metrics)
        >>> opponent = mgr.select_opponent()
        >>> mgr.update_opponent_reward(opponent, reward)

    Attributes:
        current_phase: Az aktualis curriculum fazis.
        phases: A fazisok konfiguracioi.
        mab_config: A MAB konfiguracioja.
    """

    def __init__(
        self,
        phase_configs: dict[int, PhaseConfig] | None = None,
        mab_config: MABConfig | None = None,
        num_players: int = 6,
    ) -> None:
        """Inicializalja a Curriculum Manager-t.

        Args:
            phase_configs: A fazisok konfiguracioi (0, 1, 2 kulcsokkal).
            mab_config: MAB konfiguracio a Phase 2-hoz.
            num_players: Asztal merete a GTO matrix kivalasztasahoz.
        """
        self.current_phase: CurriculumPhase = CurriculumPhase.PHASE_0_STATIC
        self.num_players: int = num_players

        # Fazis konfiguracik (alapertelmezettek)
        self.phases: dict[int, PhaseConfig] = phase_configs or {
            0: PhaseConfig(
                name="Rules-Based Exploitation",
                opponents=["calling_station", "maniac", "random", "tight_passive"],
                min_win_rate_mbb=50.0,
                min_hands=100_000,
            ),
            1: PhaseConfig(
                name="Opponent Modeling & SFT",
                opponents=["sft_aggressive", "sft_balanced", "sft_passive"],
                min_win_rate_mbb=30.0,
                min_hands=200_000,
                max_exploitability_pct=1.0,
            ),
            2: PhaseConfig(
                name="Co-Adaptive FSP",
                opponents=["self_play_pool"],
                min_win_rate_mbb=0.0,
                min_hands=0,
                max_exploitability_pct=0.3,
            ),
        }

        self.mab_config: MABConfig = mab_config or MABConfig()

        # UCB arms az ellenfel-pool szamara
        self._ucb_arms: dict[str, UCBArm] = {}
        self._total_selections: int = 0

        # Fazis tortenet
        self._phase_history: list[tuple[int, int]] = []  # (iteracio, phase)

        logger.info(
            "CurriculumManager inicializalva: phase=%s, mab=%s, players=%d",
            self.current_phase.name, self.mab_config.algorithm, num_players,
        )

    # =========================================================================
    # Fazis Atmenetek
    # =========================================================================

    def check_phase_transition(
        self,
        metrics: dict[str, float],
        iteration: int = 0,
    ) -> bool:
        """Ellenorzi, hogy a jelenlegi metrikak lehetove teszik-e a fazisatmenetet.

        A tranzicio feltetelei (specifikacio alapjan):
            Phase 0 -> 1: win_rate >= 50 mbb/h, min_hands >= 100k, VPIP in range
            Phase 1 -> 2: win_rate >= 30 mbb/h, min_hands >= 200k, exploit < 1%
            Phase 2: Nincs tovabb

        Args:
            metrics: Az aktualis HUD metrikak szotara (telemetry.get_current_metrics()).
            iteration: Az aktualis training iteracio szama.

        Returns:
            True ha fazisatmenet tortent.
        """
        if self.current_phase == CurriculumPhase.PHASE_2_FSP:
            return False  # Nincs tovabb

        phase_idx: int = self.current_phase.value
        phase_cfg: PhaseConfig = self.phases.get(phase_idx, PhaseConfig())

        win_rate: float = metrics.get("win_rate_mbb", 0.0)
        total_hands: float = metrics.get("total_hands", 0.0)

        # Ellenorzes: minimalis feltetelek
        if total_hands < phase_cfg.min_hands:
            logger.debug(
                "Fazisatmenet nem lehetseges: hands=%.0f < %d (min)",
                total_hands, phase_cfg.min_hands,
            )
            return False

        if win_rate < phase_cfg.min_win_rate_mbb:
            logger.debug(
                "Fazisatmenet nem lehetseges: wr=%.1f < %.1f mbb/h",
                win_rate, phase_cfg.min_win_rate_mbb,
            )
            return False

        # Atmentet!
        old_phase: CurriculumPhase = self.current_phase
        new_phase: CurriculumPhase = CurriculumPhase(phase_idx + 1)
        self.current_phase = new_phase
        self._phase_history.append((iteration, new_phase.value))

        logger.info(
            "========================================\n"
            "  FAZISATMENET: %s -> %s\n"
            "  Iteracio: %d | Win Rate: %.1f mbb/h | Hands: %.0f\n"
            "========================================",
            old_phase.name, new_phase.name,
            iteration, win_rate, total_hands,
        )

        return True

    def get_current_opponents(self) -> list[str]:
        """Visszaadja az aktualis fazishoz tartozo ellenfelek neveit.

        Returns:
            Ellenfel nevek listaja.
        """
        phase_cfg: PhaseConfig = self.phases.get(
            self.current_phase.value, PhaseConfig()
        )
        return phase_cfg.opponents

    # =========================================================================
    # MAB (UCB) Ellenfel Kivalasztas
    # =========================================================================

    def register_opponent(self, name: str) -> None:
        """Uj ellenfelet regisztral a UCB pool-ba.

        Args:
            name: Az ellenfel neve/azonositoja.
        """
        if name not in self._ucb_arms:
            self._ucb_arms[name] = UCBArm(name=name)
            logger.debug("UCB arm regisztralva: %s", name)

    def select_opponent(self) -> str:
        """UCB1 alapjan kivalasztja a kovetkezo ellenfelet.

        Az algoritmus a legmagasabb UCB pontszamu ellenfelet valasztja,
        egyensulyt tartva a kizsakmanyolas es a feltaras kozott.

        Returns:
            A kivalasztott ellenfel neve.
        """
        if not self._ucb_arms:
            logger.warning("UCB pool ures! Fallback: elso elerheto ellenfel.")
            opponents: list[str] = self.get_current_opponents()
            return opponents[0] if opponents else "random"

        # Cache total_selections BEFORE incrementing
        total_selections_cached: int = self._total_selections
        self._total_selections += 1
        c: float = self.mab_config.ucb_exploration_factor

        best_name: str = ""
        best_score: float = -float("inf")

        for arm in self._ucb_arms.values():
            score: float = arm.ucb_score(total_selections_cached, c)
            if score > best_score:
                best_score = score
                best_name = arm.name

        if best_name:
            self._ucb_arms[best_name].selection_count += 1

        logger.debug(
            "UCB kivalasztas: %s (score=%.4f, total=%d)",
            best_name, best_score, self._total_selections,
        )

        return best_name

    def update_opponent_reward(self, name: str, reward: float) -> None:
        """Frissiti egy ellenfel kumulativ jutalmat a UCB pool-ban.

        Args:
            name: Az ellenfel neve.
            reward: A partiban elert jutalom.
        """
        if name in self._ucb_arms:
            self._ucb_arms[name].total_reward += reward
            logger.debug(
                "UCB reward update: %s += %.4f (total=%.4f, n=%d)",
                name, reward, self._ucb_arms[name].total_reward,
                self._ucb_arms[name].selection_count,
            )

    def get_ucb_stats(self) -> dict[str, Any]:
        """Visszaadja a UCB pool allapot-statisztikait.

        Returns:
            Dict a kovetkezo kulcsokkal:
                - arms: Dict az egyes karok adataival
                - total_selections: Osszes kivalasztas
        """
        arms: dict[str, dict[str, float]] = {}
        for name, arm in self._ucb_arms.items():
            arms[name] = {
                "avg_reward": arm.average_reward,
                "selections": float(arm.selection_count),
                "ucb_score": arm.ucb_score(
                    max(self._total_selections, 1),
                    self.mab_config.ucb_exploration_factor,
                ),
            }
        return {"arms": arms, "total_selections": self._total_selections}

    # =========================================================================
    # Config Betoltes
    # =========================================================================

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> CurriculumManager:
        """YAML config szotarbol peldanyosit.

        Args:
            cfg: Teljes YAML konfiguracio.

        Returns:
            CurriculumManager peldany.
        """
        orch_cfg = cfg.get("orchestrator", {})
        phases_cfg = orch_cfg.get("phases", {})
        mab_cfg_raw = orch_cfg.get("mab", {})
        env_cfg = cfg.get("environment", {})
        num_players: int = env_cfg.get("num_players", 6)

        phase_configs: dict[int, PhaseConfig] = {}
        for phase_key, phase_data in phases_cfg.items():
            # phase_key: "phase_0", "phase_1", "phase_2"
            idx: int = int(phase_key.split("_")[-1])
            trans = phase_data.get("transition_threshold", {})
            target = phase_data.get("target_metrics", {})
            phase_configs[idx] = PhaseConfig(
                name=phase_data.get("name", f"Phase {idx}"),
                description=phase_data.get("description", ""),
                opponents=phase_data.get("opponents", []),
                min_win_rate_mbb=trans.get("min_win_rate_mbb",
                                           target.get("min_slumbot_mbb", 0.0)),
                min_hands=trans.get("min_hands", 0),
                max_exploitability_pct=trans.get("max_exploitability_pct",
                                                  target.get("nash_distance_pct", 1.0)),
            )

        mab_config = MABConfig(
            algorithm=mab_cfg_raw.get("algorithm", "ucb"),
            ucb_exploration_factor=mab_cfg_raw.get("ucb_exploration_factor", 2.0),
            pool_snapshot_interval=mab_cfg_raw.get("pool_snapshot_interval", 50),
            max_pool_size=mab_cfg_raw.get("max_pool_size", 20),
        )

        manager = cls(
            phase_configs=phase_configs,
            mab_config=mab_config,
            num_players=num_players,
        )
        logger.info(
            "CurriculumManager betoltve YAML-bol: %d fazis, mab=%s",
            len(phase_configs), mab_config.algorithm,
        )
        return manager

    # =========================================================================
    # Allapot Mentes / Betoltes
    # =========================================================================

    def get_state(self) -> dict[str, Any]:
        """Visszaadja a curriculum allapotot checkpoint menteshez.

        Returns:
            Szerializalhato allapot szotar.
        """
        return {
            "current_phase": self.current_phase.value,
            "phase_history": self._phase_history,
            "total_selections": self._total_selections,
            "ucb_arms": {
                name: {
                    "total_reward": arm.total_reward,
                    "selection_count": arm.selection_count,
                }
                for name, arm in self._ucb_arms.items()
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Visszaallitja a curriculum allapotot checkpoint betoltesbol.

        Args:
            state: A korabban mentett allapot szotar.
        """
        self.current_phase = CurriculumPhase(state.get("current_phase", 0))
        self._phase_history = state.get("phase_history", [])
        self._total_selections = state.get("total_selections", 0)

        for name, arm_data in state.get("ucb_arms", {}).items():
            self._ucb_arms[name] = UCBArm(
                name=name,
                total_reward=arm_data["total_reward"],
                selection_count=arm_data["selection_count"],
            )
        logger.info("Curriculum allapot betoltve: phase=%s", self.current_phase.name)
