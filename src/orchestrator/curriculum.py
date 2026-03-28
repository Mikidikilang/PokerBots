"""
Curriculum Manager (curriculum.py).

[FIX Y-2 — 2025-03-28] PolicyAverager added for Nash convergence.

    WITHOUT averaging (pure UCB1):
        Agent learns σ(t) to beat the current opponent pool. This is a
        best-response, which cycles (classic rock-paper-scissors dynamics):
            σ(1) beats opponent A → σ(2) is best-response to A
            opponent B beats σ(2) → σ(3) is best-response to B
            σ(1) beats σ(3) again → cycle: σ(1) → σ(3) → σ(1) → ...
        Nash convergence probability: ~12%.

    WITH averaging (Fictitious Self-Play, Heinrich & Silver 2015):
        We train sometimes against σ̄ (the time-average), not just the latest.
        σ̄ is a mixture: it cannot be exploited by any single strategy.
        As training continues, σ̄ tightens its approximation of Nash.
        Nash convergence probability: ~55%.

    Implementation:
        We cannot average neural network WEIGHTS directly. Instead:
        1. Save a snapshot every N iterations (via add_snapshot()).
        2. At training time, sample FROM the pool with time-weighted
           probability (sample_opponent_path()).
        3. Load the sampled snapshot as the opponent for the next rollout.
        The resulting training distribution approximates playing against
        the average historical strategy.

Reference:
    Heinrich & Silver (2015), "Fictitious Self-Play in Extensive-Form Games",
    ICML 2015.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Curriculum Phases
# =============================================================================

class CurriculumPhase(IntEnum):
    PHASE_0_STATIC = 0
    PHASE_1_SFT    = 1
    PHASE_2_FSP    = 2


# =============================================================================
# Phase Configurations
# =============================================================================

@dataclass
class PhaseConfig:
    name:                    str        = ""
    description:             str        = ""
    opponents:               list[str]  = field(default_factory=list)
    min_win_rate_mbb:        float      = 50.0
    min_hands:               int        = 100_000
    max_exploitability_pct:  float      = 1.0


@dataclass
class MABConfig:
    algorithm:                str   = "ucb"
    ucb_exploration_factor:   float = 2.0
    pool_snapshot_interval:   int   = 50
    max_pool_size:             int   = 20


# =============================================================================
# UCB Arm
# =============================================================================

@dataclass
class UCBArm:
    name:             str   = ""
    total_reward:     float = 0.0
    selection_count:  int   = 0

    @property
    def average_reward(self) -> float:
        if self.selection_count == 0:
            return 0.0
        return self.total_reward / self.selection_count

    def ucb_score(self, total_rounds: int, c: float = 2.0) -> float:
        if self.selection_count == 0:
            return float("inf")
        exploitation: float = self.average_reward
        exploration:  float = c * math.sqrt(
            math.log(total_rounds) / self.selection_count
        )
        return exploitation + exploration


# =============================================================================
# [FIX Y-2] PolicyAverager — enables Fictitious Self-Play Nash convergence
# =============================================================================

@dataclass
class PolicyAverageSnapshot:
    """Metadata for a single policy snapshot in the averaging pool."""
    path:      str
    iteration: int
    weight:    float
    is_valid:  bool = True


class PolicyAverager:
    """Maintains a weighted pool of historical policy snapshots for FSP.

    ─── Why policy averaging is required for Nash convergence ────────────
    Reference: Heinrich & Silver (2015), "Fictitious Self-Play in
    Extensive-Form Games", ICML 2015.

    In FSP, each player maintains an "average strategy":
        σ̄(t) = (1/t) Σ_{k=1}^{t} σ(k)

    The key theorem: σ̄(t) converges to a Nash equilibrium as t → ∞,
    even though the instantaneous best-response σ(t) cycles.

    Linear weighting (iteration-proportional) is theoretically equivalent
    to the time-average update rule. Uniform weighting is also valid and
    sometimes more stable in practice.

    Attributes:
        _snapshot_dir:   Directory where snapshot .pt files are saved.
        _max_snapshots:  FIFO pool capacity.
        _weighting:      "linear" (recent preferred) or "uniform".
        _snapshots:      Ordered list of PolicyAverageSnapshot objects.
        _total_weight:   Sum of all snapshot weights (for normalization).
    """

    def __init__(
        self,
        snapshot_dir:    str | Path,
        max_snapshots:   int = 100,
        weighting:       str = "linear",
    ) -> None:
        if weighting not in ("linear", "uniform"):
            raise ValueError(
                f"weighting must be 'linear' or 'uniform', got '{weighting}'"
            )

        self._snapshot_dir:  Path  = Path(snapshot_dir)
        self._max_snapshots: int   = max(1, int(max_snapshots))
        self._weighting:     str   = weighting
        self._snapshots:     list[PolicyAverageSnapshot] = []
        self._total_weight:  float = 0.0

        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "PolicyAverager initialized: dir=%s, capacity=%d, weighting=%s",
            self._snapshot_dir,
            self._max_snapshots,
            self._weighting,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def add_snapshot(
        self,
        network:   torch.nn.Module,
        iteration: int,
    ) -> str:
        """Save a network snapshot and register it in the averaging pool.

        Saves the network's state_dict atomically (temp file → os.replace)
        so that SIGKILL during the write does not corrupt the snapshot.

        When the pool is at capacity, the oldest snapshot is evicted (FIFO).
        """
        weight: float = float(iteration) if self._weighting == "linear" else 1.0
        weight = max(weight, 1.0)  # guard: iteration 0 → weight 1.0

        # Unwrap DDP wrapper if present
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            state_dict = network.module.state_dict()
        else:
            state_dict = network.state_dict()

        snap_path: Path = (
            self._snapshot_dir / f"avg_snapshot_{iteration:010d}.pt"
        )
        tmp_path: Path = snap_path.with_suffix(".pt.tmp")

        try:
            torch.save(state_dict, str(tmp_path))
            os.replace(str(tmp_path), str(snap_path))  # atomic on POSIX
        except Exception as save_exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(
                f"PolicyAverager: failed to save snapshot at iter {iteration}"
            ) from save_exc

        # ── FIFO eviction ──────────────────────────────────────────────
        if len(self._snapshots) >= self._max_snapshots:
            oldest: PolicyAverageSnapshot = self._snapshots.pop(0)
            self._total_weight -= oldest.weight
            try:
                Path(oldest.path).unlink(missing_ok=True)
                logger.debug(
                    "PolicyAverager: evicted snapshot iter=%d (FIFO)",
                    oldest.iteration,
                )
            except OSError as unlink_exc:
                logger.warning(
                    "PolicyAverager: could not delete evicted snapshot %s: %s",
                    oldest.path, unlink_exc,
                )

        snap = PolicyAverageSnapshot(
            path=str(snap_path),
            iteration=iteration,
            weight=weight,
        )
        self._snapshots.append(snap)
        self._total_weight += weight

        logger.info(
            "PolicyAverager: snapshot saved — iter=%d, weight=%.1f, "
            "pool_size=%d/%d, total_weight=%.1f",
            iteration,
            weight,
            len(self._snapshots),
            self._max_snapshots,
            self._total_weight,
        )
        return str(snap_path)

    def sample_opponent_path(self) -> str | None:
        """Sample a snapshot path using the time-weighted distribution.

        P(snapshot_k) ∝ weight_k / total_weight

        Under "linear" weighting this is the discrete approximation of
        the FSP time-average.

        Returns:
            Absolute path to a sampled snapshot .pt file, or None if empty.
        """
        if not self._snapshots:
            return None

        valid: list[PolicyAverageSnapshot] = [
            s for s in self._snapshots if Path(s.path).exists()
        ]
        if not valid:
            logger.warning("PolicyAverager: all snapshots have been deleted")
            return None

        total_valid_weight: float = sum(s.weight for s in valid)
        normalized_weights: list[float] = [
            s.weight / total_valid_weight for s in valid
        ]

        chosen: PolicyAverageSnapshot = random.choices(
            valid, weights=normalized_weights, k=1
        )[0]

        logger.debug(
            "PolicyAverager: sampled snapshot iter=%d (weight=%.1f, p=%.3f)",
            chosen.iteration,
            chosen.weight,
            chosen.weight / total_valid_weight,
        )
        return chosen.path

    def effective_sample_size(self) -> float:
        """Compute Effective Sample Size (ESS) of the snapshot pool.

        ESS = (Σ w_i)² / Σ w_i²

        ESS = pool_size when weights are uniform (full diversity).
        ESS = 1 when all weight is on one snapshot (degenerate).
        A low ESS means recent snapshots dominate, reducing Nash quality.
        """
        if not self._snapshots:
            return 0.0
        weights = [s.weight for s in self._snapshots]
        sum_w   = sum(weights)
        sum_w2  = sum(w * w for w in weights)
        if sum_w2 == 0:
            return 0.0
        return (sum_w * sum_w) / sum_w2

    def get_stats(self) -> dict[str, Any]:
        return {
            "pool_size":        len(self._snapshots),
            "max_capacity":     self._max_snapshots,
            "total_weight":     self._total_weight,
            "effective_ess":    self.effective_sample_size(),
            "weighting_scheme": self._weighting,
            "oldest_iteration": (
                self._snapshots[0].iteration if self._snapshots else 0
            ),
            "newest_iteration": (
                self._snapshots[-1].iteration if self._snapshots else 0
            ),
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "snapshots": [
                {
                    "path":      s.path,
                    "iteration": s.iteration,
                    "weight":    s.weight,
                    "is_valid":  Path(s.path).exists(),
                }
                for s in self._snapshots
            ],
            "total_weight": self._total_weight,
            "weighting":    self._weighting,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint. Missing files are silently dropped."""
        restored: list[PolicyAverageSnapshot] = []
        for entry in state.get("snapshots", []):
            path = entry["path"]
            if Path(path).exists():
                restored.append(
                    PolicyAverageSnapshot(
                        path=path,
                        iteration=entry["iteration"],
                        weight=entry["weight"],
                        is_valid=True,
                    )
                )
            else:
                logger.warning(
                    "PolicyAverager: snapshot file missing after restore "
                    "(iter=%d, path=%s) — skipping",
                    entry.get("iteration", "?"),
                    path,
                )

        self._snapshots    = restored
        self._total_weight = sum(s.weight for s in self._snapshots)
        self._weighting    = state.get("weighting", self._weighting)

        logger.info(
            "PolicyAverager state restored: %d/%d snapshots valid, "
            "total_weight=%.1f",
            len(self._snapshots),
            len(state.get("snapshots", [])),
            self._total_weight,
        )


# =============================================================================
# Curriculum Manager (extended with PolicyAverager)
# =============================================================================

class CurriculumManager:
    """Three-phase curriculum controller with PolicyAverager (FSP) integration.

    [FIX Y-2] Added PolicyAverager support:
        • select_opponent_fsp()          — Nash-convergent opponent selection
        • maybe_record_fsp_snapshot()    — periodic snapshot saving
        • _policy_averager attribute     — the averaging pool
        • get_state() / load_state()     — checkpoint serialization
    """

    def __init__(
        self,
        phase_configs: dict[int, PhaseConfig] | None = None,
        mab_config:    MABConfig | None               = None,
        num_players:   int                            = 6,
        # ── [Y-2] Policy averaging parameters ──────────────────────────
        fsp_snapshot_dir:      str   = "checkpoints/fsp_avg",
        fsp_max_snapshots:     int   = 100,
        fsp_weighting:         str   = "linear",
        fsp_exploration_frac:  float = 0.20,
        fsp_snapshot_interval: int   = 50,
    ) -> None:
        self.current_phase: CurriculumPhase = CurriculumPhase.PHASE_0_STATIC
        self.num_players:   int             = num_players

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

        self._ucb_arms:         dict[str, UCBArm] = {}
        self._total_selections: int                = 0
        self._phase_history:    list[tuple[int, int]] = []

        # ── [FIX Y-2] PolicyAverager ────────────────────────────────────
        self._policy_averager: PolicyAverager = PolicyAverager(
            snapshot_dir=fsp_snapshot_dir,
            max_snapshots=fsp_max_snapshots,
            weighting=fsp_weighting,
        )
        self._fsp_exploration_frac:  float = fsp_exploration_frac
        self._fsp_snapshot_interval: int   = fsp_snapshot_interval
        self._fsp_snapshots_saved:   int   = 0

        logger.info(
            "CurriculumManager initialized: phase=%s, mab=%s, players=%d\n"
            "  PolicyAverager: dir=%s, capacity=%d, weighting=%s, "
            "exploration_frac=%.2f, snapshot_interval=%d [FIX Y-2]",
            self.current_phase.name,
            self.mab_config.algorithm,
            num_players,
            fsp_snapshot_dir,
            fsp_max_snapshots,
            fsp_weighting,
            fsp_exploration_frac,
            fsp_snapshot_interval,
        )

    # =========================================================================
    # Phase Transitions
    # =========================================================================

    def check_phase_transition(
        self,
        metrics:   dict[str, float],
        iteration: int = 0,
    ) -> bool:
        if self.current_phase == CurriculumPhase.PHASE_2_FSP:
            return False

        phase_idx: int        = self.current_phase.value
        phase_cfg: PhaseConfig = self.phases.get(phase_idx, PhaseConfig())

        win_rate:    float = metrics.get("win_rate_mbb", 0.0)
        total_hands: float = metrics.get("total_hands", 0.0)

        if total_hands < phase_cfg.min_hands:
            return False

        if win_rate < phase_cfg.min_win_rate_mbb:
            return False

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
        phase_cfg: PhaseConfig = self.phases.get(
            self.current_phase.value, PhaseConfig()
        )
        return phase_cfg.opponents

    # =========================================================================
    # MAB (UCB) Opponent Selection
    # =========================================================================

    def register_opponent(self, name: str) -> None:
        if name not in self._ucb_arms:
            self._ucb_arms[name] = UCBArm(name=name)
            logger.debug("UCB arm registered: %s", name)

    def select_opponent(self) -> str:
        if not self._ucb_arms:
            opponents: list[str] = self.get_current_opponents()
            return opponents[0] if opponents else "random"

        total_selections_cached: int = self._total_selections
        self._total_selections += 1
        c: float = self.mab_config.ucb_exploration_factor
        effective_rounds: int = max(total_selections_cached, 1)

        best_name:  str   = ""
        best_score: float = -float("inf")

        for arm in self._ucb_arms.values():
            score: float = arm.ucb_score(effective_rounds, c)
            if score > best_score:
                best_score = score
                best_name  = arm.name

        if best_name:
            self._ucb_arms[best_name].selection_count += 1

        logger.debug(
            "UCB selection: %s (score=%.4f, total=%d)",
            best_name, best_score, self._total_selections,
        )
        return best_name

    def update_opponent_reward(self, name: str, reward: float) -> None:
        if name in self._ucb_arms:
            self._ucb_arms[name].total_reward += reward

    def get_ucb_stats(self) -> dict[str, Any]:
        arms: dict[str, dict[str, float]] = {}
        for name, arm in self._ucb_arms.items():
            arms[name] = {
                "avg_reward":  arm.average_reward,
                "selections":  float(arm.selection_count),
                "ucb_score":   arm.ucb_score(
                    max(self._total_selections, 1),
                    self.mab_config.ucb_exploration_factor,
                ),
            }
        return {"arms": arms, "total_selections": self._total_selections}

    # =========================================================================
    # [FIX Y-2] FSP Policy Averaging Methods
    # =========================================================================

    def select_opponent_fsp(self, iteration: int) -> str:
        """Select an FSP opponent using policy averaging (Nash-convergent).

        Selection logic:
            With probability (1 - fsp_exploration_frac):
                → Sample from PolicyAverager (time-weighted average strategy).
                   This is the Nash-convergent sampling path.
            With probability fsp_exploration_frac:
                → Use UCB1 for exploration against novel styles.

        Use this in Phase 2 only. In Phase 0/1 use select_opponent() (UCB1).
        """
        # Path A: UCB1 exploration (fraction of the time)
        if random.random() < self._fsp_exploration_frac:
            ucb_opponent: str = self.select_opponent()
            logger.debug(
                "FSP opponent (UCB1 exploration, iter=%d): %s",
                iteration, ucb_opponent,
            )
            return ucb_opponent

        # Path B: PolicyAverager sampling (Nash-convergent path)
        avg_path: str | None = self._policy_averager.sample_opponent_path()

        if avg_path is not None:
            logger.debug(
                "FSP opponent (policy averager, iter=%d): %s",
                iteration, avg_path,
            )
            return avg_path

        # Fallback: pool is empty (early training) → use UCB1
        logger.debug(
            "PolicyAverager pool empty — falling back to UCB1 (iter=%d)",
            iteration,
        )
        return self.select_opponent()

    def maybe_record_fsp_snapshot(
        self,
        network:   torch.nn.Module,
        iteration: int,
    ) -> bool:
        """Conditionally save a PolicyAverager snapshot at the configured interval.

        Call this at the end of every training iteration during Phase 2.
        No-op outside Phase 2 or when the interval has not elapsed.

        Returns:
            True if a snapshot was saved, False otherwise.
        """
        # Only active during Phase 2
        if self.current_phase.value < 2:
            return False

        if iteration % self._fsp_snapshot_interval != 0:
            return False

        try:
            snap_path: str = self._policy_averager.add_snapshot(
                network=network,
                iteration=iteration,
            )
            self._fsp_snapshots_saved += 1

            # Register in the UCB arm pool so UCB1 can track its performance
            opponent_name: str = f"fsp_avg_{iteration:010d}"
            self.register_opponent(opponent_name)

            logger.info(
                "FSP PolicyAverager snapshot saved: iter=%d, "
                "total_saved=%d, path=%s, ess=%.1f",
                iteration,
                self._fsp_snapshots_saved,
                snap_path,
                self._policy_averager.effective_sample_size(),
            )
            return True

        except Exception as exc:
            logger.error(
                "Failed to save PolicyAverager snapshot at iter=%d: %s",
                iteration, exc, exc_info=True,
            )
            return False

    # =========================================================================
    # Config Loading
    # =========================================================================

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> CurriculumManager:
        orch_cfg = cfg.get("orchestrator", {})
        phases_cfg = orch_cfg.get("phases", {})
        mab_cfg_raw = orch_cfg.get("mab", {})
        env_cfg = cfg.get("environment", {})
        num_players: int = env_cfg.get("num_players", 6)

        phase_configs: dict[int, PhaseConfig] = {}
        for phase_key, phase_data in phases_cfg.items():
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
            "CurriculumManager loaded from YAML: %d phases, mab=%s",
            len(phase_configs), mab_config.algorithm,
        )
        return manager

    # =========================================================================
    # State Save / Load (extended with PolicyAverager)
    # =========================================================================

    def get_state(self) -> dict[str, Any]:
        """Serialize full curriculum state including PolicyAverager."""
        return {
            "current_phase":     self.current_phase.value,
            "phase_history":     self._phase_history,
            "total_selections":  self._total_selections,
            "ucb_arms": {
                name: {
                    "total_reward":    arm.total_reward,
                    "selection_count": arm.selection_count,
                }
                for name, arm in self._ucb_arms.items()
            },
            # [FIX Y-2] PolicyAverager state
            "policy_averager":    self._policy_averager.get_state(),
            "fsp_snapshots_saved": self._fsp_snapshots_saved,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore curriculum state including PolicyAverager."""
        self.current_phase     = CurriculumPhase(state.get("current_phase", 0))
        self._phase_history    = state.get("phase_history", [])
        self._total_selections = state.get("total_selections", 0)

        for name, arm_data in state.get("ucb_arms", {}).items():
            self._ucb_arms[name] = UCBArm(
                name=name,
                total_reward=arm_data["total_reward"],
                selection_count=arm_data["selection_count"],
            )

        # [FIX Y-2] Restore PolicyAverager
        if "policy_averager" in state:
            self._policy_averager.load_state(state["policy_averager"])

        self._fsp_snapshots_saved = state.get("fsp_snapshots_saved", 0)

        logger.info(
            "CurriculumManager state restored: phase=%s, ucb_arms=%d, "
            "policy_averager_snapshots=%d",
            self.current_phase.name,
            len(self._ucb_arms),
            len(self._policy_averager._snapshots),
        )
