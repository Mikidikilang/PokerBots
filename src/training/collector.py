"""
Rollout Collector (src/training/collector.py).

Implements the inner loop of PPO training: steps the environment forward,
queries the network for actions, stores transitions in the replay buffer,
and — after every completed hand — submits a ``HandRecord`` to the
``TelemetryAnalyzer`` so the ``AutoAdaptiveOrchestrator`` can make
data-driven curriculum and intervention decisions.

Public interface
----------------
    RolloutCollector(network, env, obs_builder, buffer, config, orchestrator, device)
    .collect_rollout(n_steps) -> RolloutStats

Protocol contract (PokerEnvironment, required by env)
------------------------------------------------------
    env.reset()           -> dict[str, Any]          (11-key obs dict)
    env.step(action: int) -> tuple[dict, float]      (next_obs, reward)
    env.is_over()         -> bool

Network contract
----------------
    network.get_action_and_value(obs_tensor_dict, action=None)
        -> (action_tensor, log_prob, entropy, value)
    network.get_value(obs_tensor_dict) -> value_tensor

Bug F Fix (this file — Task 1.1)
---------------------------------
The ``AutoAdaptiveOrchestrator`` previously received only aggregated
iteration-level statistics (mean_reward, policy_loss, etc.) and had no
access to per-hand poker data.  As a result, VPIP, PFR, 3-Bet, AF, and
WTSD were always stale/zero, making all curriculum transitions and reward
interventions blind guesses.

The fix:
    1. A ``_HandAccumulator`` dataclass tracks per-hand state during each
       episode: actions taken, street of each action, and preflop raise
       count (for 3-Bet detection).
    2. ``_detect_street(obs)`` derives the current betting round from the
       number of public cards exposed in the observation.
    3. ``_build_hand_record(acc, reward, went_to_showdown)`` computes all
       HUD statistics from the accumulated state and returns a ``HandRecord``.
    4. At every episode termination, the ``HandRecord`` is submitted to
       ``self.orchestrator.telemetry.record_hand()``.

Phase 1.3 improvements (included here)
---------------------------------------
    - ``torch.no_grad()`` → ``torch.inference_mode()`` in the network
      inference path (lower overhead, prevents accidental gradient creation).
    - ``non_blocking=True`` on every ``.to(device)`` call, allowing the
      CUDA DMA engine to overlap data transfer with CPU work.

HandRecord field-to-HUD-stat mapping
--------------------------------------
    Field                    HUD Stat
    ─────────────────────    ─────────────────────────────────────
    vpip                     VPIP  (Voluntarily Put $ In Pot)
    pfr                      PFR   (Pre-Flop Raise %)
    three_bet                3-Bet %
    total_aggressive_actions  AF numerator (bets + raises)
    total_passive_actions     AF denominator (calls)
    went_to_showdown         WTSD (Went To ShowDown)
    won_at_showdown          WSD  (Won $ at ShowDown)
    street_reached           Furthest street seen in this hand
    reward_bb                Net result in big-blind units
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch

# ---------------------------------------------------------------------------
# HandRecord import with local-fallback for testing without full project
# ---------------------------------------------------------------------------
try:
    from src.orchestrator.telemetry import HandRecord, TelemetryAnalyzer  # type: ignore[import]
except ImportError:
    # -----------------------------------------------------------------------
    # LOCAL FALLBACK — mirrors the contract TelemetryAnalyzer expects.
    # If your telemetry.py has different field names, reconcile them here.
    # -----------------------------------------------------------------------
    @dataclass
    class HandRecord:  # type: ignore[no-redef]
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

    # Stub TelemetryAnalyzer so isinstance checks work in tests
    class TelemetryAnalyzer:  # type: ignore[no-redef]
        def record_hand(self, record: HandRecord) -> None: ...


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action index constants — must match action_mapper.PokerAction enum
# ---------------------------------------------------------------------------
_FOLD       = 0
_CHECK_CALL = 1
_MIN_RAISE  = 2
_ALL_IN     = 8
_RAISE_ACTIONS: frozenset[int] = frozenset(range(_MIN_RAISE, _ALL_IN + 1))


# ---------------------------------------------------------------------------
# PokerEnvironment Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PokerEnvironment(Protocol):
    """Structural protocol that env must satisfy.

    ``RLCardWrapper`` (src/env/wrappers.py) is the concrete implementation.
    Any object with these three methods and correct return types satisfies
    the protocol — no inheritance required.
    """

    def reset(self) -> dict[str, Any]:
        """Start a new hand and return the first observation dict."""
        ...

    def step(self, action: int) -> tuple[dict[str, Any], float]:
        """Execute action; return (next_obs_dict, step_reward)."""
        ...

    def is_over(self) -> bool:
        """Return True when the current hand has ended."""
        ...


# ---------------------------------------------------------------------------
# Internal per-hand accumulator
# ---------------------------------------------------------------------------

@dataclass
class _HandAccumulator:
    """Mutable per-episode state built up action-by-action.

    Reset at the start of every episode; consumed by ``_build_hand_record``
    at episode termination.
    """
    hand_id: int
    player_id: int
    position: int
    iteration: int

    # Raw action trace
    actions:        list[int] = field(default_factory=list)
    action_streets: list[int] = field(default_factory=list)

    # Preflop-specific state for 3-bet detection
    preflop_raises_seen: int = 0   # counts raises by any player before us

    def record_action(self, action: int, street: int) -> None:
        """Append one action to the trace and update preflop counters."""
        self.actions.append(action)
        self.action_streets.append(street)
        if street == 0 and action in _RAISE_ACTIONS:
            self.preflop_raises_seen += 1


# ---------------------------------------------------------------------------
# Rollout statistics (returned to runner.py)
# ---------------------------------------------------------------------------

class RolloutStats(NamedTuple):
    """Aggregated statistics from one collect_rollout() call."""
    n_steps:       int
    n_episodes:    int
    mean_reward:   float
    total_reward:  float
    n_hands_submitted: int   # HandRecords submitted to telemetry


# ---------------------------------------------------------------------------
# RolloutCollector
# ---------------------------------------------------------------------------

class RolloutCollector:
    """Collects PPO rollout transitions and feeds the Telemetry Bridge.

    The collector owns one environment instance.  On each call to
    ``collect_rollout(n_steps)`` it:

        1. Steps the environment for exactly ``n_steps`` actions.
        2. Pushes each transition into ``buffer`` for PPO training.
        3. At every episode termination, builds a ``HandRecord`` and submits
           it to ``orchestrator.telemetry.record_hand()``.

    Args:
        network:      ``PokerActorCritic`` instance (on ``device``).
        env:          Object satisfying the ``PokerEnvironment`` Protocol.
        obs_builder:  ``ObservationBuilder`` — converts raw obs dict to
                      tensors via ``obs_builder.build(obs_dict)``.
        buffer:       Rollout buffer with a ``push(**transition)`` method.
        config:       Full YAML config dict.
        orchestrator: ``AutoAdaptiveOrchestrator`` (or None for smoke tests).
        device:       ``torch.device`` for tensor allocation.
    """

    def __init__(
        self,
        network:      torch.nn.Module,
        env:          PokerEnvironment,
        obs_builder:  Any,
        buffer:       Any,
        config:       dict[str, Any],
        orchestrator: Any | None = None,
        device:       torch.device | str = "cpu",
    ) -> None:
        if not isinstance(env, PokerEnvironment):
            raise TypeError(
                f"env must satisfy the PokerEnvironment Protocol "
                f"(reset/step/is_over). Got {type(env).__name__}. "
                "Did you pass an RLCardWrapper?"
            )

        self.network      = network
        self.env          = env
        self.obs_builder  = obs_builder
        self.buffer       = buffer
        self.config       = config
        self.orchestrator = orchestrator
        self.device       = torch.device(device)

        # ── Per-session counters ──────────────────────────────────────────
        self._hand_counter:  int   = 0
        self._total_steps:   int   = 0
        self._iteration:     int   = 0   # updated by runner via set_iteration()

        # ── Live episode state ────────────────────────────────────────────
        self._current_obs:  dict[str, Any] | None = None
        self._episode_done: bool                  = True
        self._hand_acc:     _HandAccumulator | None = None

        # ── Config extraction ─────────────────────────────────────────────
        train_cfg = config.get("training", {})
        self._big_blind: float = float(
            config.get("environment", {}).get("big_blind", 2.0)
        )

        logger.info(
            "RolloutCollector initialised: device=%s, BB=%.1f, "
            "telemetry_enabled=%s",
            self.device,
            self._big_blind,
            orchestrator is not None,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def set_iteration(self, iteration: int) -> None:
        """Called by runner.py to stamp HandRecords with the training iteration."""
        self._iteration = iteration

    def collect_rollout(self, n_steps: int) -> RolloutStats:
        """Collect exactly ``n_steps`` environment steps.

        This is the hot loop of the training pipeline.  The network is
        queried in ``torch.inference_mode()`` so that:
            - No gradient tape is constructed (lower VRAM, faster).
            - Accidental in-place modifications are caught early.

        The loop handles mid-rollout episode boundaries cleanly: when a hand
        ends before ``n_steps`` is exhausted the environment is reset and
        collection continues seamlessly into the next hand.

        Args:
            n_steps: Number of environment steps to collect.

        Returns:
            ``RolloutStats`` namedtuple with aggregated hand/reward stats.
        """
        self.network.eval()

        n_episodes    = 0
        episode_rews  = []
        n_telemetry   = 0
        running_rew   = 0.0

        # ── Lazy reset: initialise on first call or after prev rollout ────
        if self._episode_done or self._current_obs is None:
            self._current_obs  = self.env.reset()
            self._episode_done = False
            self._hand_acc     = self._new_accumulator()
            running_rew        = 0.0

        for step in range(n_steps):
            obs_raw = self._current_obs

            # ── Detect street BEFORE the action ──────────────────────────
            current_street = _detect_street(obs_raw)

            # ── Build tensor dict and query network ───────────────────────
            obs_tensor = self._build_obs_tensor(obs_raw)

            with torch.inference_mode():  # Phase 1.3: was torch.no_grad()
                action, log_prob, entropy, value = (
                    self.network.get_action_and_value(obs_tensor)
                )

            action_int: int = int(action.reshape(-1)[0].item())

            # ── Record action in hand accumulator ─────────────────────────
            if self._hand_acc is not None:
                self._hand_acc.record_action(action_int, current_street)

            # ── Step environment ──────────────────────────────────────────
            next_obs_raw, reward = self.env.step(action_int)
            done: bool = self.env.is_over()

            running_rew += reward
            self._total_steps += 1

            # ── Push transition to buffer ─────────────────────────────────
            # obs_tensor stays on device; scalars move to float/long.
            # Buffer stores tensors; trainer will batch them.
            self.buffer.push(
                obs=obs_tensor,
                action=action.detach(),
                log_prob=log_prob.detach(),
                value=value.detach(),
                reward=torch.tensor(
                    reward, dtype=torch.float32, device=self.device
                ),
                done=torch.tensor(
                    float(done), dtype=torch.float32, device=self.device
                ),
            )

            # ── Episode termination: build HandRecord ─────────────────────
            if done:
                n_episodes    += 1
                n_telemetry   += self._close_episode(
                    terminal_obs=next_obs_raw,
                    reward=running_rew,
                )
                episode_rews.append(running_rew)
                running_rew = 0.0

                # Reset for next episode
                self._current_obs  = self.env.reset()
                self._episode_done = False
                self._hand_acc     = self._new_accumulator()
            else:
                self._current_obs = next_obs_raw

        # ── Bootstrap value for the last (possibly incomplete) episode ────
        # runner.py / GAE computation needs V(s_T).  Store it in the buffer
        # via a dedicated method if the buffer supports it.
        if not done and self._current_obs is not None:
            last_obs_tensor = self._build_obs_tensor(self._current_obs)
            with torch.inference_mode():
                last_value = self.network.get_value(last_obs_tensor)
            if hasattr(self.buffer, "set_last_value"):
                self.buffer.set_last_value(last_value.detach())

        self.network.train()

        mean_reward = (
            sum(episode_rews) / len(episode_rews) if episode_rews else 0.0
        )
        stats = RolloutStats(
            n_steps=n_steps,
            n_episodes=n_episodes,
            mean_reward=mean_reward,
            total_reward=sum(episode_rews),
            n_hands_submitted=n_telemetry,
        )
        logger.debug(
            "collect_rollout done: steps=%d, episodes=%d, "
            "mean_reward=%.4f BB, hands_to_telemetry=%d",
            n_steps, n_episodes, mean_reward, n_telemetry,
        )
        return stats

    # =========================================================================
    # Telemetry Bridge — Bug F Fix
    # =========================================================================

    def _close_episode(
        self,
        terminal_obs: dict[str, Any],
        reward: float,
    ) -> int:
        """Build a HandRecord and submit it to the orchestrator.

        Called exactly once per completed hand.

        Args:
            terminal_obs: The final observation dict returned by the env.
            reward:       Cumulative reward for this hand (in chips).

        Returns:
            1 if a HandRecord was successfully submitted, 0 otherwise.
        """
        if self._hand_acc is None:
            return 0

        # Determine whether a showdown occurred: both players saw the river
        # and neither folded (any fold action would have ended the hand before
        # all 5 community cards were dealt).
        last_street   = _detect_street(terminal_obs)
        any_fold      = _FOLD in self._hand_acc.actions
        went_to_sd    = (last_street == 3) and (not any_fold)
        won_at_sd     = went_to_sd and (reward > 0)

        try:
            record = _build_hand_record(
                acc=self._hand_acc,
                reward=reward,
                big_blind=self._big_blind,
                street_reached=last_street,
                went_to_showdown=went_to_sd,
                won_at_showdown=won_at_sd,
            )
        except Exception as exc:
            logger.warning("_build_hand_record failed: %s", exc, exc_info=True)
            return 0

        return self._submit_hand_record(record)

    def _submit_hand_record(self, record: HandRecord) -> int:
        """Deliver a HandRecord to the orchestrator's telemetry analyzer.

        Failures are logged and suppressed so a bad telemetry submission
        never crashes the training loop.

        Args:
            record: Populated HandRecord.

        Returns:
            1 on success, 0 on failure or when orchestrator is absent.
        """
        if self.orchestrator is None:
            return 0
        try:
            self.orchestrator.telemetry.record_hand(record)
            logger.debug(
                "HandRecord submitted: hand=%d iter=%d street=%d "
                "reward=%.3f BB vpip=%s pfr=%s 3b=%s wtsd=%s",
                record.hand_id, record.iteration, record.street_reached,
                record.reward_bb, record.vpip, record.pfr,
                record.three_bet, record.went_to_showdown,
            )
            return 1
        except AttributeError as exc:
            logger.warning(
                "orchestrator.telemetry.record_hand() not available: %s. "
                "Is orchestrator.telemetry a TelemetryAnalyzer instance?",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "HandRecord submission failed (hand=%d): %s",
                record.hand_id, exc, exc_info=True,
            )
        return 0

    # =========================================================================
    # Observation Tensor Building
    # =========================================================================

    def _build_obs_tensor(
        self, obs_raw: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Convert a raw obs dict to a device-resident tensor dict.

        Calls ``obs_builder.build(obs_raw)`` then moves every tensor to
        ``self.device`` with ``non_blocking=True`` (Phase 1.3).

        Args:
            obs_raw: 11-key dict from ``PokerEnvironment.reset()`` or
                     ``.step()``.

        Returns:
            Dict mapping the same keys to ``torch.Tensor`` on ``self.device``.
        """
        try:
            obs_tensor: dict[str, torch.Tensor] = self.obs_builder.build(obs_raw)
        except Exception as exc:
            raise RuntimeError(
                f"ObservationBuilder.build() failed: {exc}\n"
                f"obs_raw keys: {list(obs_raw.keys())}"
            ) from exc

        # Phase 1.3: non_blocking=True overlaps DMA transfer with CPU work
        return {
            k: v.to(self.device, non_blocking=True)
            for k, v in obs_tensor.items()
        }

    # =========================================================================
    # Accumulator Helpers
    # =========================================================================

    def _new_accumulator(self) -> _HandAccumulator:
        """Create a fresh hand accumulator for the episode just started."""
        self._hand_counter += 1
        return _HandAccumulator(
            hand_id=self._hand_counter,
            player_id=0,     # learning agent is always player-0 in self-play
            position=0,
            iteration=self._iteration,
        )


# =============================================================================
# Module-level helper functions (testable without instantiating the class)
# =============================================================================

def _detect_street(obs: dict[str, Any]) -> int:
    """Derive the current betting street from the number of public cards.

    Mapping:
        0 cards → Preflop (0)
        3 cards → Flop    (1)
        4 cards → Turn    (2)
        5 cards → River   (3)

    Args:
        obs: Raw obs dict with a ``'public_cards'`` list entry.

    Returns:
        Integer in [0, 3].
    """
    n = len(obs.get("public_cards", []))
    if n == 0:
        return 0   # preflop
    if n == 3:
        return 1   # flop
    if n == 4:
        return 2   # turn
    return 3       # river (5 cards, or any unexpected count)


def _build_hand_record(
    acc:               _HandAccumulator,
    reward:            float,
    big_blind:         float,
    street_reached:    int,
    went_to_showdown:  bool,
    won_at_showdown:   bool,
) -> HandRecord:
    """Derive all HUD stats from a completed hand's action trace.

    HUD stat derivations
    --------------------
    VPIP (Voluntarily Put $ In Pot):
        True if the agent made ANY non-fold action preflop.  This is the
        broadest correct definition in a self-play setting where we control
        both the dealer and the blind positions.

    PFR (Pre-Flop Raise %):
        True if the agent raised (action ≥ MIN_RAISE) at any point preflop.

    3-Bet:
        True if the agent raised preflop AND at least one other raise had
        already occurred before the agent's raise.  We track the running
        ``preflop_raises_seen`` counter in ``_HandAccumulator.record_action``
        to detect this: when an agent raise arrives, if
        ``preflop_raises_seen >= 2`` (BB open + one raise before us) the
        agent's raise is a 3-bet.

    AF (Aggression Factor):
        (total bets + total raises) / total calls.  Stored as raw counts
        so TelemetryAnalyzer can accumulate them across many hands before
        dividing (avoids division-by-zero on rare samples).

    WTSD / WSD:
        Derived from ``went_to_showdown`` and ``reward`` respectively.

    Args:
        acc:              ``_HandAccumulator`` for this hand.
        reward:           Raw chip delta (in chips, not BB units).
        big_blind:        Big blind chip value for normalisation.
        street_reached:   Last street with action (0-3).
        went_to_showdown: True if the hand reached showdown.
        won_at_showdown:  True if won at showdown.

    Returns:
        Populated ``HandRecord`` ready for ``TelemetryAnalyzer.record_hand()``.
    """
    actions        = acc.actions
    action_streets = acc.action_streets

    # ── Separate preflop from post-flop actions ───────────────────────────
    preflop_actions = [a for a, s in zip(actions, action_streets) if s == 0]

    # ── VPIP ─────────────────────────────────────────────────────────────
    # Any non-fold action preflop constitutes putting money in voluntarily.
    vpip = any(a != _FOLD for a in preflop_actions)

    # ── PFR ──────────────────────────────────────────────────────────────
    pfr = any(a in _RAISE_ACTIONS for a in preflop_actions)

    # ── 3-Bet ─────────────────────────────────────────────────────────────
    # The accumulator increments preflop_raises_seen AFTER recording each
    # raise.  So when we scan preflop actions in order, a raise is a 3-bet
    # when at least one other raise has already been seen at that point.
    three_bet = False
    pf_raises_before = 0
    for a, s in zip(actions, action_streets):
        if s != 0:
            break
        if a in _RAISE_ACTIONS:
            if pf_raises_before >= 1:        # there was at least one raise before ours
                three_bet = True
                break
            pf_raises_before += 1

    # ── Aggression Factor counts ──────────────────────────────────────────
    total_aggressive = sum(1 for a in actions if a in _RAISE_ACTIONS)
    total_passive    = sum(1 for a in actions if a == _CHECK_CALL)
    total_folds      = sum(1 for a in actions if a == _FOLD)

    # ── Reward normalisation ──────────────────────────────────────────────
    safe_bb     = max(big_blind, 1e-6)
    reward_bb   = reward / safe_bb

    return HandRecord(
        hand_id=acc.hand_id,
        player_id=acc.player_id,
        position=acc.position,
        iteration=acc.iteration,
        reward_bb=reward_bb,
        street_reached=street_reached,
        went_to_showdown=went_to_showdown,
        won_at_showdown=won_at_showdown,
        vpip=vpip,
        pfr=pfr,
        three_bet=three_bet,
        total_aggressive_actions=total_aggressive,
        total_passive_actions=total_passive,
        total_folds=total_folds,
        actions=list(actions),
        action_streets=list(action_streets),
    )
