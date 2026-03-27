"""
RLCard Environment Wrapper (src/env/wrappers.py).

Implements the ``PokerEnvironment`` Protocol required by
``src/training/collector.py``, translating between RLCard's native API and
the named-key state dict consumed by ``ObservationBuilder.build()``.

Protocol Contract (collector.py)
---------------------------------
    reset()           -> dict[str, Any]
    step(action: int) -> tuple[dict[str, Any], float]
    is_over()         -> bool

Required state-dict keys (ObservationBuilder.build contract)
-------------------------------------------------------------
    hand            list[str]    hole cards, SuitRank format  e.g. "SA"
    public_cards    list[str]    board cards, SuitRank format
    pot             float        total pot in absolute chips
    my_chips        float        acting player's remaining stack
    opponent_chips  list[float]  each opponent's remaining stack
    big_blind       float        big blind chip value
    amount_to_call  float        chips required to call the current bet
    min_raise       float        minimum legal raise amount
    position        int          seat index of the acting player (0-based)
    betting_history list[dict]   [{action:int, amount:float, player:int}, ...]
    legal_actions   list[int]    legal action indices within [0, 8]

Card Format Note
----------------
``features.py``'s ``_encode_cards`` expects SuitRank format:
    card[0] ∈ {S, H, D, C}   (suit character FIRST)
    card[1] ∈ {2-9,T,J,Q,K,A} (rank character SECOND)
Example: "SA" = Ace of Spades, "H9" = Nine of Hearts.

RLCard 1.x also uses SuitRank internally.  The ``_to_suitrank`` helper
handles the edge case where a card arrives in RankSuit order ("AS" → "SA").

Action Space Mapping
--------------------
Our 9-action discrete space maps to RLCard's variable-length legal-action
set via a proportional heuristic:

    Our index  Semantics          RLCard action
    ─────────  ─────────────────  ─────────────────────────────────────
    0          Fold               smallest legal ID  (almost always 0)
    1          Check / Call       second-smallest legal ID
    2          Min-Raise          first raise ID  (ID index 2)
    3-7        Pot-relative raise proportionally spread among raise IDs
    8          All-in             largest legal ID

Reward Semantics
----------------
All intermediate steps return reward = 0.0.
At the terminal step (is_over() is True), reward = chip_delta[player_0] / big_blind.
This follows the standard normalised chip-EV convention used in the MASTER_NOTE.

Multi-Player Sequencing
-----------------------
The current implementation uses **pure self-play**: the learning agent
controls every seat.  ``reset()`` returns the first acting player's state;
``step(action)`` advances the hand by exactly one player action and returns
the *next* acting player's state.  All payoffs are measured from player-0's
perspective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.env.action_mapper import ActionMapper, GameContext, PokerAction

logger = logging.getLogger(__name__)

__all__ = ["RLCardWrapper", "WrapperConfig", "make_env"]


# =============================================================================
# Card Format Utilities
# =============================================================================

_SUITS: frozenset[str] = frozenset({"S", "H", "D", "C"})
_RANKS: frozenset[str] = frozenset({
    "2", "3", "4", "5", "6", "7", "8", "9",
    "T", "J", "Q", "K", "A",
})


def _to_suitrank(card: str) -> str:
    """Normalise a two-character card string to SuitRank format.

    ``features.py`` requires ``card[0] ∈ {S,H,D,C}`` and
    ``card[1] ∈ {2-9,T,J,Q,K,A}``.

    Handles:
        ``"SA"`` → ``"SA"``  already SuitRank, pass through
        ``"AS"`` → ``"SA"``  RankSuit → SuitRank swap
        ``"sa"`` → ``"SA"``  lowercase → uppercase then normalise

    Args:
        card: Two-character card identifier in any supported format.

    Returns:
        Normalised SuitRank string, or the uppercased input unchanged if
        the format cannot be determined (the caller's ValueError from
        ``_encode_cards`` will surface the problem cleanly).
    """
    card = card.strip().upper()
    if len(card) != 2:
        return card
    c0, c1 = card[0], card[1]
    if c0 in _SUITS and c1 in _RANKS:
        return card          # already correct
    if c0 in _RANKS and c1 in _SUITS:
        return c1 + c0       # swap: RankSuit → SuitRank
    logger.debug("Unrecognised card format '%s'; passing through as-is.", card)
    return card


def _normalise_cards(cards: list[str]) -> list[str]:
    """Apply ``_to_suitrank`` to every non-empty card in the list."""
    return [_to_suitrank(c) for c in cards if c and c.strip()]


# =============================================================================
# Wrapper Configuration
# =============================================================================

@dataclass
class WrapperConfig:
    """Configuration for ``RLCardWrapper``, mirroring ``config.yaml``'s environment section.

    Attributes:
        num_players:       Table size (2–9 inclusive).
        big_blind:         Big blind value in absolute chips.
        small_blind:       Small blind value in absolute chips.
        initial_stack_bb:  Each player's starting stack in big-blind units.
        game_id:           RLCard environment identifier string.
    """

    num_players: int = 6
    big_blind: float = 2.0
    small_blind: float = 1.0
    initial_stack_bb: float = 200.0
    game_id: str = "no-limit-holdem"

    @property
    def initial_stack(self) -> float:
        """Starting stack in absolute chips (``initial_stack_bb × big_blind``)."""
        return self.initial_stack_bb * self.big_blind

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> WrapperConfig:
        """Construct from the full ``config.yaml`` dict.

        Reads from the ``environment`` top-level section.  All keys are
        optional; sensible defaults are used when absent.

        Args:
            cfg: Full YAML configuration dictionary.

        Returns:
            Populated ``WrapperConfig`` instance.
        """
        env = cfg.get("environment", {})
        return cls(
            num_players=int(env.get("num_players", 6)),
            big_blind=float(env.get("big_blind", 2)),
            small_blind=float(env.get("small_blind", 1)),
            initial_stack_bb=float(env.get("initial_stack_bb", 200)),
            game_id=str(env.get("game_type", "no-limit-holdem")),
        )


# =============================================================================
# Internal action-index constants (mirror PokerAction enum from action_mapper.py)
# =============================================================================

_FOLD       = 0
_CHECK_CALL = 1
_MIN_RAISE  = 2
_RAISE_HALF = 3   # 0.50 × pot
_RAISE_75   = 4   # 0.75 × pot
_RAISE_POT  = 5   # 1.00 × pot
_RAISE_150  = 6   # 1.50 × pot
_RAISE_2X   = 7   # 2.00 × pot
_ALL_IN     = 8

# Number of distinct raise indices in our space (2 through 7)
_N_RAISE_LEVELS: int = _RAISE_2X - _MIN_RAISE  # == 5


# =============================================================================
# RLCard Wrapper
# =============================================================================

class RLCardWrapper:
    """Adapts RLCard's no-limit-holdem to the ``PokerEnvironment`` Protocol.

    The wrapper is responsible for three translation tasks:

    1. **State translation** — ``_build_obs_dict()`` converts an RLCard state
       dict into the 11-key format ``ObservationBuilder.build()`` expects.

    2. **Action → RLCard ID** — ``_map_our_action_to_rlcard()`` converts one
       of our 9 action indices into the closest legal RLCard action ID.

    3. **RLCard legal actions → our mask** — ``_rlcard_legal_to_our_mask()``
       produces the list of legal action indices for the action-mask tensor.

    See the module docstring for full details on the design choices.
    """

    def __init__(self, config: WrapperConfig | None = None) -> None:
        """Initialise the wrapper and create the underlying RLCard environment.

        The ``rlcard`` import is deferred to this method so that the module
        can be imported (e.g., for testing with mocks) even when rlcard is
        not installed.

        Args:
            config: Wrapper configuration.  ``WrapperConfig()`` defaults used
                    if ``None``.

        Raises:
            ImportError: If the ``rlcard`` package is not installed.
            RuntimeError: If RLCard fails to create the environment.
        """
        try:
            import rlcard  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'rlcard' package is required by RLCardWrapper.\n"
                "Install it with:  pip install 'rlcard>=1.1.0'"
            ) from exc

        self.config: WrapperConfig = config or WrapperConfig()

        # ── Construct the RLCard environment ──────────────────────────
        rlcard_config: dict[str, Any] = {
            "game_num_players": self.config.num_players,
        }
        try:
            self._env = rlcard.make(self.config.game_id, config=rlcard_config)
        except TypeError:
            # Older rlcard versions may not accept keyword config
            logger.warning(
                "rlcard.make('%s', config=...) failed with TypeError; "
                "retrying without the config argument.",
                self.config.game_id,
            )
            self._env = rlcard.make(self.config.game_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create RLCard environment '{self.config.game_id}': {exc}"
            ) from exc

        # ── Action mapper (resolves our indices to chip amounts) ──────
        self._action_mapper: ActionMapper = ActionMapper()

        # ── Per-episode mutable state ─────────────────────────────────
        #   _terminal starts True so that step() cannot be called before
        #   reset() — a RuntimeError is raised instead.
        self._current_player_id: int            = 0
        self._current_state:     dict[str, Any] = {}
        self._hand_start_chips:  list[float]    = []
        self._hand_history:      list[dict]     = []
        self._terminal:          bool           = True

        logger.info(
            "RLCardWrapper initialised: game=%s, players=%d, "
            "stack=%.0f chips (%.0f BB), BB=%.0f",
            self.config.game_id,
            self.config.num_players,
            self.config.initial_stack,
            self.config.initial_stack_bb,
            self.config.big_blind,
        )

    # =========================================================================
    # PokerEnvironment Protocol — Public Interface
    # =========================================================================

    def reset(self) -> dict[str, Any]:
        """Start a new hand and return the first acting player's observation.

        Clears per-hand tracking (betting history, start-of-hand chip counts)
        and calls ``env.reset()``.

        Returns:
            State dict with all 11 keys required by ``ObservationBuilder.build()``.

        Raises:
            RuntimeError: If the underlying RLCard env raises during reset.
        """
        try:
            raw_result = self._env.reset()
        except Exception as exc:
            raise RuntimeError(f"RLCard env.reset() raised: {exc}") from exc

        state, player_id = self._unpack_reset(raw_result)

        self._current_player_id = player_id
        self._current_state     = state
        self._terminal          = False
        self._hand_history      = []

        # Snapshot starting stacks for reward computation at episode end
        raw = self._get_raw_obs(state)
        self._hand_start_chips = self._extract_all_chips(raw)

        obs = self._build_obs_dict(state, player_id)

        logger.debug(
            "reset(): player=%d, starting_stacks=%s",
            player_id,
            [f"{c:.0f}" for c in self._hand_start_chips],
        )
        return obs

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float]:
        """Execute one action and return the next state and step reward.

        The action is translated from our 9-action index to the closest legal
        RLCard action ID, executed, and the resulting RLCard state is
        translated back to our format.

        Reward is ``0.0`` for all non-terminal steps.  At the terminal step
        (``is_over()`` becomes ``True``), reward equals player-0's chip delta
        normalised by the big blind.

        Args:
            action: Discrete action index in ``[0, 8]``.

        Returns:
            ``(obs_dict, reward_in_bb)``: next observation and step reward.

        Raises:
            RuntimeError: If called before ``reset()`` or after episode end.
        """
        if self._terminal:
            raise RuntimeError(
                "step() called on a completed episode — call reset() first."
            )

        # Clamp to valid range defensively
        action = int(max(_FOLD, min(_ALL_IN, action)))

        # ── Build the current game context for ActionMapper ────────────
        raw = self._get_raw_obs(self._current_state)
        ctx = self._build_game_context(raw, self._current_player_id)

        # ── Resolve the action to a chip amount and record it ──────────
        try:
            resolved = self._action_mapper.resolve_action(
                PokerAction(action), ctx
            )
            chip_amount = float(resolved.amount)
        except Exception as exc:
            logger.warning(
                "ActionMapper.resolve_action(%d) failed (%s); "
                "recording 0.0 chip amount.",
                action, exc,
            )
            chip_amount = 0.0

        self._hand_history.append({
            "action": action,
            "amount": chip_amount,
            "player": self._current_player_id,
        })

        # ── Translate to RLCard action ID ──────────────────────────────
        legal = self._current_state.get("legal_actions", {})
        rlcard_id = self._map_our_action_to_rlcard(action, legal, ctx)

        logger.debug(
            "step(): player=%d, our_action=%d → rlcard_id=%d",
            self._current_player_id, action, rlcard_id,
        )

        # ── Execute in RLCard ──────────────────────────────────────────
        try:
            raw_result = self._env.step(rlcard_id)
        except Exception as exc:
            # Last-resort fallback: use the smallest legal action (fold)
            logger.error(
                "RLCard env.step(%d) raised: %s. "
                "Falling back to smallest legal action.",
                rlcard_id, exc,
            )
            fallback_id = min(legal.keys()) if legal else 0
            raw_result = self._env.step(fallback_id)

        next_state, next_player = self._unpack_step(raw_result)
        self._current_player_id = next_player
        self._current_state     = next_state
        self._terminal          = bool(self._env.is_over())

        # ── Compute reward (non-zero only at episode end) ──────────────
        reward = 0.0
        if self._terminal:
            reward = self._compute_terminal_reward()
            logger.debug(
                "Episode terminal: reward=%.4f BB (player-0 perspective)",
                reward,
            )

        obs = self._build_obs_dict(next_state, next_player)
        return obs, reward

    def is_over(self) -> bool:
        """Return ``True`` when the current hand has ended.

        After this returns ``True``, call ``reset()`` to start a new hand.

        Returns:
            Boolean episode-termination flag.
        """
        return self._terminal

    # =========================================================================
    # State Translation
    # =========================================================================

    def _build_obs_dict(
        self,
        state: dict[str, Any],
        player_id: int,
    ) -> dict[str, Any]:
        """Convert an RLCard state dict to the 11-key ObservationBuilder format.

        Computes all derived quantities (pot total, amount_to_call, min_raise)
        from the raw RLCard state fields.  Missing fields are handled
        gracefully with sensible defaults so that unexpected RLCard versions
        do not crash the training pipeline.

        Args:
            state:     RLCard state dict (from ``reset()`` or ``step()``).
            player_id: Seat index of the player whose perspective we want.

        Returns:
            Dict with all 11 keys ``ObservationBuilder.build()`` requires.
        """
        raw = self._get_raw_obs(state)
        n   = self.config.num_players
        bb  = self.config.big_blind
        pid = int(player_id) % n

        # ── Cards (SuitRank normalisation) ─────────────────────────────
        hand         = _normalise_cards(raw.get("hand", []))
        public_cards = _normalise_cards(raw.get("public_cards", []))

        # ── Chip counts ────────────────────────────────────────────────
        all_chips = self._extract_all_chips(raw)
        while len(all_chips) < n:
            all_chips.append(self.config.initial_stack)

        my_chips: float       = float(all_chips[pid])
        opponent_chips: list[float] = [
            float(all_chips[i]) for i in range(n) if i != pid
        ]

        # ── Pot & stakes ───────────────────────────────────────────────
        # RLCard convention:
        #   raw_obs['pot']    = chips committed on PREVIOUS streets
        #   raw_obs['stakes'] = chips committed on the CURRENT street
        pot_base: float     = float(raw.get("pot", 0.0))
        stakes:   list[float] = self._extract_stakes(raw, n)
        pot:      float     = pot_base + sum(stakes)

        # ── Amount to call ─────────────────────────────────────────────
        my_stake:     float = stakes[pid] if pid < len(stakes) else 0.0
        max_stake:    float = max(stakes) if stakes else 0.0
        amount_to_call: float = max(0.0, max_stake - my_stake)

        # ── Minimum raise ──────────────────────────────────────────────
        # Use the value from RLCard if provided; otherwise estimate.
        # Standard rule: minimum raise = previous raise size (or BB if no raise).
        raw_min: float = float(raw.get("min_raise", 0.0))
        if raw_min > 0:
            min_raise: float = raw_min
        elif max_stake > 0:
            # Caller's bet is max_stake; minimum re-raise adds at least that
            min_raise = max_stake * 2.0
        else:
            min_raise = bb * 2.0   # no bet yet — standard open-raise minimum

        # ── Legal actions (our 9-action indices) ───────────────────────
        rlcard_legal = state.get("legal_actions", {})
        legal_indices = self._rlcard_legal_to_our_mask(
            rlcard_legal, my_chips, amount_to_call,
        )

        obs: dict[str, Any] = {
            "hand":            hand,
            "public_cards":    public_cards,
            "pot":             pot,
            "my_chips":        my_chips,
            "opponent_chips":  opponent_chips,
            "big_blind":       bb,
            "amount_to_call":  amount_to_call,
            "min_raise":       min_raise,
            "position":        pid,
            "betting_history": list(self._hand_history),
            "legal_actions":   legal_indices,
        }

        logger.debug(
            "_build_obs_dict: pid=%d hand=%s board=%s "
            "pot=%.1f stack=%.1f call=%.1f legal=%s",
            pid, hand, public_cards,
            pot, my_chips, amount_to_call, legal_indices,
        )
        return obs

    # =========================================================================
    # Action Mapping: Our index → RLCard action ID
    # =========================================================================

    def _map_our_action_to_rlcard(
        self,
        action: int,
        legal: dict[int, Any],
        ctx: GameContext,
    ) -> int:
        """Return the best-matching legal RLCard action ID for our index.

        Mapping heuristic (relies on the convention that RLCard orders its
        action IDs from least aggressive to most aggressive):

            fold (0)       → smallest legal ID (almost always 0)
            check/call (1) → second-smallest legal ID
            all-in (8)     → largest legal ID
            raises (2–7)   → proportionally spread across intermediate IDs

        If the desired class of action is unavailable (e.g., no raises
        legal), the nearest available alternative is returned silently.

        Args:
            action: Our action index in ``[0, 8]``.
            legal:  ``{rlcard_id: info}`` OrderedDict from the current state.
            ctx:    Current ``GameContext`` (reserved for future amount-aware
                    mapping; currently unused but kept in the signature for
                    forward compatibility).

        Returns:
            A legal RLCard action ID.
        """
        if not legal:
            logger.warning(
                "_map_our_action_to_rlcard: empty legal dict — returning 0."
            )
            return 0

        sorted_ids = sorted(legal.keys())
        n          = len(sorted_ids)

        # ── Fold ───────────────────────────────────────────────────────
        if action == _FOLD:
            return sorted_ids[0]

        # ── Check / Call ───────────────────────────────────────────────
        if action == _CHECK_CALL:
            return sorted_ids[1] if n > 1 else sorted_ids[0]

        # ── All-in ─────────────────────────────────────────────────────
        if action == _ALL_IN:
            return sorted_ids[-1]

        # ── Raise actions 2–7 ──────────────────────────────────────────
        raise_ids = sorted_ids[2:]   # fold + call occupy the first two slots
        if not raise_ids:
            # No raises legal; best fallback is check/call
            return sorted_ids[min(1, n - 1)]

        # Linear mapping: action 2 → raise_ids[0], action 7 → raise_ids[-1]
        proportion = (action - _MIN_RAISE) / max(_RAISE_2X - _MIN_RAISE, 1)
        mapped_idx = round(proportion * (len(raise_ids) - 1))
        mapped_idx = max(0, min(mapped_idx, len(raise_ids) - 1))

        return raise_ids[mapped_idx]

    # =========================================================================
    # Legal Action Mask: RLCard legal IDs → Our 9-action mask
    # =========================================================================

    def _rlcard_legal_to_our_mask(
        self,
        rlcard_legal: dict[int, Any],
        my_chips: float,
        amount_to_call: float,
    ) -> list[int]:
        """Compute which of our 9 action indices are currently legal.

        Heuristic (mirrors the inverse of ``_map_our_action_to_rlcard``):

            ≥1 RLCard action  → fold (0) legal
            ≥2 RLCard actions → check/call (1) legal
            ≥3 RLCard actions → min_raise (2) + all_in (8) legal
            ≥4 RLCard actions → also half-pot raise (3)
            ≥5 RLCard actions → also 3/4-pot (4) and full-pot (5) raises
            ≥6 RLCard actions → also 1.5× pot (6)
            ≥7 RLCard actions → also 2× pot (7)

        All-in is always included whenever any raise is legal (the player
        has at least some chips remaining).

        Args:
            rlcard_legal:   ``{rlcard_id: info}`` dict from the env state.
            my_chips:       Acting player's remaining chip count.
            amount_to_call: Chips required to call (used for all-in guard).

        Returns:
            Sorted, deduplicated list of legal action indices.
        """
        if not rlcard_legal:
            return [_FOLD]

        sorted_ids = sorted(rlcard_legal.keys())
        n_total    = len(sorted_ids)
        n_raises   = max(0, n_total - 2)   # subtract fold and call slots

        legal: set[int] = set()

        # Fold — always legal if there is at least one action
        legal.add(_FOLD)

        # Check / Call — legal if ≥2 RLCard actions
        if n_total >= 2:
            legal.add(_CHECK_CALL)

        # Raise family — legal if ≥3 RLCard actions (at least one raise)
        if n_raises >= 1:
            legal.add(_MIN_RAISE)

        if n_raises >= 2:
            legal.add(_RAISE_HALF)

        if n_raises >= 3:
            legal.add(_RAISE_75)
            legal.add(_RAISE_POT)

        if n_raises >= 4:
            legal.add(_RAISE_150)

        if n_raises >= 5:
            legal.add(_RAISE_2X)

        # All-in — legal whenever any raise is available and we still have chips
        if n_raises >= 1 and my_chips > 0:
            legal.add(_ALL_IN)

        return sorted(legal)

    # =========================================================================
    # Reward Computation
    # =========================================================================

    def _compute_terminal_reward(self) -> float:
        """Compute the normalised reward for player 0 at hand completion.

        Primary path: ``env.get_payoffs()[0]`` / big_blind.
        Fallback path: (current_chips[0] - start_chips[0]) / big_blind.

        Returns:
            Float reward in big-blind units. Positive = chip gain,
            negative = chip loss.
        """
        bb = self.config.big_blind

        # ── Primary: env.get_payoffs() ────────────────────────────────
        try:
            payoffs = self._env.get_payoffs()
            return float(payoffs[0]) / bb
        except Exception as exc:
            logger.warning(
                "env.get_payoffs() failed (%s); computing reward from chip delta.",
                exc,
            )

        # ── Fallback: chip-delta heuristic ────────────────────────────
        try:
            raw       = self._get_raw_obs(self._current_state)
            end_chips = self._extract_all_chips(raw)
            start     = (
                self._hand_start_chips[0]
                if self._hand_start_chips
                else self.config.initial_stack
            )
            end = float(end_chips[0]) if end_chips else start
            return (end - start) / bb
        except Exception as exc:
            logger.error(
                "Fallback reward computation also failed (%s); returning 0.0.",
                exc,
            )
            return 0.0

    # =========================================================================
    # Game Context Construction
    # =========================================================================

    def _build_game_context(
        self,
        raw: dict[str, Any],
        player_id: int,
    ) -> GameContext:
        """Construct a ``GameContext`` for the current player from a raw obs dict.

        ``GameContext`` is consumed by ``ActionMapper`` to resolve our
        abstract action indices into concrete chip amounts.

        Args:
            raw:       Raw obs dict (from ``_get_raw_obs``).
            player_id: Acting player's seat index.

        Returns:
            Populated ``GameContext`` instance.
        """
        n   = self.config.num_players
        bb  = self.config.big_blind
        pid = int(player_id) % n

        all_chips = self._extract_all_chips(raw)
        while len(all_chips) < n:
            all_chips.append(self.config.initial_stack)
        my_chips: float = float(all_chips[pid])

        stakes:     list[float] = self._extract_stakes(raw, n)
        pot_base:   float       = float(raw.get("pot", 0.0))
        pot:        float       = pot_base + sum(stakes)

        my_stake:       float = stakes[pid] if pid < len(stakes) else 0.0
        max_stake:      float = max(stakes) if stakes else 0.0
        amount_to_call: float = max(0.0, max_stake - my_stake)

        raw_min: float = float(raw.get("min_raise", 0.0))
        if raw_min > 0:
            min_raise: float = raw_min
        elif max_stake > 0:
            min_raise = max_stake * 2.0
        else:
            min_raise = bb * 2.0

        return GameContext(
            pot_size=pot,
            my_stack=my_chips,
            amount_to_call=amount_to_call,
            min_raise_amount=min_raise,
            big_blind=bb,
        )

    # =========================================================================
    # RLCard State Extraction Helpers
    # =========================================================================

    def _get_raw_obs(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract the structured ``raw_obs`` sub-dict from an RLCard state.

        RLCard 1.x states have a nested structure::

            state['obs']         → flat numpy array  (for baseline agents)
            state['raw_obs']     → structured dict   ← we want this
            state['legal_actions'] → OrderedDict

        Some RLCard builds flatten ``raw_obs`` into the top-level state dict.
        This method handles both layouts.

        Args:
            state: RLCard state dict returned by ``reset()`` or ``step()``.

        Returns:
            The ``raw_obs`` sub-dict, or the top-level ``state`` as a fallback.
        """
        raw = state.get("raw_obs")
        if isinstance(raw, dict):
            return raw
        # Flat layout: fields merged directly into state
        return state

    def _extract_all_chips(self, raw: dict[str, Any]) -> list[float]:
        """Extract per-player remaining chip counts from a raw obs dict.

        Tries multiple field names for cross-version RLCard compatibility:
            ``'all_chips'``  primary name in RLCard 1.x
            ``'raw_chips'``  alternate name used in some builds
            ``'chips'``      legacy name

        Args:
            raw: Raw obs dict (from ``_get_raw_obs``).

        Returns:
            List of chip floats, one per player.  Falls back to
            ``[initial_stack] × num_players`` if no chip field is present.
        """
        for key in ("all_chips", "raw_chips", "chips"):
            chips = raw.get(key)
            if chips is not None:
                try:
                    return [float(c) for c in chips]
                except (TypeError, ValueError):
                    continue
        logger.debug("No chip field found in raw_obs; using initial_stack defaults.")
        return [self.config.initial_stack] * self.config.num_players

    def _extract_stakes(self, raw: dict[str, Any], n: int) -> list[float]:
        """Extract per-player current-street bets from a raw obs dict.

        ``stakes[i]`` = chips player ``i`` has committed in the **current
        betting round** (resets to zero at the start of each new street).

        Tries: ``'stakes'``, ``'bets'``, ``'current_bets'``.

        Args:
            raw: Raw obs dict.
            n:   Expected number of players (result is padded / truncated).

        Returns:
            Float list of length ``n`` (missing entries zero-padded).
        """
        for key in ("stakes", "bets", "current_bets"):
            raw_stakes = raw.get(key)
            if raw_stakes is not None:
                try:
                    result: list[float] = [float(s) for s in raw_stakes]
                    while len(result) < n:
                        result.append(0.0)
                    return result[:n]
                except (TypeError, ValueError):
                    continue
        return [0.0] * n

    # =========================================================================
    # RLCard API Version-Compatibility Helpers
    # =========================================================================

    @staticmethod
    def _unpack_reset(result: Any) -> tuple[dict[str, Any], int]:
        """Unpack ``env.reset()`` into ``(state_dict, player_id)``.

        RLCard 1.x returns ``(state, player_id)``.
        Older or custom builds may return just ``state``.

        Args:
            result: Direct return value of ``env.reset()``.

        Returns:
            ``(state_dict, player_id)`` — ``player_id`` defaults to 0.
        """
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            state, player_id = result[0], result[1]
            state = dict(state) if not isinstance(state, dict) else state
            return state, int(player_id)
        state = dict(result) if not isinstance(result, dict) else result
        return state, 0

    @staticmethod
    def _unpack_step(result: Any) -> tuple[dict[str, Any], int]:
        """Unpack ``env.step()`` into ``(next_state_dict, next_player_id)``.

        Args:
            result: Direct return value of ``env.step()``.

        Returns:
            ``(next_state_dict, next_player_id)`` — defaults to player 0.
        """
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            state, player_id = result[0], result[1]
            state = dict(state) if not isinstance(state, dict) else state
            return state, int(player_id)
        state = dict(result) if not isinstance(result, dict) else result
        return state, 0

    # =========================================================================
    # Public Utility Methods
    # =========================================================================

    def get_player_id(self) -> int:
        """Return the seat index of the player currently to act."""
        return self._current_player_id

    def get_num_actions(self) -> int:
        """Return the size of our discrete action space (always 9)."""
        return 9

    def get_betting_history(self) -> list[dict[str, Any]]:
        """Return a copy of the current hand's accumulated betting history."""
        return list(self._hand_history)


# =============================================================================
# Factory Function
# =============================================================================

def make_env(cfg: dict[str, Any] | WrapperConfig | None = None) -> RLCardWrapper:
    """Create a fully configured ``RLCardWrapper`` from a YAML config dict or WrapperConfig.

    This factory function implements polymorphic type coercion to accept both
    native Python dictionaries and fully instantiated WrapperConfig objects,
    providing a seamless and safe interface regardless of input type.

    This is intended as a drop-in replacement for the raw ``rlcard.make()``
    call in ``scripts/train_local.py``.  Update
    ``train_local.create_environment()`` to call this function instead:

    .. code-block:: python

        # Before (broken — RLCard env doesn't conform to Protocol):
        env = rlcard.make("no-limit-holdem", config={...})

        # After (correct — returns a PokerEnvironment-compliant wrapper):
        from src.env.wrappers import make_env
        env = make_env(yaml_config)  # dict or WrapperConfig; both work

    Args:
        cfg: Full ``config.yaml`` dict, a WrapperConfig instance, or ``None``
             to use all defaults. Polymorphic type coercion ensures both
             native Python dicts and instantiated WrapperConfig objects are
             safely ingested without AttributeError.

    Returns:
        Configured ``RLCardWrapper`` instance ready to call ``reset()`` on.

    Raises:
        TypeError: If cfg is not dict, WrapperConfig, or None.
    """
    # Polymorphic type coercion: handle dict, WrapperConfig, or None
    if cfg is None:
        config = WrapperConfig()
    elif isinstance(cfg, WrapperConfig):
        # Already a WrapperConfig; use it directly
        config = cfg
    elif isinstance(cfg, dict):
        # Native Python dict; construct WrapperConfig from it
        config = WrapperConfig.from_dict(cfg)
    else:
        raise TypeError(
            f"cfg must be dict, WrapperConfig, or None; got {type(cfg).__name__}"
        )

    return RLCardWrapper(config)
