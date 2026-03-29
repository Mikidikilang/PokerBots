"""
RLCard Environment Wrapper (src/env/wrappers.py).

[FIX Y-1 COMPLETION — 2025-03-28] street field added to betting history entries.

    features.py FIX Y-1 added column 10 (normalized street) to the betting
    history tensor. However RLCardWrapper._hand_history never emitted a
    "street" key, so step.get("street", 0) always returned 0 — every
    action looked like a preflop action and column 10 carried no information.

    Fix: _current_street is tracked from public-card count and injected
    into every history entry:
        {"action": int, "amount": float, "player": int, "street": int}

[FIX P0-A — 2025-03-28] MultiAgentRLCardWrapper added.

    The OpponentPool / CurriculumManager / UCB1 MAB were architecturally
    disconnected from the training loop: opponents were never drawn from
    the configured pool during rollout collection.  RLCard handled all
    non-hero seats internally with its default (random) agent.

    MultiAgentRLCardWrapper fixes this by:
      1. Designating seat 0 as the learning agent ("hero").
      2. After every hero action, stepping RLCard for each opponent seat
         in turn, injecting actions from the active OpponentPool archetype,
         until it is the hero's turn again.
      3. Exposing set_active_opponent(name) so the training runner can
         rotate the opponent every iteration via the curriculum manager.

    make_env(cfg, opponent_pool=pool) returns a MultiAgentRLCardWrapper;
    make_env(cfg) returns the original RLCardWrapper (self-play, unchanged).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.env.action_mapper import ActionMapper, GameContext, PokerAction

logger = logging.getLogger(__name__)

__all__ = [
    "RLCardWrapper",
    "MultiAgentRLCardWrapper",
    "WrapperConfig",
    "make_env",
]


# =============================================================================
# Card Format Utilities
# =============================================================================

_SUITS: frozenset[str] = frozenset({"S", "H", "D", "C"})
_RANKS: frozenset[str] = frozenset({
    "2", "3", "4", "5", "6", "7", "8", "9",
    "T", "J", "Q", "K", "A",
})

# Mapping: public-card count  →  street index (0=preflop … 3=river)
_PUBLIC_CARDS_TO_STREET: dict[int, int] = {0: 0, 3: 1, 4: 2, 5: 3}


def _to_suitrank(card: str) -> str:
    """Convert card to SUIT+RANK uppercase format (e.g., 'S2', 'HD', 'CA').
    
    features.py _encode_cards() expects SUIT+RANK format where:
    - First character is suit from SUIT_MAP: {'S', 'H', 'D', 'C'}
    - Second character is rank from RANK_MAP: {'2'-'9', 'T', 'J', 'Q', 'K', 'A'}
    """
    card = card.strip().upper()
    if len(card) != 2:
        return card
    c0, c1 = card[0], card[1]
    
    # If already suit+rank format (e.g., 'S2'), keep as is
    if c0 in _SUITS and c1 in _RANKS:
        return card
    
    # If rank+suit format (e.g., '2S'), swap to suit+rank
    if c0 in _RANKS and c1 in _SUITS:
        return c1 + c0
    
    logger.debug("Unrecognised card format '%s'; passing through as-is.", card)
    return card


def _normalise_cards(cards) -> list[str]:
    """Normalize cards to SUIT+RANK uppercase format (e.g., 'S2', 'HD', 'CA').
    
    Handles RLCard Card objects and various string formats.
    Returns standard SUIT+RANK format for features._encode_cards().
    """
    result = []
    for c in cards:
        if not c:
            continue
        
        # Convert to string and uppercase for processing
        card_str = str(c).strip().upper()
        
        # Try standard 2-char format first
        if len(card_str) == 2:
            normalized = _to_suitrank(card_str)
            if normalized and len(normalized) == 2:
                result.append(normalized)
                continue
        
        # Extract suit and rank from any format
        suit = None
        rank = None
        
        # Find suit (single letter)
        for s in _SUITS:
            if s in card_str:
                suit = s
                break
        
        # Find rank (could be 2-char like 10, or 1-char)
        for r in ["10"] + list(_RANKS):
            if r in card_str:
                rank = r
                break
        
        # Construct SUIT+RANK uppercase format
        if suit and rank:
            result.append(suit + rank)
        else:
            # Fallback: just use as-is
            logger.warning("Could not normalize card '%s', using as-is", card_str)
            result.append(card_str)
    
    return result


# =============================================================================
# Wrapper Configuration
# =============================================================================

@dataclass
class WrapperConfig:
    """Configuration for RLCardWrapper, mirroring config.yaml's environment section."""

    num_players: int = 6
    big_blind: float = 2.0
    small_blind: float = 1.0
    initial_stack_bb: float = 200.0
    game_id: str = "no-limit-holdem"

    @property
    def initial_stack(self) -> float:
        return self.initial_stack_bb * self.big_blind

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> WrapperConfig:
        env = cfg.get("environment", {})
        return cls(
            num_players=int(env.get("num_players", 6)),
            big_blind=float(env.get("big_blind", 2)),
            small_blind=float(env.get("small_blind", 1)),
            initial_stack_bb=float(env.get("initial_stack_bb", 200)),
            game_id=str(env.get("game_type", "no-limit-holdem")),
        )


# =============================================================================
# Internal action-index constants (mirrors PokerAction enum in action_mapper.py)
# Must be kept in sync with action_mapper.PokerAction.
# =============================================================================

_FOLD        = 0
_CHECK       = 1      # NEW — separated from CALL for check-raise learning
_CALL        = 2      # NEW — separated from CHECK
_MIN_RAISE   = 3
_RAISE_QUARTER = 4    # 25% pot early position sizing
_RAISE_THIRD = 5      # 33% pot block bet
_RAISE_HALF  = 6
_RAISE_75    = 7
_RAISE_POT   = 8
_RAISE_150   = 9
_RAISE_2X    = 10
_ALL_IN      = 11     # shifted from 10 (checkpoint-breaking)
_N_RAISE_LEVELS: int = _RAISE_2X - _MIN_RAISE  # == 7 (was 6)


# =============================================================================
# RLCard Wrapper
# =============================================================================

class RLCardWrapper:
    """Adapts RLCard's no-limit-holdem to the PokerEnvironment Protocol."""

    def __init__(self, config: WrapperConfig | None = None) -> None:
        try:
            import rlcard  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'rlcard' package is required by RLCardWrapper.\n"
                "Install it with:  pip install 'rlcard>=1.1.0'"
            ) from exc

        self.config: WrapperConfig = config or WrapperConfig()

        rlcard_config: dict[str, Any] = {
            "game_num_players": self.config.num_players,
        }
        try:
            self._env = rlcard.make(self.config.game_id, config=rlcard_config)
        except TypeError:
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

        self._action_mapper: ActionMapper = ActionMapper()

        self._current_player_id: int            = 0
        self._current_state:     dict[str, Any] = {}
        self._hand_start_chips:  list[float]    = []
        self._hand_history:      list[dict]     = []
        self._terminal:          bool           = True
        # [FIX Y-1 COMPLETION] Track current street so history entries carry
        # street context.  0=preflop, 1=flop, 2=turn, 3=river.
        self._current_street:    int            = 0

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
        try:
            raw_result = self._env.reset()
        except Exception as exc:
            raise RuntimeError(f"RLCard env.reset() raised: {exc}") from exc

        state, player_id = self._unpack_reset(raw_result)

        self._current_player_id = player_id
        self._current_state     = state
        self._terminal          = False
        self._hand_history      = []
        # [FIX Y-1 COMPLETION] Reset street at hand start
        self._current_street    = 0

        raw = self._get_raw_obs(state)
        self._hand_start_chips = self._extract_all_chips(raw)

        obs = self._build_obs_dict(state, player_id)

        logger.debug(
            "reset(): player=%d, starting_stacks=%s",
            player_id,
            [f"{c:.0f}" for c in self._hand_start_chips],
        )
        return obs

    def step(self, action: int) -> tuple[dict[str, Any], float]:
        if self._terminal:
            raise RuntimeError(
                "step() called on a completed episode — call reset() first."
            )

        action = int(max(_FOLD, min(_ALL_IN, action)))

        raw = self._get_raw_obs(self._current_state)
        ctx = self._build_game_context(raw, self._current_player_id)

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

        # [FIX Y-1 COMPLETION] Determine street from the CURRENT state
        # (before the step), so the recorded street is the one on which
        # this action was taken.
        n_public = len(raw.get("public_cards", []))
        self._current_street = _PUBLIC_CARDS_TO_STREET.get(n_public, 0)

        # Calculate SPR (Stack-to-Pot Ratio) before the action
        # SPR = effective_stack / pot_before_action
        all_chips = self._extract_all_chips(raw)
        while len(all_chips) < self.config.num_players:
            all_chips.append(self.config.initial_stack)
        
        my_chips = float(all_chips[self._current_player_id])
        opp_chips = float(all_chips[1 - self._current_player_id]) if self.config.num_players == 2 else max(
            all_chips[i] for i in range(len(all_chips)) if i != self._current_player_id
        )
        effective_stack = min(my_chips, opp_chips)
        pot_before = float(raw.get("pot", 0.0))
        spr_before = effective_stack / pot_before if pot_before > 0.0 else 0.0

        self._hand_history.append({
            "action":     action,
            "amount":     chip_amount,
            "player":     self._current_player_id,
            "street":     self._current_street,
            # [RTA-1 FIX] pot_before is the pot size BEFORE this action.
            # Required by ObservationBuilder dim-11 bet-ratio encoding.
            # Computed from raw obs state captured before env.step() is called.
            "pot_before": pot_before,
            # [NEW] SPR (Stack-to-Pot Ratio) before this action.
            # Used for opponent range assessment and bet sizing heuristics.
            "spr_before": spr_before,
        })

        legal = self._current_state.get("legal_actions", {})
        rlcard_id = self._map_our_action_to_rlcard(action, legal, ctx)

        logger.debug(
            "step(): player=%d, our_action=%d → rlcard_id=%d",
            self._current_player_id, action, rlcard_id,
        )

        try:
            raw_result = self._env.step(rlcard_id)
        except Exception as exc:
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
        return self._terminal

    # =========================================================================
    # State Translation
    # =========================================================================

    def _build_obs_dict(
        self,
        state: dict[str, Any],
        player_id: int,
    ) -> dict[str, Any]:
        raw = self._get_raw_obs(state)
        n   = self.config.num_players
        bb  = self.config.big_blind
        pid = int(player_id) % n

        hand         = _normalise_cards(raw.get("hand", []))
        public_cards = _normalise_cards(raw.get("public_cards", []))

        all_chips = self._extract_all_chips(raw)
        while len(all_chips) < n:
            all_chips.append(self.config.initial_stack)

        my_chips: float = float(all_chips[pid])
        opponent_chips: list[float] = [
            float(all_chips[i]) for i in range(n) if i != pid
        ]

        pot_base: float = float(raw.get("pot", 0.0))
        stakes: list[float] = self._extract_stakes(raw, n)
        pot: float = pot_base + sum(stakes)

        my_stake:     float = stakes[pid] if pid < len(stakes) else 0.0
        max_stake:    float = max(stakes) if stakes else 0.0
        amount_to_call: float = max(0.0, max_stake - my_stake)

        raw_min: float = float(raw.get("min_raise", 0.0))
        if raw_min > 0:
            min_raise: float = raw_min
        elif max_stake > 0:
            min_raise = max_stake * 2.0
        else:
            min_raise = bb * 2.0

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
    # Action Mapping
    # =========================================================================

    def _map_our_action_to_rlcard(
        self,
        action: int,
        legal: dict[int, Any],
        ctx: GameContext,
    ) -> int:
        if not legal:
            logger.warning(
                "_map_our_action_to_rlcard: empty legal dict — returning 0."
            )
            return 0

        sorted_ids = sorted(legal.keys())
        n          = len(sorted_ids)

        if action == _FOLD:
            return sorted_ids[0]
        if action == _CHECK:
            # CHECK: find check action (no bet required)
            return sorted_ids[1] if n > 1 else sorted_ids[0]
        if action == _CALL:
            # CALL: find call action (match bet required)
            return sorted_ids[1] if n > 1 else sorted_ids[0]
        if action == _ALL_IN:
            return sorted_ids[-1]

        raise_ids = sorted_ids[2:]
        if not raise_ids:
            return sorted_ids[min(1, n - 1)]

        # Map our raise indices (MIN_RAISE..RAISE_2X inclusive = 3..10) linearly
        # onto RLCard's available raise slots. _ALL_IN (11) is handled above.
        proportion = (action - _MIN_RAISE) / max(_RAISE_2X - _MIN_RAISE, 1)  # 0.0 -> 1.0
        mapped_idx = round(proportion * (len(raise_ids) - 1))
        mapped_idx = max(0, min(mapped_idx, len(raise_ids) - 1))
        return raise_ids[mapped_idx]

    def _rlcard_legal_to_our_mask(
        self,
        rlcard_legal: dict[int, Any],
        my_chips: float,
        amount_to_call: float,
    ) -> list[int]:
        if not rlcard_legal:
            return [_FOLD]

        sorted_ids = sorted(rlcard_legal.keys())
        n_total    = len(sorted_ids)
        n_raises   = max(0, n_total - 2)

        legal: set[int] = set()
        legal.add(_FOLD)
        
        # [PHASE 1] Separate CHECK and CALL for Deep CFR check-raise learning
        if n_total >= 2:
            if amount_to_call <= 0.0:
                legal.add(_CHECK)       # No bet required -> CHECK legal
            else:
                legal.add(_CALL)        # Bet required -> CALL legal
        
        if n_raises >= 1:
            legal.add(_MIN_RAISE)
        if n_raises >= 2:
            legal.add(_RAISE_QUARTER)  # 25% pot early position sizing
        if n_raises >= 2:
            legal.add(_RAISE_THIRD)    # 33% pot block bet
        if n_raises >= 3:
            legal.add(_RAISE_HALF)     # 50% pot
        if n_raises >= 4:
            legal.add(_RAISE_75)
            legal.add(_RAISE_POT)
        if n_raises >= 5:
            legal.add(_RAISE_150)
        if n_raises >= 6:
            legal.add(_RAISE_2X)
        if n_raises >= 1 and my_chips > 0:
            legal.add(_ALL_IN)         # index 11

        return sorted(legal)

    # =========================================================================
    # Reward Computation
    # =========================================================================

    def _compute_terminal_reward(self) -> float:
        bb = self.config.big_blind
        try:
            payoffs = self._env.get_payoffs()
            return float(payoffs[0]) / bb
        except Exception as exc:
            logger.warning(
                "env.get_payoffs() failed (%s); computing reward from chip delta.",
                exc,
            )
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
        n   = self.config.num_players
        bb  = self.config.big_blind
        pid = int(player_id) % n

        all_chips = self._extract_all_chips(raw)
        while len(all_chips) < n:
            all_chips.append(self.config.initial_stack)
        my_chips: float = float(all_chips[pid])

        stakes:   list[float] = self._extract_stakes(raw, n)
        pot_base: float       = float(raw.get("pot", 0.0))
        pot:      float       = pot_base + sum(stakes)

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
        raw = state.get("raw_obs")
        if isinstance(raw, dict):
            return raw
        return state

    def _extract_all_chips(self, raw: dict[str, Any]) -> list[float]:
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

    @staticmethod
    def _unpack_reset(result: Any) -> tuple[dict[str, Any], int]:
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            state, player_id = result[0], result[1]
            state = dict(state) if not isinstance(state, dict) else state
            return state, int(player_id)
        state = dict(result) if not isinstance(result, dict) else result
        return state, 0

    @staticmethod
    def _unpack_step(result: Any) -> tuple[dict[str, Any], int]:
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
        return self._current_player_id

    def get_num_actions(self) -> int:
        return 9

    def get_betting_history(self) -> list[dict[str, Any]]:
        return list(self._hand_history)


# =============================================================================
# [FIX P0-A] MultiAgentRLCardWrapper
# =============================================================================

class MultiAgentRLCardWrapper(RLCardWrapper):
    """RLCard wrapper that injects opponent actions from an OpponentPool.

    [FIX P0-A] Wires the OpponentPool into the training loop.

    The learning agent (hero) permanently occupies seat ``_hero_seat``
    (default: 0). All other seats are controlled by the active archetype
    selected from the configured OpponentPool.

    Design
    ------
    ``step(hero_action)`` first applies the hero's action via the parent
    class, then loops — calling RLCardWrapper.step() with opponent actions
    from the pool — until it is the hero's turn again or the hand ends.
    This means every call to ``step()`` from the collector's perspective
    advances the hand by one *hero decision*, not one raw RLCard tick.

    Opponent actions ARE recorded in ``_hand_history`` (with their correct
    player seat and street) so the hero's betting-history tensor reflects
    the full multi-player action sequence.

    Usage::

        pool = OpponentPool(archetype_names=["calling_station", "maniac"])
        env  = MultiAgentRLCardWrapper(WrapperConfig(), opponent_pool=pool)
        env.set_active_opponent("calling_station")
        obs = env.reset()
        obs, reward = env.step(hero_action)

    The training runner should call ``set_active_opponent`` at the start
    of each iteration after consulting the CurriculumManager.
    """

    def __init__(
        self,
        config: WrapperConfig | None = None,
        opponent_pool: Any | None = None,
        hero_seat: int = 0,
    ) -> None:
        super().__init__(config)
        self._opponent_pool:         Any = opponent_pool
        self._hero_seat:             int = hero_seat
        self._active_opponent_name:  str = "random"

        logger.info(
            "MultiAgentRLCardWrapper: hero_seat=%d, pool=%s",
            hero_seat,
            type(opponent_pool).__name__ if opponent_pool else "None",
        )

    # ------------------------------------------------------------------
    # Opponent selection API
    # ------------------------------------------------------------------

    def set_active_opponent(self, name: str) -> None:
        """Set which archetype from the pool controls non-hero seats."""
        self._active_opponent_name = name
        logger.debug("Active opponent: '%s'", name)

    def get_active_opponent_name(self) -> str:
        return self._active_opponent_name

    # ------------------------------------------------------------------
    # Internal: get one opponent action
    # ------------------------------------------------------------------

    def _get_opponent_action(self, obs: dict[str, Any]) -> int:
        """Return an action for the current non-hero seat.

        Priority: OpponentPool archetype → random fallback.
        """
        legal_actions: list[int] = obs.get("legal_actions", [0, 1])
        if not legal_actions:
            return 0

        if self._opponent_pool is not None:
            agent = self._opponent_pool.get_archetype(self._active_opponent_name)
            if agent is not None:
                try:
                    return int(agent.select_action(legal_actions, obs))
                except Exception as exc:
                    logger.warning(
                        "Opponent '%s' action failed: %s — using random.",
                        self._active_opponent_name, exc,
                    )

        import random as _rnd
        return _rnd.choice(legal_actions)

    # ------------------------------------------------------------------
    # Protocol overrides
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Reset the hand; advance opponents until it is the hero's turn."""
        obs = super().reset()

        # In some RLCard games the hero is not always first to act
        # (e.g. hero has the big blind and SB must post first).
        # Drain those non-hero ticks before returning.
        while not self._terminal and self._current_player_id != self._hero_seat:
            opp_action = self._get_opponent_action(obs)
            obs, _ = RLCardWrapper.step(self, opp_action)

        return obs

    def step(self, hero_action: int) -> tuple[dict[str, Any], float]:
        """Execute the hero's action then advance all opponent seats.

        Returns the observation from the hero's next decision point (or
        the terminal observation if the hand ends mid-opponents' actions).
        The reward is always from the hero's (seat-0) perspective.
        """
        # ── 1. Hero acts ──────────────────────────────────────────────
        obs, reward = RLCardWrapper.step(self, hero_action)

        if self._terminal:
            return obs, reward

        # ── 2. Opponents act until hero's turn or hand ends ───────────
        while not self._terminal and self._current_player_id != self._hero_seat:
            opp_action = self._get_opponent_action(obs)
            obs, opp_reward = RLCardWrapper.step(self, opp_action)
            if self._terminal:
                # The terminal reward is always computed from seat-0's
                # perspective inside _compute_terminal_reward(), so we
                # propagate it regardless of who triggered the terminal.
                reward = opp_reward

        return obs, reward


# =============================================================================
# Factory Function
# =============================================================================

def make_env(
    cfg: dict[str, Any] | WrapperConfig | None = None,
    opponent_pool: Any | None = None,
) -> RLCardWrapper:
    """Create a configured RLCard wrapper, optionally with an opponent pool.

    Args:
        cfg:           Full ``config.yaml`` dict, a ``WrapperConfig`` instance,
                       or ``None`` to use all defaults.
        opponent_pool: When provided, returns a ``MultiAgentRLCardWrapper``
                       that injects opponent actions for non-hero seats.
                       When ``None``, returns a plain ``RLCardWrapper``
                       (self-play, all seats use the learning agent — the
                       original behaviour, unchanged).

    Returns:
        ``MultiAgentRLCardWrapper`` if *opponent_pool* is supplied,
        else ``RLCardWrapper``.

    Raises:
        TypeError: If *cfg* is not ``dict``, ``WrapperConfig``, or ``None``.
    """
    if cfg is None:
        config = WrapperConfig()
    elif isinstance(cfg, WrapperConfig):
        config = cfg
    elif isinstance(cfg, dict):
        config = WrapperConfig.from_dict(cfg)
    else:
        raise TypeError(
            f"cfg must be dict, WrapperConfig, or None; got {type(cfg).__name__}"
        )

    if opponent_pool is not None:
        return MultiAgentRLCardWrapper(config, opponent_pool=opponent_pool)
    return RLCardWrapper(config)
