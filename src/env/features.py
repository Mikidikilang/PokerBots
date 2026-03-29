"""
Observation Space Constructor (features.py).

[FIX Y-1 — 2025-03-28] Extended Betting History (obs_dim 281 → 317 for 6-Max).

    BREAKING CHANGE: The observation dimension increases from 281 to 317.
    Existing checkpoints trained with the old 9-dim history are INCOMPATIBLE.
    Training MUST restart from scratch after applying this fix.

    New betting history encoding (11 dims per step, was 9):

        Indices 0-8:  One-hot action type (unchanged)
        Index 9:      Hero indicator (1.0 = this player's action, 0.0 = opponent's)
        Index 10:     Normalized street (0.00=preflop, 0.33=flop, 0.67=turn, 1.00=river)

    WHY these two dimensions matter:
        Index 9 (hero indicator):
            Without it, the agent cannot distinguish:
                [villain raises preflop] [hero 3-bets] [villain folds]
            from:
                [hero raises preflop] [villain 3-bets] [hero folds]
            Both produce the same one-hot sequence but have entirely different
            strategic implications.

        Index 10 (street):
            A raise on the river has far stronger polarization implications
            than a preflop raise. The network needs this context to separate
            "villain raised preflop with a wide range" from
            "villain raised river into the nuts."

    Config update required (config.yaml):
        environment:
          observation_space:
            betting_history_dim: [18, 11]   # was [18, 9]

[FIX H1 — 2025-03-28] Corrigated chip normalization (bounded [0,1]).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Import NUM_ACTIONS dynamically to stay in sync with action_mapper expansions.
# Avoid circular imports: action_mapper does not import from features.
from src.env.action_mapper import NUM_ACTIONS as _ACTION_SPACE_SIZE  # noqa: E402

DECK_SIZE:  int = 52
NUM_SUITS:  int = 4
NUM_RANKS:  int = 13

SUIT_MAP: dict[str, int] = {"S": 0, "H": 1, "D": 2, "C": 3}
RANK_MAP: dict[str, int] = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}

# [FIX H1] Maximum chip value for normalization (5x initial stack)
_CHIP_NORMALIZATION_MAX_MULTIPLIER: float = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# [FIX Y-1] Extended history dimension constants
# ─────────────────────────────────────────────────────────────────────────────

ACTION_DIM_LEGACY: int = 9
"""Original action feature dimension (9 action types only)."""

ACTION_DIM_V2: int = 11
"""Intermediate extended dimension: hero indicator + street (no bet encoding).
Used for loading checkpoints trained before the RTA-1 bet-size fix."""

ACTION_DIM_EXTENDED: int = 13
"""Full extended action feature dimension with player attribution, street, bet size, and SPR.

Layout of the 13 dimensions per betting-history step:

    Indices 0-8:  One-hot encoding of the action type  (9 dims — FOLD, CHECK/CALL, and 7 raise buckets;
                  All-in at index 10 is not one-hot encoded).
    Index 9:      Hero indicator (1.0 = hero's action, 0.0 = opponent's).
    Index 10:     Normalized street (0.00=preflop, 0.33=flop, 0.67=turn, 1.00=river).
    Index 11:     Normalized bet ratio: min(bet_amount / pot_before_action, 3.0) / 3.0
                  Captures bet sizing information critical for RTA opponent modelling.
                  0.0 = no bet/check, ~0.11 = min-raise, ~0.33 = pot-sized, 1.0 = 3x pot+.
                  [RTA-1 FIX — checkpoint-breaking change]
    Index 12:     Normalized SPR (stack-to-pot ratio): min(SPR, 20.0) / 20.0
                  Captures effective stacks relative to pot (0.0 = pushed-in, 1.0 = 20+ BB).
                  Critical for deep-stack vs shallow-stack decision-making.
                  [NEW — checkpoint-breaking change]
"""

_STREET_NORMALIZATION: dict[int, float] = {
    0: 0.00,   # pre-flop
    1: 0.33,   # flop
    2: 0.67,   # turn
    3: 1.00,   # river
}


@dataclass(frozen=True)
class ObservationConfig:
    """Observation Space configuration.

    action_feature_dim valid values:
        9  (ACTION_DIM_LEGACY)   — original, action type only
        11 (ACTION_DIM_V2)       — hero indicator + street (pre-RTA-1 checkpoints)
        12 (ACTION_DIM_EXTENDED) — full: hero indicator + street + bet ratio [old default]
        13 (ACTION_DIM_EXTENDED) — full+SPR: hero + street + bet ratio + SPR [new default]
    """

    num_players:          int   = 6
    max_betting_actions:  int   = 18
    action_feature_dim:   int   = ACTION_DIM_EXTENDED   # 13 (was 12 in v0.4.x)
    initial_stack_bb:     float = 200.0
    normalization_range:  tuple[float, float] = (0.0, 1.0)
    use_extended_history: bool  = True    # enables dims 9, 10, 12
    use_bet_encoding:     bool  = True    # enables dim 11 (bet ratio)

    def __post_init__(self) -> None:
        if not 2 <= self.num_players <= 9:
            raise ValueError(
                f"num_players must be in [2, 9], got {self.num_players}"
            )
        if self.max_betting_actions < 1:
            raise ValueError(
                f"max_betting_actions must be >= 1, got {self.max_betting_actions}"
            )
        if self.action_feature_dim not in (ACTION_DIM_LEGACY, ACTION_DIM_V2, ACTION_DIM_EXTENDED):
            raise ValueError(
                f"action_feature_dim must be {ACTION_DIM_LEGACY} (legacy), "
                f"{ACTION_DIM_V2} (v2 extended), or {ACTION_DIM_EXTENDED} (full), "
                f"got {self.action_feature_dim}"
            )
        logger.debug(
            "ObservationConfig: players=%d, max_betting=%d, "
            "action_dim=%d, extended=%s, bet_encoding=%s",
            self.num_players,
            self.max_betting_actions,
            self.action_feature_dim,
            self.use_extended_history,
            self.use_bet_encoding,
        )


class ObservationBuilder:
    """Converts raw game state to a structured, normalized tensor dict.

    [FIX Y-1] get_observation_dim() and _encode_betting_history() updated
    for the 13-dim extended history format with SPR tracking.

    6-Max, extended history (action_feature_dim=13):
        hole_cards:      52
        community_cards: 52
        env_metrics:      4 + (6-1) = 9
        betting_history: 18 × 13   = 234   (was 216 with dim=12)
        position:         6
        ──────────────────────────────────
        Total:           353                (was 335 with dim=12)

    [FIX H1] Chip normalization bounded to [0, 1] via 5x stack cap.
    """

    # Keys that must be present in every observation dict passed to build().
    # Missing keys in RTA (live screen-scrape / log-parse) produce silent zeros
    # without this check, leading to incorrect fold/call decisions.
    _REQUIRED_KEYS: frozenset[str] = frozenset({
        "hand", "public_cards", "pot", "my_chips", "big_blind",
        "amount_to_call", "position", "legal_actions",
    })

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config: ObservationConfig = config or ObservationConfig()
        self._norm_min: float = self.config.normalization_range[0]
        self._norm_max: float = self.config.normalization_range[1]

        logger.info(
            "ObservationBuilder initialized: %d players, %d max actions, "
            "action_dim=%d (extended=%s, bet_encoding=%s), obs_dim=%d",
            self.config.num_players,
            self.config.max_betting_actions,
            self.config.action_feature_dim,
            self.config.use_extended_history,
            self.config.use_bet_encoding,
            self.get_observation_dim(),
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def build(
        self,
        raw_state: dict[str, Any],
        validate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Build the full observation dict from a raw game state.

        Args:
            raw_state: Raw game state dictionary from environment or RTA parser.
            validate:  If True, raise ValueError if any of _REQUIRED_KEYS is
                       missing. Set to True in RTA/inference paths to catch
                       incomplete screen-scrape results early. [RTA-2 FIX]

        Raises:
            ValueError: validate=True and required keys are absent.
            KeyError:   'hand' key missing (always enforced regardless of validate).
        """
        if isinstance(raw_state, tuple):
            raw_state = raw_state[0]

        state_source = raw_state
        if "raw_obs" in raw_state and isinstance(raw_state["raw_obs"], dict):
            state_source = {**raw_state, **raw_state["raw_obs"]}

        # [RTA-2 FIX] Strict key validation for live online data paths.
        if validate:
            missing = self._REQUIRED_KEYS - set(state_source.keys())
            if missing:
                raise ValueError(
                    f"ObservationBuilder.build(validate=True): required keys "
                    f"missing from state: {sorted(missing)}. "
                    f"Available keys: {sorted(state_source.keys())}. "
                    f"This likely indicates an incomplete RTA scrape result."
                )

        # [FIX Y-1] Hero seat needed for player attribution in history encoding
        hero_seat: int = int(state_source.get("position", 0))

        try:
            observation: dict[str, torch.Tensor] = {
                "hole_cards":      self._encode_cards(state_source["hand"]),
                "community_cards": self._encode_cards(
                    state_source.get("public_cards", [])
                ),
                "env_metrics":     self._encode_env_metrics(state_source),
                "betting_history": self._encode_betting_history(
                    history=state_source.get("betting_history", []),
                    hero_seat=hero_seat,   # [FIX Y-1] pass hero seat
                ),
                "position":        self._encode_position(
                    state_source.get("position", 0)
                ),
                "action_mask":     self._encode_action_mask(
                    state_source.get("legal_actions", list(range(9)))
                ),
            }
        except KeyError as exc:
            logger.error(
                "Missing key in state source: %s. Available: %s",
                exc, list(state_source.keys()),
            )
            raise

        return observation

    def flatten(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        components: list[torch.Tensor] = [
            observation["hole_cards"],
            observation["community_cards"],
            observation["env_metrics"],
            observation["betting_history"].flatten(),
            observation["position"],
        ]
        return torch.cat(components, dim=0)

    def get_observation_dim(self) -> int:
        """Compute the flat observation dimension from current config.

        With action_feature_dim=13 (default, 6-Max):
            hole_cards:       52
            community_cards:  52
            env_metrics:       4 + (num_players - 1)  = 9
            betting_history:  18 × 13                  = 234  (was 216 with dim=12)
            position:          6
            ──────────────────────────────────────────────────
            Total:            353                       (was 335 with dim=12)
        """
        card_dim:     int = DECK_SIZE * 2
        metrics_dim:  int = 4 + (self.config.num_players - 1)
        history_dim:  int = (
            self.config.max_betting_actions * self.config.action_feature_dim
        )
        position_dim: int = self.config.num_players
        return card_dim + metrics_dim + history_dim + position_dim

    # =========================================================================
    # Private Encoders
    # =========================================================================

    def _encode_cards(self, cards: list[str]) -> torch.Tensor:
        """Multi-hot encode a list of cards into a 52-dim binary vector."""
        encoding: torch.Tensor = torch.zeros(DECK_SIZE, dtype=torch.float32)

        for card_str in cards:
            card_str = card_str.strip().upper()
            if len(card_str) != 2:
                raise ValueError(f"Invalid card format: '{card_str}'")

            suit_char: str = card_str[0]
            rank_char: str = card_str[1]

            if rank_char not in RANK_MAP:
                raise ValueError(f"Unknown rank: '{rank_char}' in '{card_str}'")
            if suit_char not in SUIT_MAP:
                raise ValueError(f"Unknown suit: '{suit_char}' in '{card_str}'")

            rank_idx: int = RANK_MAP[rank_char]
            suit_idx: int = SUIT_MAP[suit_char]
            card_index: int = rank_idx * NUM_SUITS + suit_idx
            encoding[card_index] = 1.0

        return encoding

    def _encode_env_metrics(self, raw_state: dict[str, Any]) -> torch.Tensor:
        """Encode environment metrics as a normalized float vector.

        [FIX H1] Chip normalization is bounded to [0, 1] via 5x stack cap,
        replacing the previous unbounded log-scale approach.
        """
        big_blind: float = float(raw_state.get("big_blind", 2.0))
        if big_blind <= 0:
            logger.warning("big_blind=%.2f not positive. Using 2.0.", big_blind)
            big_blind = 2.0

        initial_stack: float = self.config.initial_stack_bb * big_blind
        max_chip_value: float = initial_stack * _CHIP_NORMALIZATION_MAX_MULTIPLIER

        def _normalize_chips(value: float) -> float:
            if initial_stack <= 0:
                return 0.0
            capped: float = min(float(value), max_chip_value)
            normalized: float = capped / max_chip_value
            return float(max(0.0, min(1.0, normalized)))

        pot:            float = float(raw_state.get("pot", 0.0))
        my_chips:       float = float(raw_state.get("my_chips", 0.0))
        amount_to_call: float = float(raw_state.get("amount_to_call", 0.0))
        min_raise:      float = float(raw_state.get("min_raise", big_blind))

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

        return torch.tensor(metrics, dtype=torch.float32)

    def _encode_betting_history(
        self,
        history:   list[dict[str, Any]],
        hero_seat: int = 0,
    ) -> torch.Tensor:
        """Encode the hand's betting history into a fixed-size 2D tensor.

        [FIX Y-1] When use_extended_history=True (action_feature_dim=13):
            - Columns 0-8: one-hot action type
            - Column 9:    hero indicator (1.0 = hero's action, 0.0 = opponent's)
            - Column 10:   normalized street (0.0=preflop, 0.33=flop, etc.)
            - Column 11:   normalized bet ratio (bet / pot, capped at 3x)
            - Column 12:   normalized SPR (stack / pot, capped at 20)

        When use_extended_history=False (action_feature_dim=9, legacy):
            - Columns 0-8 only — backward compatible with old checkpoints.

        The "player", "street", "pot_before", and "spr_before" keys are emitted 
        by the updated RLCardWrapper.step() method. Old wrappers that don't emit 
        them produce correct but partial encodings (missing dimensions = 0.0).

        Args:
            history:   List of action dicts.
                       Each dict may contain:
                           "action": int  — action index (0-8)
                           "player": int  — seat index of acting player
                                           (optional, defaults to -1)
                           "street": int  — street index (optional, defaults 0)
            hero_seat: Seat index of the current acting player (the "hero").
        """
        max_actions: int  = self.config.max_betting_actions
        action_dim:  int  = self.config.action_feature_dim
        extended:    bool = self.config.use_extended_history

        history_tensor: torch.Tensor = torch.zeros(
            (max_actions, action_dim),
            dtype=torch.float32,
        )

        for step_idx, step in enumerate(history[:max_actions]):
            # ── Dimensions 0-8: one-hot action type ──────────────────────
            action_idx: int = int(step.get("action", 0))
            if 0 <= action_idx < 9:
                history_tensor[step_idx, action_idx] = 1.0

            if not extended:
                continue  # Legacy mode: only action-type encoding

            # ── Dimension 9: hero indicator ───────────────────────────────
            actor_seat: int = int(step.get("player", -1))
            if actor_seat == -1:
                # "player" field absent (old wrapper). Default to 0.0.
                # This degrades gracefully: one-hot info still present.
                history_tensor[step_idx, 9] = 0.0
            else:
                history_tensor[step_idx, 9] = (
                    1.0 if actor_seat == hero_seat else 0.0
                )

            # ── Dimension 10: normalized street ───────────────────────────
            street: int = int(step.get("street", 0))
            street = max(0, min(street, 3))  # clamp to [0, 3]
            history_tensor[step_idx, 10] = _STREET_NORMALIZATION[street]

            # ── Dimension 11: normalized bet ratio ────────────────────────
            # [RTA-1 FIX] Encodes how large the bet was relative to the pot.
            # Critical for RTA villain range-reading: 20% c-bet ≠ 90% c-bet.
            # Formula: min(bet_amount / pot_before_action, 3.0) / 3.0 → [0, 1]
            # Falls back to 0.0 if fields absent (old wrappers without pot_before).
            if action_dim >= 12 and self.config.use_bet_encoding:
                pot_before = float(step.get("pot_before", 0.0))
                bet_amount = float(step.get("amount", 0.0))
                if pot_before > 0.0:
                    normalized_bet = min(bet_amount / pot_before, 3.0) / 3.0
                else:
                    # pot_before absent (legacy wrapper) — degrade gracefully
                    normalized_bet = 0.0
                history_tensor[step_idx, 11] = normalized_bet

            # ── Dimension 12: normalized SPR (Stack-to-Pot Ratio) ─────────
            # [NEW] Encodes effective stack relative to pot at decision point.
            # SPR = effective_stack / pot_before_action
            # Normalized: min(SPR, 20.0) / 20.0 → [0, 1]
            # Critical for deep-stack vs pushed-in decision making.
            if action_dim >= 13:
                spr_before = float(step.get("spr_before", 0.0))
                normalized_spr = min(spr_before, 20.0) / 20.0
                history_tensor[step_idx, 12] = normalized_spr

        if logger.isEnabledFor(logging.DEBUG):
            hero_actions: int = (
                int(history_tensor[:, 9].sum().item()) if extended else 0
            )
            total_steps: int = min(len(history), max_actions)
            logger.debug(
                "Betting history encoded: steps=%d/%d, hero_actions=%d, "
                "action_dim=%d, extended=%s, bet_encoded=%s",
                total_steps, max_actions, hero_actions, action_dim, extended,
                (action_dim >= 12 and self.config.use_bet_encoding),
            )

        return history_tensor

    def _encode_position(self, position_index: int) -> torch.Tensor:
        num_positions: int = self.config.num_players
        position_vector: torch.Tensor = torch.zeros(num_positions, dtype=torch.float32)

        if 0 <= position_index < num_positions:
            position_vector[position_index] = 1.0
        else:
            logger.warning(
                "Invalid position index: %d (max: %d). Zero vector returned.",
                position_index, num_positions - 1,
            )

        return position_vector

    def _encode_action_mask(self, legal_actions: list[int | Any]) -> torch.Tensor:
        # Use dynamic NUM_ACTIONS from action_mapper (currently 10 after expansion)
        num_actions: int = _ACTION_SPACE_SIZE
        mask: torch.Tensor = torch.zeros(num_actions, dtype=torch.float32)

        for action_item in legal_actions:
            try:
                if hasattr(action_item, "value"):
                    action_idx: int = int(action_item.value)
                else:
                    action_idx = int(action_item)

                if 0 <= action_idx < num_actions:
                    mask[action_idx] = 1.0
            except (ValueError, TypeError, AttributeError) as exc:
                logger.warning("Action item not convertible to int: %r (%s)", action_item, exc)

        active_count: int = int(mask.sum().item())
        if active_count == 0:
            logger.error(
                "CRITICAL: empty action mask! Enabling Fold (index 0) as fallback."
            )
            mask[0] = 1.0

        return mask

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def card_str_to_index(card_str: str) -> int:
        card_str = card_str.strip().upper()
        if len(card_str) != 2:
            raise ValueError(f"Invalid card format: '{card_str}'")
        suit_idx: int = SUIT_MAP[card_str[0]]
        rank_idx: int = RANK_MAP[card_str[1]]
        return rank_idx * NUM_SUITS + suit_idx

    @staticmethod
    def index_to_card_str(index: int) -> str:
        if not 0 <= index < DECK_SIZE:
            raise ValueError(f"Card index {index} out of range [0, 51].")
        rank_idx:  int = index // NUM_SUITS
        suit_idx:  int = index % NUM_SUITS
        rank_chars: str = "23456789TJQKA"
        suit_chars: str = "SHDC"
        return suit_chars[suit_idx] + rank_chars[rank_idx]
