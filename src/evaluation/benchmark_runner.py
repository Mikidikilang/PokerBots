"""
Slumbot HUNL Benchmark Runner (benchmark_runner.py).

[FIX R-2 — 2025-03-28] Three ACPC protocol bugs fixed:

    BUG 1 — () parsing error:
        MATCHSTATE messages contain NO parentheses. Legal actions are NOT
        embedded in the message string. They must be computed from the
        parsed game state (stacks, pot, call amounts).

    BUG 2 — Missing turn detection:
        In ACPC HU NLHE, the server sends MATCHSTATE only when the client
        must act. We double-check via is_my_turn from the game-state parser.

    BUG 3 — Broken game-state parsing:
        The old _extract_game_state() used a character-by-character loop
        that mis-handled multi-raise streets and didn't track the
        acting-player's turn correctly. Replaced with a complete action-
        history replayer (_parse_game_state_from_acpc) that tracks
        stacks, bets, and the next actor accurately.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING, Any

import torch

from src.env.action_mapper import ActionMapper
from src.env.features import ObservationBuilder, ObservationConfig
from src.evaluation.acpc_client import AcpcClient, MatchState
from src.model.networks import NetworkConfig, PokerActorCritic

logger = logging.getLogger(__name__)


# =============================================================================
# Slumbot Evaluator Configuration
# =============================================================================

@dataclass
class SlumbotEvalConfig:
    """Configuration for Slumbot HUNL evaluation."""

    checkpoint_path: str
    acpc_host:    str   = "slumbot.com"
    acpc_port:    int   = 9000
    stack_size_bb: int  = 200
    big_blind:    float = 2.0
    max_hands:    int   = 100_000
    device:       str   = "cpu"


# =============================================================================
# [FIX R-2] Parsed game state data structure
# =============================================================================

class _ParsedGameState(NamedTuple):
    """Immutable snapshot of game state derived from ACPC action history."""
    my_chips:       float  # Remaining stack for the hero
    opponent_chips: float  # Remaining stack for the villain
    pot:            float  # Total chips committed by both players so far
    amount_to_call: float  # Chips required to call (0 = can check)
    min_raise:      float  # Minimum legal raise amount (absolute chips)
    is_my_turn:     bool   # True if the hero should act next


# =============================================================================
# [FIX R-2] Correct ACPC game-state parser
# =============================================================================

def _parse_game_state_from_acpc(
    action_history: str,
    position: int,
    stack_bb: int,
    big_blind: float,
) -> _ParsedGameState:
    """Reconstruct game state by replaying the ACPC action history.

    ACPC HU NLHE action encoding (streets separated by '/'):
        'f'          — fold
        'c'          — call or check
        'r{amount}'  — raise TO {amount} chips (absolute, not incremental)

    Position convention:
        0 = button / small blind — acts FIRST preflop, SECOND postflop
        1 = big blind            — acts SECOND preflop, FIRST postflop
    """
    initial_stack: float = stack_bb * big_blind
    sb: float = big_blind / 2.0
    bb: float = big_blind

    stacks: list[float] = [initial_stack - sb, initial_stack - bb]
    street_bets: list[float] = [sb, bb]   # Preflop: blinds already posted
    pot: float = sb + bb
    street_idx: int = 0

    streets: list[str] = action_history.split("/") if action_history else [""]

    for s_idx, street_str in enumerate(streets):
        street_idx = s_idx

        if s_idx > 0:
            street_bets = [0.0, 0.0]

        # Preflop: button (pos 0) acts first. Postflop: big blind (pos 1) first.
        acting: int = 0 if s_idx == 0 else 1

        i: int = 0
        last_raise_size: float = bb

        while i < len(street_str):
            ch: str = street_str[i]

            if ch == "f":
                i += 1
                acting = 1 - acting

            elif ch == "c":
                other: int = 1 - acting
                call_delta: float = max(0.0, street_bets[other] - street_bets[acting])
                call_delta = min(call_delta, stacks[acting])
                stacks[acting] -= call_delta
                pot += call_delta
                street_bets[acting] += call_delta
                i += 1
                acting = 1 - acting

            elif ch == "r":
                j: int = i + 1
                while j < len(street_str) and street_str[j].isdigit():
                    j += 1
                raise_to: float = float(street_str[i + 1: j]) if j > i + 1 else 0.0

                prev_commitment: float = street_bets[acting]
                raise_delta: float = min(
                    max(0.0, raise_to - prev_commitment),
                    stacks[acting],
                )

                other = 1 - acting
                last_raise_size = max(raise_to - street_bets[other], bb)

                stacks[acting] -= raise_delta
                pot += raise_delta
                street_bets[acting] = min(
                    raise_to, prev_commitment + stacks[acting] + raise_delta
                )
                i = j
                acting = 1 - acting

            else:
                i += 1

    hero: int = position
    villain: int = 1 - position

    amount_to_call: float = max(0.0, street_bets[villain] - street_bets[hero])

    min_raise_increment: float = max(last_raise_size, bb)
    min_raise: float = amount_to_call + min_raise_increment

    is_my_turn: bool = (acting == position)

    return _ParsedGameState(
        my_chips=stacks[hero],
        opponent_chips=stacks[villain],
        pot=pot,
        amount_to_call=amount_to_call,
        min_raise=min_raise,
        is_my_turn=is_my_turn,
    )


def _compute_legal_actions_from_state(
    game_state: _ParsedGameState,
    big_blind: float,
) -> list[int]:
    """Compute legal 9-action indices from the parsed game state.

    [FIX R-2] Legal actions are computed from game state, NOT parsed from
    the MATCHSTATE string (which contains no legal-action field).
    """
    legal: list[int] = [0, 1]  # Fold and check/call are always legal

    remaining_after_call: float = game_state.my_chips - game_state.amount_to_call

    if remaining_after_call > 0 and game_state.my_chips > 0:
        if game_state.my_chips >= game_state.min_raise:
            legal.append(2)  # min-raise

        # Indices match expanded PokerAction enum (Priority-3 action space fix):
        #   3 = 0.33x (block bet), 4 = 0.50x, 5 = 0.75x, 6 = 1.0x, 7 = 1.5x, 8 = 2.0x
        pot_multipliers: dict[int, float] = {
            3: 0.33, 4: 0.50, 5: 0.75, 6: 1.00, 7: 1.50, 8: 2.00
        }
        for action_idx, mult in pot_multipliers.items():
            raise_size: float = (
                game_state.amount_to_call
                + mult * (game_state.pot + game_state.amount_to_call)
            )
            if game_state.my_chips >= raise_size:
                legal.append(action_idx)

        legal.append(9)  # All-in always available when chips above call (Priority-3 action space fix)

    return sorted(set(legal))


def _action_idx_to_acpc_string(
    action_idx: int,
    game_state: _ParsedGameState,
    big_blind: float,
) -> str:
    """Convert our discrete action index to an ACPC-protocol action string.

    ACPC format:
        'f'          — fold
        'c'          — call or check
        'r{amount}'  — raise TO {amount} (total commitment this street)
    """
    if action_idx == 0:
        return "f"

    if action_idx == 1:
        return "c"

    if action_idx == 9:   # ALL_IN shifted from 8 → 9 (Priority-3 action space fix)
        return f"r{int(game_state.my_chips)}"

    # Index mapping after action space expansion (10 actions, indices 2-8 are raises):
    #   2=min, 3=0.33x, 4=0.50x, 5=0.75x, 6=1.0x, 7=1.5x, 8=2.0x
    pot_multipliers: dict[int, float] = {
        2: 0.0, 3: 0.33, 4: 0.50, 5: 0.75, 6: 1.00, 7: 1.50, 8: 2.00,
    }
    mult: float = pot_multipliers.get(action_idx, 1.0)

    if action_idx == 2:
        raise_amount: float = game_state.min_raise
    else:
        raise_amount = (
            game_state.amount_to_call
            + mult * (game_state.pot + game_state.amount_to_call)
        )

    raise_amount = min(raise_amount, game_state.my_chips)
    raise_amount = max(raise_amount, game_state.min_raise)

    return f"r{int(raise_amount)}"


def _is_terminal_action_history(action_history: str) -> bool:
    """Return True if the action history contains a fold (hand is over)."""
    return "f" in action_history.replace("/", "")


# =============================================================================
# Card Format Conversion
# =============================================================================

def _convert_card_acpc_to_observationbuilder(acpc_card: str) -> str:
    """Convert ACPC 'RankSuit' (e.g., 'As') to ObservationBuilder 'SuitRank' (e.g., 'SA')."""
    if len(acpc_card) != 2:
        raise ValueError(f"Invalid card format: '{acpc_card}'")
    rank_char = acpc_card[0].upper()
    suit_char = acpc_card[1].upper()
    return suit_char + rank_char


# =============================================================================
# Slumbot Evaluator
# =============================================================================

class SlumbotEvaluator:
    """Fixed SlumbotEvaluator — corrected play_hand() and all helper methods."""

    def __init__(self, config: SlumbotEvalConfig, network_config: NetworkConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.network = PokerActorCritic(network_config).to(self.device)
        self._load_checkpoint(config.checkpoint_path)

        obs_config = ObservationConfig(num_players=2)
        self.obs_builder = ObservationBuilder(obs_config)
        self.action_mapper = ActionMapper()

        self.acpc_client = AcpcClient(
            host=config.acpc_host,
            port=config.acpc_port,
        )
        self.acpc_client.handshake()

        self.hands_played     = 0
        self.total_chip_delta = 0.0
        self.hands_won        = 0
        self.hands_lost       = 0

        logger.info(
            "SlumbotEvaluator initialized: %s vs %s:%d, stack=%d BB, device=%s",
            config.checkpoint_path, config.acpc_host, config.acpc_port,
            config.stack_size_bb, config.device,
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        from pathlib import Path
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.network.load_state_dict(state_dict)
            logger.info("Model checkpoint loaded: %s", checkpoint_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    def _matchstate_to_observation(
        self,
        acpc_state: MatchState,
        legal_actions: list[int],
    ) -> dict[str, Any] | None:
        """Convert ACPC MATCHSTATE to 11-key observation format."""
        try:
            hole_cards = [
                _convert_card_acpc_to_observationbuilder(c)
                for c in acpc_state.hole_cards
            ]
            public_cards = [
                _convert_card_acpc_to_observationbuilder(c)
                for c in acpc_state.board_cards
            ]

            big_blind   = self.config.big_blind
            small_blind = big_blind / 2.0

            obs_dict = {
                "hand":            hole_cards,
                "public_cards":    public_cards,
                "pot":             0.0,          # overridden after parsing
                "my_chips":        0.0,          # overridden after parsing
                "opponent_chips":  0.0,          # overridden after parsing
                "big_blind":       float(big_blind),
                "small_blind":     float(small_blind),
                "position":        int(acpc_state.position),
                "betting_history": [],
                "legal_actions":   legal_actions,
                "amount_to_call":  0.0,          # overridden after parsing
                "min_raise":       float(big_blind * 2),
            }
            return obs_dict

        except Exception as e:
            logger.error("Failed to convert MATCHSTATE to observation: %s", e)
            return None

    def _select_action(
        self,
        obs_dict: dict[str, Any],
        legal_actions: list[int],
    ) -> int | None:
        """Select action using model inference."""
        try:
            with torch.inference_mode():
                action_idx, _, _ = self.network.get_action(obs_dict, deterministic=True)

            if action_idx not in legal_actions:
                logger.warning(
                    "Model action %d not legal (%s). Falling back to check/call.",
                    action_idx, legal_actions,
                )
                action_idx = 1 if 1 in legal_actions else legal_actions[0]

            return action_idx

        except Exception as e:
            logger.error("Failed to select action: %s", e)
            return 1 if 1 in legal_actions else legal_actions[0]

    def play_hand(self) -> bool:
        """Play one complete hand against the ACPC server.

        [FIX R-2] Protocol loop:
            1. Receive a line from the server.
            2. If MATCHSTATE:  parse game state, compute legal actions FROM
               game state (NOT from message string), verify our turn, act.
            3. If HAND_RESULT: record chip delta, mark hand complete.
            4. Repeat until hand_complete or error.

        Key corrections:
            • Legal actions are COMPUTED from game state — NOT parsed from
              the MATCHSTATE string (which has no legal-action field).
            • Turn detection uses is_my_turn from the action-history parser.
            • Game state parsing uses _parse_game_state_from_acpc, which
              correctly replays the full ACPC action history.
        """
        hand_complete: bool = False
        action_count:  int  = 0
        MAX_ACTIONS_PER_HAND: int = 200

        while not hand_complete and action_count < MAX_ACTIONS_PER_HAND:
            try:
                raw_line: str = self.acpc_client._recv_line()
            except Exception as recv_exc:
                logger.error("ACPC recv failed: %s", recv_exc)
                return False

            if not raw_line:
                logger.warning("Received empty line from ACPC server")
                return False

            # ── Branch 1: Game state update ─────────────────────────────
            if raw_line.startswith("MATCHSTATE:"):
                acpc_state = self.acpc_client.parse_matchstate(raw_line)
                if acpc_state is None:
                    logger.warning("Failed to parse MATCHSTATE: %s", raw_line[:80])
                    return False

                # Detect terminal state (fold in history)
                if _is_terminal_action_history(acpc_state.action_history):
                    logger.debug("Terminal MATCHSTATE (fold) — waiting for HAND_RESULT")
                    action_count += 1
                    continue

                # ── Parse game state from action history ──────────────────
                game_state: _ParsedGameState = _parse_game_state_from_acpc(
                    action_history=acpc_state.action_history,
                    position=acpc_state.position,
                    stack_bb=self.config.stack_size_bb,
                    big_blind=self.config.big_blind,
                )

                # ── Verify it is our turn ──────────────────────────────────
                if not game_state.is_my_turn:
                    logger.debug("MATCHSTATE received but is_my_turn=False — skipping")
                    action_count += 1
                    continue

                # ── [FIX R-2] Compute legal actions from game state ────────
                # Legal actions are NOT in the MATCHSTATE string.
                legal_actions: list[int] = _compute_legal_actions_from_state(
                    game_state=game_state,
                    big_blind=self.config.big_blind,
                )

                if not legal_actions:
                    logger.error("No legal actions computed — fallback to check/call")
                    legal_actions = [1]

                # ── Build observation dict ─────────────────────────────────
                obs_dict = self._matchstate_to_observation(acpc_state, legal_actions)
                if obs_dict is None:
                    logger.warning("Observation construction failed — folding")
                    self.acpc_client.send_action("f")
                    action_count += 1
                    continue

                # ── Inject computed game state (precise values) ────────────
                obs_dict["pot"]            = game_state.pot
                obs_dict["my_chips"]       = game_state.my_chips
                obs_dict["opponent_chips"] = game_state.opponent_chips
                obs_dict["amount_to_call"] = game_state.amount_to_call
                obs_dict["min_raise"]      = game_state.min_raise
                obs_dict["legal_actions"]  = legal_actions

                # ── Select action via neural network ──────────────────────
                action_idx: int | None = self._select_action(obs_dict, legal_actions)
                if action_idx is None:
                    action_idx = 1 if 1 in legal_actions else legal_actions[0]

                # ── Convert to ACPC protocol string and send ──────────────
                action_str: str = _action_idx_to_acpc_string(
                    action_idx=action_idx,
                    game_state=game_state,
                    big_blind=self.config.big_blind,
                )

                logger.debug(
                    "Hand step: position=%d, action_idx=%d → '%s', "
                    "pot=%.0f, call=%.0f, stacks=(%.0f, %.0f)",
                    acpc_state.position,
                    action_idx,
                    action_str,
                    game_state.pot,
                    game_state.amount_to_call,
                    game_state.my_chips,
                    game_state.opponent_chips,
                )

                try:
                    self.acpc_client.send_action(action_str)
                except Exception as send_exc:
                    logger.error("Failed to send action '%s': %s", action_str, send_exc)
                    return False

                action_count += 1

            # ── Branch 2: Hand result (terminal) ────────────────────────
            elif "HAND_RESULT" in raw_line or "SCORE" in raw_line:
                chip_delta: float = self._parse_hand_result(
                    raw_line, self.config.big_blind
                )
                self.total_chip_delta += chip_delta
                self.hands_played += 1

                if chip_delta > 0:
                    self.hands_won += 1
                elif chip_delta < 0:
                    self.hands_lost += 1

                logger.debug(
                    "Hand complete: delta=%.1f chips (%.3f BB), "
                    "cumulative mbb/h=%.2f over %d hands",
                    chip_delta,
                    chip_delta / self.config.big_blind,
                    self._calculate_mbb_hand(),
                    self.hands_played,
                )
                hand_complete = True

            else:
                logger.debug("Unknown ACPC message (skipping): %s", raw_line[:60])
                action_count += 1

        if action_count >= MAX_ACTIONS_PER_HAND:
            logger.error("Hand aborted: exceeded MAX_ACTIONS_PER_HAND (%d)", MAX_ACTIONS_PER_HAND)
            return False

        return hand_complete

    @staticmethod
    def _parse_hand_result(raw_line: str, big_blind: float) -> float:
        """Parse a HAND_RESULT or SCORE line into hero's chip delta.

        Supported formats:
            HAND_RESULT:hand_num:delta_p0:delta_p1   (Slumbot)
            SCORE:delta_p0:delta_p1:hand_num:cards   (older ACPC)
        """
        tokens: list[str] = re.findall(r"-?\d+(?:\.\d+)?", raw_line)

        if not tokens:
            logger.warning("Could not parse numeric delta from: %s", raw_line[:80])
            return 0.0

        if raw_line.startswith("HAND_RESULT:") and len(tokens) >= 2:
            try:
                return float(tokens[1])
            except (ValueError, IndexError):
                pass

        try:
            return float(tokens[0])
        except ValueError:
            return 0.0

    def _calculate_mbb_hand(self) -> float:
        if self.hands_played == 0:
            return 0.0
        chip_delta_bb = self.total_chip_delta / self.config.big_blind
        return (chip_delta_bb / self.hands_played) * 1000.0

    def run_evaluation(self) -> dict[str, Any]:
        logger.info("Starting Slumbot evaluation: %d hands", self.config.max_hands)
        try:
            for hand_num in range(self.config.max_hands):
                success = self.play_hand()
                if not success:
                    logger.warning("Hand %d failed", hand_num + 1)
                    continue
                if (hand_num + 1) % 1000 == 0:
                    logger.info(
                        "Progress: %d hands played, mbb/hand=%.2f",
                        self.hands_played, self._calculate_mbb_hand(),
                    )
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
        except Exception as e:
            logger.error("Fatal error during evaluation: %s", e)
        finally:
            self.acpc_client.close()

        return self.get_results()

    def get_results(self) -> dict[str, Any]:
        mbb_hand     = self._calculate_mbb_hand()
        win_rate_pct = (self.hands_won / max(self.hands_played, 1)) * 100.0
        return {
            "hands_played":      self.hands_played,
            "total_chip_delta":  self.total_chip_delta,
            "mbb_hand":          mbb_hand,
            "hands_won":         self.hands_won,
            "hands_lost":        self.hands_lost,
            "win_rate_pct":      win_rate_pct,
            "superhuman":        mbb_hand >= 50.0,
        }
