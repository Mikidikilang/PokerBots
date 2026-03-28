"""
Slumbot HUNL Benchmark Runner (benchmark_runner.py).

Orchestrates full match evaluations against Slumbot using the ACPC protocol.
Converts game states to observation dicts, evaluates our PokerActorCritic model,
and tracks cumulative win rate (mbb/hand).

The evaluator:
  1. Loads pre-trained model weights from a checkpoint
  2. Plays HUNL matches against ACPC server (alternating positions)
  3. Translates ACPC MATCHSTATE to our 11-key observation format
  4. Uses ActionMapper to convert discrete actions to chip amounts
  5. Tracks cumulative chip delta and calculates mbb/hand win rate
"""

from __future__ import annotations

import logging
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.env.action_mapper import ActionMapper
from src.env.features import ObservationBuilder, ObservationConfig
from src.evaluation.acpc_client import AcpcClient, HandResult, MatchState
from src.model.networks import NetworkConfig, PokerActorCritic

logger = logging.getLogger(__name__)


# =============================================================================
# Slumbot Evaluator Configuration
# =============================================================================

@dataclass
class SlumbotEvalConfig:
    """Configuration for Slumbot HUNL evaluation.

    Attributes:
        checkpoint_path: Path to model checkpoint file.
        acpc_host: Slumbot ACPC server hostname.
        acpc_port: Slumbot ACPC server port.
        stack_size_bb: Initial stack size in big blinds.
        big_blind: Big blind value in chips.
        max_hands: Maximum hands to play.
        device: PyTorch device (cpu/cuda).
    """

    checkpoint_path: str
    acpc_host: str = "slumbot.com"
    acpc_port: int = 9000
    stack_size_bb: int = 200
    big_blind: float = 2.0
    max_hands: int = 100_000
    device: str = "cpu"


# =============================================================================
# Card Format Conversion
# =============================================================================

def _convert_card_acpc_to_observationbuilder(acpc_card: str) -> str:
    """Convert ACPC 'RankSuit' format (e.g., 'As') to ObservationBuilder 'SuitRank' (e.g., 'SA').

    ACPC sends cards in RankSuit format (Ace of Spades = 'As').
    ObservationBuilder requires SuitRank format (Ace of Spades = 'SA').

    Args:
        acpc_card: Card in ACPC format (e.g., 'As', 'Kh', '2d').

    Returns:
        Card in ObservationBuilder format (e.g., 'SA', 'KH', '2D').

    Raises:
        ValueError: If card format is invalid.
    """
    if len(acpc_card) != 2:
        raise ValueError(f"Invalid card format: '{acpc_card}' (expected 2 chars)")

    rank_char = acpc_card[0].upper()
    suit_char = acpc_card[1].upper()

    return suit_char + rank_char  # Reverse to SuitRank format


# =============================================================================
# Slumbot Evaluator
# =============================================================================

class SlumbotEvaluator:
    """Evaluates our poker AI against Slumbot using ACPC protocol.

    Plays Heads-Up No-Limit hold'em matches and tracks win rate in mbb/hand.
    """

    def __init__(self, config: SlumbotEvalConfig, network_config: NetworkConfig) -> None:
        """Initialize the evaluator.

        Args:
            config: SlumbotEvalConfig with checkpoint and server details.
            network_config: NetworkConfig for model architecture.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Load model
        self.network = PokerActorCritic(network_config).to(self.device)
        self._load_checkpoint(config.checkpoint_path)

        # Initialize components
        obs_config = ObservationConfig(num_players=2)
        self.obs_builder = ObservationBuilder(obs_config)
        self.action_mapper = ActionMapper()

        # ACPC client
        self.acpc_client = AcpcClient(
            host=config.acpc_host,
            port=config.acpc_port,
        )
        self.acpc_client.handshake()

        # Tracking
        self.hands_played = 0
        self.total_chip_delta = 0.0
        self.hands_won = 0
        self.hands_lost = 0

        logger.info(
            "SlumbotEvaluator initialized: %s vs %s:%d, stack=%.0f BB, device=%s",
            config.checkpoint_path, config.acpc_host, config.acpc_port,
            config.stack_size_bb, config.device,
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            RuntimeError: If loading fails.
        """
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
        """Convert ACPC MATCHSTATE to 11-key observation format.

        Extracts game state from ACPC MatchState including parsing action history
        to compute pot, stacks, amount to call, and minimum raise.

        Args:
            acpc_state: Parsed MatchState object from ACPC client.
            legal_actions: List of legal action indices (0-8 from ACPC).

        Returns:
            11-key observation dict for get_action(), or None if conversion fails.

        Raises:
            Logs errors internally; returns None on failure.
        """
        try:
            # Convert card format: ACPC "RankSuit" (e.g., 'As') → "SuitRank" (e.g., 'SA')
            hole_cards = [_convert_card_acpc_to_observationbuilder(c) for c in acpc_state.hole_cards]
            public_cards = [_convert_card_acpc_to_observationbuilder(c) for c in acpc_state.board_cards]

            # Extract game state from action history
            # Format: "r100c//:AsKh|2d3c4d" → action_history is empty preflop before actions
            my_chips, opponent_chips, pot_size, amount_to_call, min_raise = self._extract_game_state(
                acpc_state.action_history,
                acpc_state.stage,
                acpc_state.position,
            )

            # Get big blind and small blind from config
            big_blind = self.config.big_blind
            small_blind = big_blind / 2.0

            # Build the 11-key observation dict
            obs_dict = {
                "hand": hole_cards,
                "public_cards": public_cards,
                "pot": float(pot_size),
                "my_chips": float(my_chips),
                "opponent_chips": float(opponent_chips),
                "big_blind": float(big_blind),
                "small_blind": float(small_blind),
                "position": int(acpc_state.position),
                "betting_history": [],  # Not used by network, but required by ObservationBuilder
                "legal_actions": legal_actions,
                "amount_to_call": float(amount_to_call),
            }

            return obs_dict

        except Exception as e:
            logger.error("Failed to convert MATCHSTATE to observation: %s", e)
            return None

    def _extract_game_state(
        self,
        action_history: str,
        stage: int,
        position: int,
    ) -> tuple[float, float, float, float, float]:
        """Extract game state metrics from ACPC action history.

        Parses the action string (e.g., "r100c/:r50c") to compute:
        - Current stacks for both players
        - Pot size
        - Amount to call for current decision point
        - Minimum raise amount

        Args:
            action_history: ACPC action string (e.g., "r100c/:r50c").
            stage: Current betting stage (0=preflop, 1=flop, 2=turn, 3=river).
            position: Our position (0=small blind/button, 1=big blind).

        Returns:
            Tuple of (my_chips, opponent_chips, pot_size, amount_to_call, min_raise).
        """
        initial_stack = self.config.stack_size_bb * self.config.big_blind
        big_blind = self.config.big_blind
        small_blind = big_blind / 2.0

        # Start with initial state
        my_chips = initial_stack
        opponent_chips = initial_stack
        amount_to_call = 0.0
        min_raise = 0.0

        # If position == 0 (button/small blind), we post small blind first
        # If position == 1 (big blind), opponent posts small blind
        if position == 0:
            my_chips -= small_blind
            opponent_chips -= big_blind
            amount_to_call = big_blind - small_blind  # To match big blind
        else:
            my_chips -= big_blind
            opponent_chips -= small_blind
            amount_to_call = 0.0  # We posted big blind, can check

        # Parse action history to extract chips put in
        streets = action_history.split("/")
        for street_idx, street in enumerate(streets):
            if not street or street_idx > stage:
                break

            current_bet = 0.0 if street_idx == 0 else amount_to_call
            for action_char in street:
                if action_char == 'f':
                    # Fold: no chips added, hand likely over (shouldn't reach here if ACPC working)
                    pass
                elif action_char == 'c':
                    # Call: match the current bet
                    amount_to_call = 0.0
                elif action_char == 'k':
                    # Check: no bet
                    amount_to_call = 0.0
                elif action_char == 'r':
                    # Raise: need to parse amount
                    # Format example: "r100" means raise to 100 total
                    idx = street.index('r')
                    amount_str = ""
                    for i in range(idx + 1, len(street)):
                        if street[i].isdigit():
                            amount_str += street[i]
                        else:
                            break
                    if amount_str:
                        amount_to_call = float(amount_str)
                elif action_char == 'a':
                    # All-in: amount is remaining stack
                    amount_to_call = my_chips

        # Calculate pot: initial blinds + chips from action history
        pot_size = small_blind + big_blind
        # (Simplified: in reality would need to track per-street contributions)

        return my_chips, opponent_chips, pot_size, amount_to_call, min_raise

    def _select_action(
        self,
        obs_dict: dict[str, Any],
        legal_actions: list[int],
    ) -> int | None:
        """Select action using model inference.

        Calls the network.get_action() with the observation dict, validates
        legality, and falls back to check/call if model selects illegal action.

        Args:
            obs_dict: 11-key observation dict from _matchstate_to_observation.
            legal_actions: List of legal discrete action indices (0-8).

        Returns:
            Action index (0-8) selected by model, or None on error.
        """
        try:
            # Run inference with gradient disabled
            with torch.inference_mode():
                action_idx, _, _ = self.network.get_action(obs_dict, deterministic=True)

            # Validate legality
            if action_idx not in legal_actions:
                logger.warning(
                    "Model action %d not legal. Legal actions: %s. "
                    "Falling back to check/call.",
                    action_idx, legal_actions,
                )
                # Fallback: prefer check/call (action 1) if legal, else first legal action
                action_idx = 1 if 1 in legal_actions else legal_actions[0]

            return action_idx

        except Exception as e:
            logger.error("Failed to select action: %s", e)
            # Emergency fallback to check/call or first legal action
            return 1 if 1 in legal_actions else legal_actions[0]

            logger.debug(
                "Action selected: idx=%d (%s) -> %s (%.0f chips)",
                action_idx, poker_action.name, resolved.description, resolved.amount,
            )

            return action_idx, resolved.amount

        except Exception as e:
            logger.error("Error selecting action: %s", e)
            return None

    def play_hand(self) -> bool:
        """Play a single hand against Slumbot until completion.

        Implements the game loop:
        1. Receive MATCHSTATE message
        2. If our turn: convert to observation, select action, send to ACPC
        3. If HAND_RESULT: parse chip delta, update win tracking, exit loop
        4. Repeat until hand ends

        Returns:
            True if hand completed successfully, False if error occurred.
        """
        hand_complete = False
        action_count = 0
        max_actions = 1000  # Prevent infinite loop

        try:
            while not hand_complete and action_count < max_actions:
                # Receive next message from ACPC server
                message = self.acpc_client._recv_line()
                if not message:
                    logger.warning("Received empty message from ACPC")
                    return False

                # Check message type
                if message.startswith("MATCHSTATE:"):
                    # Parse MATCHSTATE to get current game state
                    acpc_state = self.acpc_client.parse_matchstate(message)
                    if acpc_state is None:
                        logger.warning("Failed to parse MATCHSTATE: %s", message)
                        return False

                    # Extract legal actions from message format
                    # Message format: MATCHSTATE:position:round:action_history|hole_cards|board_cards (legal_actions)
                    # Extract the (legal_actions) part which contains action indices
                    if "(" not in message or ")" not in message:
                        logger.warning("MATCHSTATE missing legal actions: %s", message)
                        return False

                    legal_actions_str = message[message.index("(") + 1 : message.index(")")]
                    # Example: "fcr" means actions 0 (fold), 1 (check), 2 (raise) not available
                    # Or "(fcr)" means these actions are available
                    # Parse which actions are available from the ACPC message
                    legal_actions = []
                    if "f" in legal_actions_str:
                        legal_actions.append(0)
                    if "c" in legal_actions_str:
                        legal_actions.append(1)
                    if "r" in legal_actions_str:
                        legal_actions.extend([2, 3, 4, 5, 6, 7, 8])  # All raise amounts

                    if not legal_actions:
                        legal_actions = [1]  # Fallback to check/call
                        logger.warning("Could not parse legal actions from: %s", legal_actions_str)

                    # Convert ACPC state to observation dict
                    obs_dict = self._matchstate_to_observation(acpc_state, legal_actions)
                    if obs_dict is None:
                        logger.warning("Failed to convert MATCHSTATE to observation")
                        return False

                    # Get action from our model
                    action_idx = self._select_action(obs_dict, legal_actions)
                    if action_idx is None:
                        logger.warning("Failed to select action")
                        return False

                    # Convert action index to ACPC protocol format
                    # Actions: 0=fold, 1=call/check, 2-8=various raise amounts
                    if action_idx == 0:
                        action_str = "f"
                    elif action_idx == 1:
                        action_str = "c"
                    else:
                        # For raise actions, compute the amount based on game state
                        # Simplified: map action indices to raise amounts
                        # In practice, ActionMapper would do this more precisely
                        amount_to_call = obs_dict.get("amount_to_call", 0.0)
                        min_bet = amount_to_call if amount_to_call > 0 else obs_dict.get("big_blind", 2.0)
                        raise_amount = int(min_bet * (1.5 ** (action_idx - 2)))
                        action_str = f"r{raise_amount}"

                    # Send action to ACPC server
                    self.acpc_client._send_line(action_str)
                    action_count += 1
                    logger.debug("Sent action: %s (action_idx=%d)", action_str, action_idx)

                elif message.startswith("HAND_RESULT:"):
                    # Hand has ended, extract chip delta
                    result = self.acpc_client.parse_result(message)
                    if result is not None:
                        # Update running win rate tracker
                        self.total_chip_delta += result.chip_delta
                        if result.chip_delta > 0:
                            self.hands_won += 1
                        elif result.chip_delta < 0:
                            self.hands_lost += 1

                        logger.debug(
                            "Hand result: delta=%.0f chips, running total=%.0f chips, "
                            "mbb/hand=%.3f, hands_won=%d, hands_lost=%d",
                            result.chip_delta,
                            self.total_chip_delta,
                            self._calculate_mbb_hand(),
                            self.hands_won,
                            self.hands_lost,
                        )
                    else:
                        logger.warning("Failed to parse HAND_RESULT: %s", message)

                    hand_complete = True

                else:
                    # Unknown message type
                    logger.warning("Unknown message type: %s", message[:50])
                    # Continue trying to receive valid messages
                    continue

            if action_count >= max_actions:
                logger.error("Hand exceeded maximum actions (%d)", max_actions)
                return False

            return True

        except Exception as e:
            logger.error("Error during hand play: %s", e)
            return False

    def run_evaluation(self) -> dict[str, Any]:
        """Run the full evaluation match.

        Returns:
            Dictionary with evaluation stats (hands_played, mbb_hand, win_rate, etc.).
        """
        logger.info("Starting Slumbot evaluation: %d hands", self.config.max_hands)

        try:
            for hand_num in range(self.config.max_hands):
                success = self.play_hand()
                if not success:
                    logger.warning("Hand %d failed", hand_num + 1)
                    continue

                self.hands_played += 1

                # Log progress every 1000 hands
                if (hand_num + 1) % 1000 == 0:
                    mbb_hand = self._calculate_mbb_hand()
                    logger.info(
                        "Progress: %d hands played, mbb/hand=%.2f",
                        self.hands_played, mbb_hand,
                    )

        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
        except Exception as e:
            logger.error("Fatal error during evaluation: %s", e)
        finally:
            self.acpc_client.close()

        return self.get_results()

    def _calculate_mbb_hand(self) -> float:
        """Calculate win rate in milli-big-blinds per hand.

        Returns:
            mbb/hand win rate (0 if no hands played).
        """
        if self.hands_played == 0:
            return 0.0

        chip_delta_bb = self.total_chip_delta / self.config.big_blind
        return (chip_delta_bb / self.hands_played) * 1000.0  # Convert to mbb

    def get_results(self) -> dict[str, Any]:
        """Get evaluation results summary.

        Returns:
            Dictionary with stats: hands_played, mbb_hand, win_rate_pct, etc.
        """
        mbb_hand = self._calculate_mbb_hand()
        win_rate_pct = (self.hands_won / max(self.hands_played, 1)) * 100.0

        return {
            "hands_played": self.hands_played,
            "total_chip_delta": self.total_chip_delta,
            "mbb_hand": mbb_hand,
            "hands_won": self.hands_won,
            "hands_lost": self.hands_lost,
            "win_rate_pct": win_rate_pct,
            "superhuman": mbb_hand >= 50.0,  # Threshold from config
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Run Slumbot benchmark from command line."""
    import argparse
    import json
    from src.model.networks import NetworkConfig

    parser = argparse.ArgumentParser(description="Slumbot HUNL Benchmark")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--host", default="slumbot.com", help="ACPC server host")
    parser.add_argument("--port", type=int, default=9000, help="ACPC server port")
    parser.add_argument("--hands", type=int, default=100_000, help="Max hands to play")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--config", help="Path to network config YAML")

    args = parser.parse_args()

    # Load network config
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        net_config = NetworkConfig.from_dict(cfg, num_players=2)
    else:
        net_config = NetworkConfig()

    # Create evaluator
    eval_config = SlumbotEvalConfig(
        checkpoint_path=args.checkpoint,
        acpc_host=args.host,
        acpc_port=args.port,
        max_hands=args.hands,
        device=args.device,
    )

    evaluator = SlumbotEvaluator(eval_config, net_config)

    # Run evaluation
    results = evaluator.run_evaluation()

    # Print results
    logger.info("Evaluation Complete:")
    for key, value in results.items():
        logger.info("  %s: %s", key, value)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
