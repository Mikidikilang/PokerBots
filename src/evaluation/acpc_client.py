"""
ACPC (Annual Computer Poker Competition) Protocol Client (acpc_client.py).

A TCP socket-based client implementation for the ACPC poker protocol, which is
used by Slumbot and other AI poker benchmarks for standardized evaluation.

The ACPC protocol is a text-based line-oriented communication format:
  1. Server sends: "VERSION:2.0.0" (or variant)
  2. Client responds: "VERSION:2.0.0"
  3. Server sends: Match configuration (game name, stack sizes, etc.)
  4. Game starts: Server sends MATCHSTATE messages, Client responds with actions

MATCHSTATE Format (example):
    MATCHSTATE:0:0:r100c//:Kh9h|2d3c4d5s6h

Where:
    - Position (0 or 1 for HUNL)
    - Game state number
    - Board stage and action history
    - Public cards and hole cards

Actions:
    - 'f' : Fold
    - 'c' : Check or Call
    - 'r{amount}' : Raise to {amount}

Robustness:
    - Automatic reconnection with exponential backoff
    - Socket timeouts to prevent hanging
    - Graceful handling of malformed messages
    - Hand-level error recovery
"""

from __future__ import annotations

import logging
import re
import socket
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# ACPC Message Data Structures
# =============================================================================

@dataclass
class MatchState:
    """Parsed ACPC MATCHSTATE message.

    Attributes:
        state_id: Unique state identifier.
        position: Player position (0 or 1 for HUNL).
        stage: Game stage (0=preflop, 1=flop, 2=turn, 3=river).
        board_cards: Community cards as list of card strings in RankSuit format.
        hole_cards: Player's hole cards as list of 2 card strings (RankSuit: As, Kh).
        action_history: String of past actions (e.g., "r100c").
    """

    state_id: int
    position: int
    stage: int
    board_cards: list[str]
    hole_cards: list[str]
    action_history: str


@dataclass
class HandResult:
    """Final result of a hand.

    Attributes:
        chip_delta: Chips won (+) or lost (-) from our perspective.
        reason: How the hand ended (fold, showdown, etc.).
    """

    chip_delta: float
    reason: str

    Manages a persistent connection to an ACPC server (Slumbot), handles
    the handshake, parses game states, and sends actions.

    Robust error handling includes:
      - Automatic reconnection with exponential backoff
      - Socket timeouts
      - Graceful handling of malformed messages
      - Per-hand error recovery
    """

    def __init__(
        self,
        host: str,
        port: int,
        max_retry_attempts: int = 5,
        base_retry_delay_seconds: float = 1.0,
        socket_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the ACPC client.

        Args:
            host: Server hostname or IP address.
            port: Server port number.
            max_retry_attempts: Maximum number of reconnection attempts.
            base_retry_delay_seconds: Base delay for exponential backoff.
            socket_timeout_seconds: Socket timeout in seconds.
        """
        self.host: str = host
        self.port: int = port
        self.max_retry_attempts: int = max_retry_attempts
        self.base_retry_delay: float = base_retry_delay_seconds
        self.socket_timeout: float = socket_timeout_seconds

        self.socket: socket.socket | None = None
        self._buffer: bytes = b""  # Leftover data from previous recv()
        self._ensure_connected()

        logger.info(
            "AcpcClient initialized: %s:%d, timeout=%.1fs, max_retries=%d",
            host, port, socket_timeout_seconds, max_retry_attempts,
        )

    def _ensure_connected(self) -> None:
        """Ensure socket is connected with exponential backoff retry."""
        for attempt in range(self.max_retry_attempts):
            try:
                if self.socket is not None:
                    try:
                        self.socket.close()
                    except Exception:
                        pass

                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.socket_timeout)
                self.socket.connect((self.host, self.port))
                self._buffer = b""
                logger.info("AcpcClient connected to %s:%d", self.host, self.port)
                return

            except Exception as e:
                delay = self.base_retry_delay * (2 ** attempt)
                logger.warning(
                    "Connection attempt %d/%d failed: %s. "
                    "Retrying in %.1fs...",
                    attempt + 1, self.max_retry_attempts, e, delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Failed to connect to {self.host}:{self.port} "
            f"after {self.max_retry_attempts} attempts."
        )

    def _send_line(self, message: str) -> None:
        """Send a line to the server.

        Args:
            message: Message string (without newline).

        Raises:
            RuntimeError: If send fails.
        """
        try:
            if self.socket is None:
                self._ensure_connected()
            self.socket.sendall((message + "\n").encode("utf-8"))
            logger.debug("Sent: %s", message)
        except Exception as e:
            logger.error("Send failed: %s. Reconnecting...", e)
            self._ensure_connected()
            try:
                self.socket.sendall((message + "\n").encode("utf-8"))
            except Exception as e2:
                raise RuntimeError(f"Failed to send after reconnect: {e2}") from e2

    def _recv_line(self) -> str:
        """Receive a line from the server with 4KB chunk buffering.

        Returns:
            The received message (without trailing whitespace).

        Raises:
            RuntimeError: If connection fails or socket times out.
        """
        try:
            if self.socket is None:
                self._ensure_connected()

            while True:
                # Check if we already have a complete line in buffer
                if b"\n" in self._buffer:
                    line, _, self._buffer = self._buffer.partition(b"\n")
                    return line.decode("utf-8").strip()

                # Read 4KB chunk from socket
                try:
                    chunk = self.socket.recv(4096)
                except socket.timeout:
                    logger.error("Socket timeout. Reconnecting...")
                    self._ensure_connected()
                    raise RuntimeError("Socket timeout.")

                if not chunk:
                    raise RuntimeError("Connection closed by server.")

                self._buffer += chunk

        except Exception as e:
            logger.error("Receive failed: %s", e)
            raise

    def handshake(self) -> None:
        """Execute the ACPC handshake.

        Expects to receive "VERSION:X.X.X" from server and responds with the same.
        """
        try:
            version_msg = self._recv_line()
            logger.debug("Server version message: %s", version_msg)

            # Default response (server accepts any compatible version)
            response = "VERSION:2.0.0"
            self._send_line(response)
            logger.info("ACPC handshake complete: %s", response)

        except Exception as e:
            logger.error("Handshake failed: %s", e)
            raise

    def parse_matchstate(self, message: str) -> MatchState | None:
        """Parse an ACPC MATCHSTATE message.

        Format: MATCHSTATE:state_id:position:action_history:hole_cards|board_cards

        Args:
            message: Raw MATCHSTATE string.

        Returns:
            Parsed MatchState object, or None if parsing fails.
        """
        try:
            if not message.startswith("MATCHSTATE:"):
                return None

            parts = message[len("MATCHSTATE:"):].split(":")
            if len(parts) < 4:
                logger.warning("Malformed MATCHSTATE: %s", message)
                return None

            state_id = int(parts[0])
            position = int(parts[1])
            action_history = parts[2]
            cards_part = parts[3]

            # Count "/" to determine game stage (0=preflop, 1=flop, 2=turn, 3=river)
            stage = action_history.count("/")

            # Parse cards: "AsKh|2d3c4d5s6h"
            card_parts = cards_part.split("|")
            if len(card_parts) < 1:
                logger.warning("Malformed cards: %s", cards_part)
                return None

            # Hole cards (first 4 chars: 2 cards × 2 chars each)
            hole_str = card_parts[0]
            hole_cards = [hole_str[i : i + 2] for i in range(0, len(hole_str), 2)]

            # Board cards (remaining)
            board_str = card_parts[1] if len(card_parts) > 1 else ""
            board_cards = [board_str[i : i + 2] for i in range(0, len(board_str), 2)] if board_str else []

            return MatchState(
                state_id=state_id,
                position=position,
                stage=stage,
                board_cards=board_cards,
                hole_cards=hole_cards,
                action_history=action_history,
            )

        except Exception as e:
            logger.error("Error parsing MATCHSTATE: %s", e)
            return None

    def parse_legal_actions(
        self, legal_actions_str: str
    ) -> tuple[list[int], dict[str, float]]:
        """Parse ACPC legal actions string.

        Format: "(fcr) /0:9000" means fold/check/raise with bounds 0-9000 chips.

        Args:
            legal_actions_str: ACPC legal actions string.

        Returns:
            Tuple of (legal_action_indices, action_bounds_dict).
        """
        legal_indices = []
        action_bounds = {}

        try:
            if "(" not in legal_actions_str or ")" not in legal_actions_str:
                logger.warning("Malformed legal actions: %s", legal_actions_str)
                return [1], {}

            # Extract actions from parentheses
            start = legal_actions_str.index("(") + 1
            end = legal_actions_str.index(")")
            actions_str = legal_actions_str[start:end]

            # 'f' = fold (0), 'c' = check/call (1), 'r' = raise (2-8)
            if "f" in actions_str:
                legal_indices.append(0)

            if "c" in actions_str:
                legal_indices.append(1)

            if "r" in actions_str:
                # Extract raise bounds if present
                bounds_match = re.search(r"/(\d+):(\d+)", legal_actions_str)
                if bounds_match:
                    min_raise = float(bounds_match.group(1))
                    max_raise = float(bounds_match.group(2))
                    action_bounds["min_raise"] = min_raise
                    action_bounds["max_raise"] = max_raise
                    # Add raise actions (2-8)
                    for i in range(2, 9):
                        legal_indices.append(i)

            if not legal_indices:
                legal_indices = [1]

            legal_indices = sorted(list(set(legal_indices)))
            return legal_indices, action_bounds

        except Exception as e:
            logger.error("Error parsing legal actions: %s", e)
            return [1], {}

    def send_action(self, action_str: str) -> None:
        """Send an action to the server.

        Args:
            action_str: Action string ('f', 'c', 'r{amount}', 'a{amount}').
        """
        try:
            self._send_line(action_str)
        except Exception as e:
            logger.error("Failed to send action '%s': %s", action_str, e)
            raise

    def parse_result(self, result_message: str) -> HandResult | None:
        """Parse a hand result message.

        Format: "HAND_RESULT:outcome:chips_chip_delta" or similar.

        Args:
            result_message: Raw result message.

        Returns:
            HandResult object, or None if parsing fails.
        """
        try:
            # Extract chip amounts (generic approach for various ACPC variants)
            # Format typically: "HAND_RESULT:+500:-500" (our delta first)
            if ":" in result_message:
                parts = result_message.split(":")
                if len(parts) >= 2:
                    try:
                        our_delta = float(parts[1].strip())
                        return HandResult(chip_delta=our_delta, reason="game_end")
                    except ValueError:
                        pass

            logger.warning("Could not parse result: %s", result_message)
            return None

        except Exception as e:
            logger.error("Error parsing result: %s", e)
            return None

    def close(self) -> None:
        """Close the socket connection."""
        if self.socket:
            try:
                self.socket.close()
                logger.info("AcpcClient closed")
            except Exception as e:
                logger.warning("Error closing socket: %s", e)
            finally:
                self.socket = None
