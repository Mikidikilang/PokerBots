"""
ACPC (Annual Computer Poker Competition) Protocol Client (acpc_client.py).

[FIX L5 - 2025-03-28] Hianyzo AcpcClient Osztalydefinicio Hozzaadva:
    A korabbi forrasban az AcpcClient osztaly torzsebe (docstring, __init__,
    metodusok) nem volt megelozo `class AcpcClient:` deklaracio. A `Manages
    a persistent connection...` docstring szoveg az HandResult adatosztaly
    belsejeben lebegett, nem valodi osztalydefinicio reszeként. Python
    ezt futasidoben string literalkent ertelmezi, nem emesl hibat, de
    az `AcpcClient` nev nem krul definiálásra — ezert az `acpc_client.py`
    importja NameError-t dobott, ami az egesz `src/evaluation` modult
    megtorte.

A TCP socket-alapu kliens implementalja az ACPC poker protokollt:
  1. Server kuldez: "VERSION:2.0.0"
  2. Kliens valaszol: "VERSION:2.0.0"
  3. Server kuldez: Match konfiguraciot
  4. Jatek: Server MATCHSTATE uzeneteket kuldez, kliens akciot valaszol

MATCHSTATE Format (pelda):
    MATCHSTATE:0:0:r100c//:Kh9h|2d3c4d5s6h

Akciok:
    - 'f' : Fold
    - 'c' : Check or Call
    - 'r{amount}' : Raise to {amount}
"""

from __future__ import annotations

import logging
import re
import socket
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MatchState:
    """Feldolgozott ACPC MATCHSTATE uzenet.

    Attributes:
        state_id: Egyedi allapot azonosito.
        position: Jatekos pozicio (0 vagy 1 HU-ban).
        stage: Jatekfazis (0=preflop, 1=flop, 2=turn, 3=river).
        board_cards: Kozos lapok kartyajelolések listajakent (RankSuit).
        hole_cards: Sajat lapok listajakent (RankSuit: As, Kh).
        action_history: Eddigi akciok karakterlancban (pl. "r100c").
    """

    state_id: int
    position: int
    stage: int
    board_cards: list[str]
    hole_cards: list[str]
    action_history: str


@dataclass
class HandResult:
    """Egy lez vegeredmenye.

    Attributes:
        chip_delta: Nyert (+) vagy vesztett (-) zsetonok a mi szemszogunkbol.
        reason: A lez vegzodesi oka (fold, showdown stb.).
    """

    chip_delta: float
    reason: str


# =============================================================================
# [FIX L5] AcpcClient osztaly deklaracio hozzaadva
# =============================================================================

class AcpcClient:
    """Allandó TCP kapcsolatot kezel egy ACPC poker szerverrel (pl. Slumbot).

    Kezeli a protokoll handshake-et, a jatekallapot parszolast, az akcio
    kuldest es az eredmeny parszolast. Tartalmaz automatikus ujracsatlakozast
    exponencialis backoff-tal.

    [FIX L5] Ez az osztaly deklaracio hianyzott a korabbi forrasban.
    Az osztaly torzsebe (docstring, __init__, metodusok) megvoltak, de
    a `class AcpcClient:` sor hianyzott. Python a docstringet az HandResult
    osztaly belsejeben string literalkent ertelmezi, az AcpcClient nev
    nem kerult definialasra, es az import NameError-t dobott.

    Args:
        host: Szerver hostname vagy IP cim.
        port: Szerver port szama.
        max_retry_attempts: Maximalis ujracsatlakozasi kiserletek.
        base_retry_delay_seconds: Alap varakozasi ido exponencialis backoff-hoz.
        socket_timeout_seconds: Socket timeout masodpercben.

    Example:
        >>> client = AcpcClient(host="slumbot.com", port=9000)
        >>> client.handshake()
        >>> state = client.parse_matchstate(message)
        >>> client.send_action("c")
        >>> client.close()
    """

    def __init__(
        self,
        host: str,
        port: int,
        max_retry_attempts: int = 5,
        base_retry_delay_seconds: float = 1.0,
        socket_timeout_seconds: float = 30.0,
    ) -> None:
        """Inicializalja az ACPC klienst es csatlakozik a szerverhez.

        Args:
            host: Szerver hostname vagy IP cim.
            port: Szerver port szama.
            max_retry_attempts: Maximalis ujracsatlakozasi kiserletek.
            base_retry_delay_seconds: Alap varakozasi ido backoff-hoz.
            socket_timeout_seconds: Socket timeout masodpercben.
        """
        self.host: str = host
        self.port: int = port
        self.max_retry_attempts: int = max_retry_attempts
        self.base_retry_delay: float = base_retry_delay_seconds
        self.socket_timeout: float = socket_timeout_seconds

        self.socket: socket.socket | None = None
        self._buffer: bytes = b""
        self._ensure_connected()

        logger.info(
            "AcpcClient inicializalva: %s:%d, timeout=%.1fs, max_retries=%d",
            host, port, socket_timeout_seconds, max_retry_attempts,
        )

    def _ensure_connected(self) -> None:
        """Szallitja a socket csatlakozasat exponencialis backoff retry-al."""
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
                logger.info("AcpcClient csatlakozva: %s:%d", self.host, self.port)
                return

            except Exception as e:
                delay = self.base_retry_delay * (2 ** attempt)
                logger.warning(
                    "Csatlakozasi kiserlet %d/%d meghiusult: %s. "
                    "Ujraprobalkozas %.1fs utan...",
                    attempt + 1, self.max_retry_attempts, e, delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Nem sikerult csatlakozni: {self.host}:{self.port} "
            f"({self.max_retry_attempts} kiserlet utan)."
        )

    def _send_line(self, message: str) -> None:
        """Egy sort kuld a szervernek.

        Args:
            message: Uzenet szoveg (newline nelkul).

        Raises:
            RuntimeError: Ha a kuldes sikertelen.
        """
        try:
            if self.socket is None:
                self._ensure_connected()
            self.socket.sendall((message + "\n").encode("utf-8"))
            logger.debug("Kuldve: %s", message)
        except Exception as e:
            logger.error("Kuldes sikertelen: %s. Ujracsatlakozas...", e)
            self._ensure_connected()
            try:
                self.socket.sendall((message + "\n").encode("utf-8"))
            except Exception as e2:
                raise RuntimeError(f"Ujracsatlakozas utan is sikertelen: {e2}") from e2

    def _recv_line(self) -> str:
        """Egy sort fogad a szervertol 4KB-os chunk pufferelessel.

        Returns:
            A fogadott uzenet (trailing whitespace nelkul).

        Raises:
            RuntimeError: Ha a csatlakozas megszakad.
        """
        try:
            if self.socket is None:
                self._ensure_connected()

            while True:
                if b"\n" in self._buffer:
                    line, _, self._buffer = self._buffer.partition(b"\n")
                    return line.decode("utf-8").strip()

                try:
                    chunk = self.socket.recv(4096)
                except socket.timeout:
                    logger.error("Socket timeout. Ujracsatlakozas...")
                    self._ensure_connected()
                    raise RuntimeError("Socket timeout.")

                if not chunk:
                    raise RuntimeError("A szerver lezarta a kapcsolatot.")

                self._buffer += chunk

        except Exception as e:
            logger.error("Fogadas sikertelen: %s", e)
            raise

    def handshake(self) -> None:
        """Vegrehajtja az ACPC handshake-et.

        Elvarja a "VERSION:X.X.X" uzenetet a szervertol, majd ugyan azt valaszolja.
        """
        try:
            version_msg = self._recv_line()
            logger.debug("Szerver verzio uzenet: %s", version_msg)
            response = "VERSION:2.0.0"
            self._send_line(response)
            logger.info("ACPC handshake kesz: %s", response)
        except Exception as e:
            logger.error("Handshake sikertelen: %s", e)
            raise

    def parse_matchstate(self, message: str) -> MatchState | None:
        """Feldolgoz egy ACPC MATCHSTATE uzenetet.

        Formatum: MATCHSTATE:state_id:position:action_history:hole_cards|board_cards

        Args:
            message: Nyers MATCHSTATE string.

        Returns:
            Feldolgozott MatchState objektum, vagy None ha sikertelen.
        """
        try:
            if not message.startswith("MATCHSTATE:"):
                return None

            parts = message[len("MATCHSTATE:"):].split(":")
            if len(parts) < 4:
                logger.warning("Hibas MATCHSTATE: %s", message)
                return None

            state_id = int(parts[0])
            position = int(parts[1])
            action_history = parts[2]
            cards_part = parts[3]

            stage = action_history.count("/")

            card_parts = cards_part.split("|")
            if len(card_parts) < 1:
                logger.warning("Hibas kartya resz: %s", cards_part)
                return None

            hole_str = card_parts[0]
            hole_cards = [hole_str[i: i + 2] for i in range(0, len(hole_str), 2)]

            board_str = card_parts[1] if len(card_parts) > 1 else ""
            board_cards = (
                [board_str[i: i + 2] for i in range(0, len(board_str), 2)]
                if board_str else []
            )

            return MatchState(
                state_id=state_id,
                position=position,
                stage=stage,
                board_cards=board_cards,
                hole_cards=hole_cards,
                action_history=action_history,
            )

        except Exception as e:
            logger.error("Hiba a MATCHSTATE feldolgozasaban: %s", e)
            return None

    def parse_legal_actions(
        self, legal_actions_str: str
    ) -> tuple[list[int], dict[str, float]]:
        """Feldolgozza az ACPC legalis akciok stringet.

        Args:
            legal_actions_str: ACPC legalis akciok string.

        Returns:
            (legalis_akcio_indexek, akcio_hatarok) tuple.
        """
        legal_indices = []
        action_bounds: dict[str, float] = {}

        try:
            if "(" not in legal_actions_str or ")" not in legal_actions_str:
                logger.warning("Hibas legalis akciok: %s", legal_actions_str)
                return [1], {}

            start = legal_actions_str.index("(") + 1
            end = legal_actions_str.index(")")
            actions_str = legal_actions_str[start:end]

            if "f" in actions_str:
                legal_indices.append(0)
            if "c" in actions_str:
                legal_indices.append(1)
            if "r" in actions_str:
                bounds_match = re.search(r"/(\d+):(\d+)", legal_actions_str)
                if bounds_match:
                    action_bounds["min_raise"] = float(bounds_match.group(1))
                    action_bounds["max_raise"] = float(bounds_match.group(2))
                for i in range(2, 9):
                    legal_indices.append(i)

            if not legal_indices:
                legal_indices = [1]

            legal_indices = sorted(list(set(legal_indices)))
            return legal_indices, action_bounds

        except Exception as e:
            logger.error("Hiba a legalis akciok feldolgozasaban: %s", e)
            return [1], {}

    def send_action(self, action_str: str) -> None:
        """Elkuld egy akciot a szervernek.

        Args:
            action_str: Akcio string ('f', 'c', 'r{amount}').
        """
        try:
            self._send_line(action_str)
        except Exception as e:
            logger.error("Akcio kuldes sikertelen '%s': %s", action_str, e)
            raise

    def parse_result(self, result_message: str) -> HandResult | None:
        """Feldolgoz egy kez vegeredmeny uzenetet.

        Args:
            result_message: Nyers eredmeny uzenet.

        Returns:
            HandResult objektum, vagy None ha sikertelen.
        """
        try:
            if ":" in result_message:
                parts = result_message.split(":")
                if len(parts) >= 2:
                    try:
                        our_delta = float(parts[1].strip())
                        return HandResult(chip_delta=our_delta, reason="game_end")
                    except ValueError:
                        pass

            logger.warning("Nem lehet feldolgozni az eredmenyt: %s", result_message)
            return None

        except Exception as e:
            logger.error("Hiba az eredmeny feldolgozasaban: %s", e)
            return None

    def close(self) -> None:
        """Lezarja a socket kapcsolatot."""
        if self.socket:
            try:
                self.socket.close()
                logger.info("AcpcClient lezarva")
            except Exception as e:
                logger.warning("Hiba a socket lezarasanal: %s", e)
            finally:
                self.socket = None
