"""
Graceful Shutdown es Hibaturesi Monitor (fault_tolerance.py).

A Kaggle platform 12 oras futasi limitjenek kezeleseert felelos.
A SIGKILL (Exit Code: 137) elkeruleseere a modul egy prediktiv
idozitot alkalmaz, amely 11.5 oranal (30 perces puffer) proaktivan
befagyasztja a treninget es elinditja a biztonságos mentes folyamatot.

Fo komponensek:
    - GracefulShutdownMonitor: time.monotonic() alapu idozito
    - FaultHandler: Kivetelekezeles (NaN, OOM) es rollback logika

Hivatkozasok:
    - Specifikacio: fault_tolerance.py — GracefulShutdown, 12 oras limit
    - Infrastruktura doc: A 12 oras idokorlat prediktiv kezelese
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Graceful Shutdown Monitor
# =============================================================================

@dataclass
class ShutdownConfig:
    """A graceful shutdown konfiguracioja.

    Attributes:
        max_runtime_hours: Maximalis futasi ido oraban.
        use_monotonic_clock: time.monotonic() hasznalata (NTP-fuggetlenseg).
        warning_threshold_minutes: Figyelmeztetes ennyi perccel a limit elott.
        register_signal_handlers: SIGTERM/SIGINT kezelo regisztralasa.
    """

    max_runtime_hours: float = 11.5
    use_monotonic_clock: bool = True
    warning_threshold_minutes: float = 45.0
    register_signal_handlers: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> ShutdownConfig:
        """YAML config szotarbol peldanyosit.

        Args:
            cfg: Teljes YAML konfiguracio.

        Returns:
            ShutdownConfig peldany.
        """
        gs = cfg.get("mlops", {}).get("graceful_shutdown", {})
        return cls(
            max_runtime_hours=gs.get("max_runtime_hours", 11.5),
            use_monotonic_clock=gs.get("use_monotonic_clock", True),
        )


class GracefulShutdownMonitor:
    """Prediktiv idozito a Kaggle 12 oras limit biztonsagos kezelesehez.

    A monitor a time.monotonic() fuggvenyt hasznalja az NTP ugrasok
    elkerulese erdekeben. Folyamatosan meri a futasidot, es jelzi
    ha a leallitas szukseges.

    A should_shutdown() metodust a training loop minden iteracio
    vegenek elejen kell meghivni.

    Example:
        >>> monitor = GracefulShutdownMonitor(ShutdownConfig(max_runtime_hours=11.5))
        >>> while not monitor.should_shutdown():
        ...     train_one_iteration()
        >>> # Graceful shutdown: mentes, feltoltes
        >>> save_checkpoint()

    Attributes:
        config: A shutdown konfiguracioja.
        start_time: A futtas indulasanak idopontja (monotonic).
    """

    def __init__(self, config: ShutdownConfig | None = None) -> None:
        """Inicializalja a shutdown monitort.

        Args:
            config: Shutdown konfiguracio. Alapertelmezett ha None.
        """
        self.config: ShutdownConfig = config or ShutdownConfig()

        # Idozites
        if self.config.use_monotonic_clock:
            self.start_time: float = time.monotonic()
        else:
            self.start_time = time.time()

        self.max_seconds: float = self.config.max_runtime_hours * 3600.0
        self._warning_issued: bool = False
        self._shutdown_requested: bool = False
        self._shutdown_callbacks: list[Callable[[], None]] = []

        # Signal kezelok regisztralasa (SIGTERM, SIGINT)
        if self.config.register_signal_handlers:
            self._register_signal_handlers()

        logger.info(
            "GracefulShutdownMonitor inicializalva: max=%.1f ora (%.0f sec), "
            "clock=%s, signal_handlers=%s",
            self.config.max_runtime_hours,
            self.max_seconds,
            "monotonic" if self.config.use_monotonic_clock else "wall",
            self.config.register_signal_handlers,
        )

    # =========================================================================
    # Fo Ellenorzes
    # =========================================================================

    def should_shutdown(self) -> bool:
        """Ellenorzi, hogy a training-et le kell-e allitani.

        Harom ok miatt lehet True:
            1. A futasido elerte a max_runtime_hours-t
            2. Kulso signal (SIGTERM/SIGINT) erkezett
            3. Programatikus request_shutdown() hivas

        A warning_threshold_minutes-nel logo figyelmeztetes kerul.

        Returns:
            True ha a treninget azonnal le kell allitani.
        """
        if self._shutdown_requested:
            logger.info("Shutdown: kulso keres altal triggerelve.")
            return True

        elapsed: float = self._get_elapsed_seconds()
        remaining: float = self.max_seconds - elapsed

        # Figyelmeztetes a kuszob eleresekkor
        warning_sec: float = self.config.warning_threshold_minutes * 60.0
        if remaining <= warning_sec and not self._warning_issued:
            self._warning_issued = True
            logger.warning(
                "FIGYELMEZTETES: Hatraleveo ido: %.1f perc (%.1f ora / %.1f max). "
                "Keszulj a graceful shutdown-ra!",
                remaining / 60.0,
                elapsed / 3600.0,
                self.config.max_runtime_hours,
            )

        # Leallitas ellenorzes
        if remaining <= 0:
            logger.warning(
                "IDOKORLAT ELERVE: Futasido %.2f ora >= %.2f ora. "
                "Graceful shutdown indul!",
                elapsed / 3600.0,
                self.config.max_runtime_hours,
            )
            return True

        return False

    # =========================================================================
    # Idokezeles
    # =========================================================================

    def _get_elapsed_seconds(self) -> float:
        """Visszaadja az eltelt idot masodpercben.

        Returns:
            Eltelt masodpercek szama.
        """
        if self.config.use_monotonic_clock:
            return time.monotonic() - self.start_time
        return time.time() - self.start_time

    def get_elapsed_hours(self) -> float:
        """Visszaadja az eltelt idot oraban.

        Returns:
            Eltelt orak szama.
        """
        return self._get_elapsed_seconds() / 3600.0

    def get_remaining_hours(self) -> float:
        """Visszaadja a hatralevo idot oraban.

        Returns:
            Hatralevo orak szama (0.0 ha lejart).
        """
        remaining: float = self.max_seconds - self._get_elapsed_seconds()
        return max(0.0, remaining / 3600.0)

    def get_remaining_seconds(self) -> float:
        """Visszaadja a hatralevo idot masodpercben.

        Returns:
            Hatralevo masodpercek szama (0.0 ha lejart).
        """
        return max(0.0, self.max_seconds - self._get_elapsed_seconds())

    def get_progress_pct(self) -> float:
        """Visszaadja a futasido szazalekos elorehaladast.

        Returns:
            Szazalekos ertek [0.0, 100.0].
        """
        elapsed: float = self._get_elapsed_seconds()
        return min(100.0, (elapsed / self.max_seconds) * 100.0)

    # =========================================================================
    # Kulso Leallitas
    # =========================================================================

    def request_shutdown(self, reason: str = "kulso keres") -> None:
        """Programatikus leallitasi keres.

        Args:
            reason: A leallitas indoka (logolashoz).
        """
        self._shutdown_requested = True
        logger.info("Shutdown kerelve: %s", reason)

    def register_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Regisztral egy callback fuggvenyt a shutdown elott.

        A callback-ek a signal handler altal hivja meg SIGTERM/SIGINT eseten.

        Args:
            callback: A meghivando fuggveny (argumentum nelkuli).
        """
        self._shutdown_callbacks.append(callback)
        logger.debug(
            "Shutdown callback regisztralva (osszes: %d)",
            len(self._shutdown_callbacks),
        )

    # =========================================================================
    # Signal Kezelok
    # =========================================================================

    def _register_signal_handlers(self) -> None:
        """SIGTERM es SIGINT kezeloket regisztral.

        A Kaggle SIGKILL-t kuld (nem kezhato), de elofordulhat
        SIGTERM is mas kornyezetekben. A SIGINT a KeyboardInterrupt
        ekvivalense.
        """
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            logger.debug("SIGTERM es SIGINT kezelok regisztralva.")
        except (OSError, ValueError) as exc:
            # Nem fo szalban nem lehet signal kezelot regisztralni
            logger.warning(
                "Signal kezelok regisztralasa sikertelen: %s. "
                "Valoszinuleg nem a fo szalban vagyunk.",
                exc,
            )

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Signal kezelö fuggveny.

        Args:
            signum: A signal szama (15=SIGTERM, 2=SIGINT).
            frame: Az aktualis stack frame.
        """
        sig_name: str = signal.Signals(signum).name
        logger.warning(
            "SIGNAL fogadva: %s (%d). Graceful shutdown indul...",
            sig_name, signum,
        )
        self._shutdown_requested = True

        # Callback-ek futtatasa
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as exc:
                logger.error("Shutdown callback hiba: %s", exc)

    # =========================================================================
    # Statisztikak
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Visszaadja a monitor aktualis allapot-informacioit.

        Returns:
            Dict a monitor allapotaval.
        """
        return {
            "elapsed_hours": self.get_elapsed_hours(),
            "remaining_hours": self.get_remaining_hours(),
            "progress_pct": self.get_progress_pct(),
            "max_runtime_hours": self.config.max_runtime_hours,
            "shutdown_requested": self._shutdown_requested,
            "warning_issued": self._warning_issued,
        }


# =============================================================================
# Hiba Kezelo (Fault Handler)
# =============================================================================

class FaultHandler:
    """Kritikus hibak (NaN, OOM, gradiens robbanas) kezelese.

    A FaultHandler a training ciklus try-except blokkjabol hivodik,
    es dontest hoz a hibakezelesrol:
        - NaN loss: Rollback az utolso jo checkpoint-ra
        - OOM: Batch meret csokkentes es ujrainditas
        - Ismeretlen hiba: Naplozas es biztonsagi mentes

    Attributes:
        max_nan_retries: Maximalis NaN utani ujraprobalkozas.
        nan_retry_count: Aktualis NaN szamlalo.
    """

    def __init__(self, max_nan_retries: int = 3) -> None:
        """Inicializalja a FaultHandler-t.

        Args:
            max_nan_retries: Hanyszor probalkozzon ujra NaN eseten.
        """
        self.max_nan_retries: int = max_nan_retries
        self.nan_retry_count: int = 0
        self._error_log: list[dict[str, Any]] = []

        logger.info(
            "FaultHandler inicializalva: max_nan_retries=%d",
            max_nan_retries,
        )

    def handle_nan_loss(self) -> str:
        """NaN loss detektalasakor hivando.

        Returns:
            Az ajanlott akcio: "retry" vagy "abort".
        """
        self.nan_retry_count += 1
        self._log_error("nan_loss", f"NaN loss #{self.nan_retry_count}")

        if self.nan_retry_count <= self.max_nan_retries:
            logger.warning(
                "NaN loss detektalt (%d/%d). Rollback ajanlott.",
                self.nan_retry_count, self.max_nan_retries,
            )
            return "retry"
        else:
            logger.error(
                "NaN loss %d alkalommal! Maximalis ujraprobalkozas elerve. "
                "A training nem folytathato biztonsagosan.",
                self.nan_retry_count,
            )
            return "abort"

    def handle_oom(self) -> str:
        """CUDA Out of Memory hiba eseten hivando.

        Returns:
            Az ajanlott akcio: "reduce_batch" vagy "abort".
        """
        self._log_error("oom", "CUDA Out of Memory")
        logger.error(
            "CUDA OOM hiba! Ajanlott: batch meret csokkentes es ujrainditas."
        )
        return "reduce_batch"

    def handle_generic_error(self, error: Exception) -> str:
        """Altalanos hiba eseten hivando.

        Args:
            error: A kivetel peldany.

        Returns:
            Az ajanlott akcio: "retry" vagy "abort".
        """
        error_type: str = type(error).__name__
        self._log_error(error_type, str(error))

        logger.error(
            "Altalanos hiba: %s — %s. Biztonsagi mentes ajanlott.",
            error_type, error,
        )
        return "retry"

    def reset_nan_counter(self) -> None:
        """NaN szamlalo nullazasa (sikeres iteracio utan)."""
        if self.nan_retry_count > 0:
            logger.debug("NaN szamlalo resetelve (volt: %d).", self.nan_retry_count)
            self.nan_retry_count = 0

    def _log_error(self, error_type: str, message: str) -> None:
        """Hiba esemeny naplozasa a belso logba.

        Args:
            error_type: A hiba tipusa.
            message: A hiba uzenet.
        """
        self._error_log.append({
            "type": error_type,
            "message": message,
            "timestamp": time.monotonic(),
        })

    def get_error_summary(self) -> dict[str, Any]:
        """Visszaadja a hibak osszefoglalo statisztikait.

        Returns:
            Dict a hiba statisztikakkal.
        """
        return {
            "total_errors": len(self._error_log),
            "nan_retries": self.nan_retry_count,
            "error_types": [e["type"] for e in self._error_log],
        }
