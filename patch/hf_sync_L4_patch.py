"""
hf_sync.py — PATCH: trigger_manual_upload() javitasa (L4 Fix)

[FIX L4 - 2025-03-28] trigger_manual_upload() mindig "sikertelen" loggolt:
    A _do_upload() belso fuggveny hivas utan None-t adott vissza
    (self._scheduler.trigger() nem teret vissza semmit). A retry_with_backoff()
    None visszateretest sikertelensegnek ertelmezte, ezert a "sikertelen"
    figyelmeztetoes mindig megjelent, meg sikeres feltoltes eseten is.
    Az "Manualis feltoltes triggerelve" info log soha nem jelent meg.

    A javitas: _do_upload() explicit True-t ad vissza, jelezve a sikert.
    retry_with_backoff() igy csak akkor ad vissza None-t, ha az osszes
    ujraprobalkozas is meghibasodott.

Csak a modositott metodus szukseges — ezt kell beilleszteni a teljes hf_sync.py-ba.
"""


def trigger_manual_upload_FIXED(self) -> None:
    """Azonnali feltoltest kenyszerit ki (pl. graceful shutdown eseten).

    [FIX L4] A korabbi implementacioban _do_upload() None-t adott vissza
    (a self._scheduler.trigger() void fuggveny). A retry_with_backoff()
    None visszaterest sikertelennek ertelmezte, ezert:
        - A "sikertelen" warning mindig megjelent (meg sikeres eseten is)
        - A "triggerelve" info log sohasem jelent meg

    A javitas: _do_upload() explicit True-t ad vissza. retry_with_backoff()
    csak ackkor adja vissza None-t, ha az osszes kiserlet Exception-t dobott.
    """
    if self._scheduler is None:
        logger.debug("Nincs aktiv scheduler, manual upload kihagyva.")
        return

    def _do_upload() -> bool:   # [FIX L4] Explicit bool visszateresi tipus
        self._scheduler.trigger()
        return True              # [FIX L4] True = siker jelzese a retry_with_backoff-nak

    result = retry_with_backoff(
        _do_upload,
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0,
    )

    # [FIX L4] result is None CSAK ha az osszes kiserlet is meghibasodott
    # Siker eseten result == True (nem None)
    if result is None:
        logger.warning(
            "Manualis feltoltes sikertelen az osszes ujraprobalkozas utan: %s",
            self.repo_id,
        )
    else:
        logger.info("Manualis feltoltes sikeresen triggerelve: %s", self.repo_id)


# =============================================================================
# Beillesztesi utmutato:
# A hf_sync.py AsyncModelUploader osztalyban csereld le a
# trigger_manual_upload() metodus tozsset a fentire.
# A "logger" es "retry_with_backoff" mar definialva van a modulus szintjen.
# =============================================================================
