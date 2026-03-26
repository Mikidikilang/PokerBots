"""
Aszinkron Hugging Face Hub Szinkronizacio (hf_sync.py).

A Kaggle efemer kornyezetbol a HF Hub felhobe torteno aszinkron
checkpoint szinkronizaciot vegzi. A fo komponensek:

    1. Headless Autentikacio: A HF_TOKEN kornyezeti valtozon
       keresztul, interaktiv prompt nelkul (Kaggle Save & Run All).

    2. AsyncModelUploader: A CommitScheduler hatterszalon figyeli
       a lokalis checkpoint konyvtarat es periodikusan (pl. 15 percenkent)
       feltolti a valtozasokat a HF Hub-ra Git-LFS-sel.

    3. HuggingFaceStateManager: Teljes checkpoint letoltes
       (snapshot_download) es feltoltes (upload_folder) kezeles
       a resume training szamara.

Hivatkozasok:
    - Specifikacio: hf_sync.py — CommitScheduler, headless auth
    - Infrastruktura doc: I/O szuk keresztmetszetek, GPU Starvation
    - huggingface_hub: https://huggingface.co/docs/huggingface_hub
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Headless Autentikacio
# =============================================================================

def configure_headless_auth(
    token: str | None = None,
    env_var_name: str = "HF_TOKEN",
    use_kaggle_secrets: bool = False,
) -> bool:
    """Konfigurálja a Hugging Face API hozzaferést kornyezeti valtozon keresztul.

    Ez a fuggveny a training script legelejen hivando. A token
    harom forrasbol szarmazhat (prioritas sorrendben):
        1. Kozvetlen atadas (token parameter)
        2. Mar letező kornyezeti valtozo (HF_TOKEN)
        3. Kaggle UserSecretsClient (use_kaggle_secrets=True)

    Args:
        token: Kozvetlen HF token (pl. teszteleshez).
        env_var_name: A kornyezeti valtozo neve.
        use_kaggle_secrets: True ha a Kaggle Secrets-bol kell olvasni.

    Returns:
        True ha az autentikacio sikeres.
    """
    # 1. Kozvetlen token
    if token:
        os.environ[env_var_name] = token
        logger.info("HF auth: kozvetlen token beallitva.")
        return True

    # 2. Mar letező kornyezeti valtozo
    if os.environ.get(env_var_name):
        logger.info("HF auth: %s kornyezeti valtozo mar letezik.", env_var_name)
        return True

    # 3. Kaggle Secrets
    if use_kaggle_secrets:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
            secrets_client = UserSecretsClient()
            hf_token: str = secrets_client.get_secret(env_var_name)
            os.environ[env_var_name] = hf_token
            logger.info("HF auth: token kinyerve a Kaggle Secrets-bol.")
            return True
        except ImportError:
            logger.warning(
                "kaggle_secrets modul nem elerheto. "
                "Valoszinuleg nem Kaggle kornyezetben vagyunk."
            )
        except Exception as exc:
            logger.error("Kaggle Secrets hiba: %s", exc)

    logger.warning(
        "HF auth: nem talalhato token! "
        "Allitsd be a %s kornyezeti valtozot vagy hasznalj Kaggle Secrets-et.",
        env_var_name,
    )
    return False


def verify_auth() -> dict[str, str] | None:
    """Ellenorzi az aktiv HF autentikaciott es visszaadja a felhasznaloi adatokat.

    Returns:
        Felhasznaloi profil szotar, vagy None ha nincs autentikacio.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
        api = HfApi()
        user_info: dict[str, Any] = api.whoami()
        username: str = user_info.get("name", "ismeretlen")
        logger.info("HF auth verifikaciok OK. Felhasznalo: %s", username)
        return {"name": username, "type": user_info.get("type", "")}
    except ImportError:
        logger.warning("huggingface_hub nem telepitett.")
        return None
    except Exception as exc:
        logger.error("HF auth verifikacio sikertelen: %s", exc)
        return None


# =============================================================================
# Aszinkron Model Feltolto (CommitScheduler Wrapper)
# =============================================================================

class AsyncModelUploader:
    """Hatterszalas aszinkron checkpoint feltoltes a HF Hub-ra.

    A CommitScheduler-t hasznalva egy fuggetlen hatterszal (daemon thread)
    figyeli a lokalis konyvtarat es periodikusan feltolti a valtozasokat
    a HF Hub repozitoriumba.

    A fo training szal csak torch.save()-vel ir a lokalis diszkre,
    a hatterszal automatikusan kezeli a halozati transzfert, megkerulve
    a GIL korlatozasait az I/O varakozasok alatt.

    Example:
        >>> uploader = AsyncModelUploader("user/poker-ai", "checkpoints/")
        >>> # A training loop torch.save()-vel ir a checkpoints/ konyvtarba
        >>> # A CommitScheduler 15 percenkent szinkronizal
        >>> uploader.trigger_manual_upload()  # Graceful shutdown-kor
        >>> uploader.shutdown()

    Attributes:
        repo_id: A cel HF repozitorium azonositoja.
        local_dir: A figyelt lokalis konyvtar.
        scheduler: A CommitScheduler peldany (ha elerheto).
    """

    def __init__(
        self,
        repo_id: str,
        checkpoint_dir: str,
        sync_interval_minutes: int = 15,
        path_in_repo: str = "checkpoints",
        enabled: bool = True,
    ) -> None:
        """Inicializalja az aszinkron feltoltot.

        Args:
            repo_id: A HF repozitorium azonositoja (pl. "user/poker-ai").
            checkpoint_dir: A lokalis checkpoint konyvtar.
            sync_interval_minutes: Szinkronizacios intervallum percben.
            path_in_repo: A cel konyvtar a repozitoriumban.
            enabled: Ha False, a feltoltes kikapcsolt (teszt mod).
        """
        self.repo_id: str = repo_id
        self.local_dir: Path = Path(checkpoint_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.enabled: bool = enabled
        self._scheduler: Any = None

        if not enabled:
            logger.info("AsyncModelUploader: KIKAPCSOLVA (enabled=False).")
            return

        try:
            from huggingface_hub import CommitScheduler  # type: ignore[import-untyped]

            self._scheduler = CommitScheduler(
                repo_id=self.repo_id,
                repo_type="model",
                folder_path=str(self.local_dir),
                path_in_repo=path_in_repo,
                every=sync_interval_minutes,
            )

            logger.info(
                "AsyncModelUploader aktiválva: repo=%s, dir=%s, "
                "interval=%d perc, path_in_repo=%s",
                repo_id, self.local_dir,
                sync_interval_minutes, path_in_repo,
            )

        except ImportError:
            logger.warning(
                "huggingface_hub nem elerheto. "
                "Aszinkron feltoltes nem aktiv."
            )
        except Exception as exc:
            logger.error("CommitScheduler inicializalas hiba: %s", exc)

    def trigger_manual_upload(self) -> None:
        """Azonnali feltoltest kenyszerit ki (pl. graceful shutdown eseten).

        Ez a metodus blokkol amig a feltoltes befejezodik.
        """
        if self._scheduler is None:
            logger.debug("Nincs aktiv scheduler, manual upload kihagyva.")
            return

        try:
            self._scheduler.trigger()
            logger.info("Manualis feltoltes triggerelve: %s", self.repo_id)
        except Exception as exc:
            logger.error("Manualis feltoltes sikertelen: %s", exc)

    def shutdown(self) -> None:
        """Biztonsagos leallitas: befejezi az utolso feltoltest.

        A CommitScheduler belso atexit hook-ja garantalja az utolso
        commit-ot, de expliciten is triggerelheto.
        """
        if self._scheduler is not None:
            try:
                self.trigger_manual_upload()
                logger.info("AsyncModelUploader leallitva. Utolso szinkronizacios kesz.")
            except Exception as exc:
                logger.error("Shutdown kozben hiba: %s", exc)

    def is_active(self) -> bool:
        """Visszaadja, hogy a scheduler aktiv-e.

        Returns:
            True ha a CommitScheduler fut.
        """
        return self._scheduler is not None and self.enabled


# =============================================================================
# HuggingFace Allapottér Manager (Teljes Letoltes/Feltoltes)
# =============================================================================

class HuggingFaceStateManager:
    """A HF Hub repozitorium teljes allapotter-kezeleseert felelos.

    Ket fo feladata:
        1. download_latest_state: A legfrissebb checkpoint letoltese
           a resume training-hez (snapshot_download).
        2. upload_current_state: A lokalis checkpoint teljes szinkron
           feltoltese (upload_folder) — graceful shutdown eseten.

    A symlink-ek letiltasa (local_dir_use_symlinks=False) kritikus a Kaggle
    kornyezetben a fajl irhatatossag elkerulese erdekeben.

    Example:
        >>> mgr = HuggingFaceStateManager("user/poker-ai", "checkpoints/")
        >>> path = mgr.download_latest_state()
        >>> if path:
        ...     load_checkpoint(path)
        >>> # Training...
        >>> mgr.upload_current_state("Checkpoint iter 5000")

    Attributes:
        repo_id: A HF repozitorium azonositoja.
        local_dir: A lokalis checkpoint konyvtar.
    """

    def __init__(self, repo_id: str, local_dir: str = "checkpoints") -> None:
        """Inicializalja a state manager-t.

        Args:
            repo_id: A HF repozitorium azonositoja.
            local_dir: A lokalis checkpoint konyvtar.
        """
        self.repo_id: str = repo_id
        self.local_dir: str = local_dir
        self._api: Any = None

        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]
            self._api = HfApi()
            logger.info("HuggingFaceStateManager inicializalva: repo=%s", repo_id)
        except ImportError:
            logger.warning(
                "huggingface_hub nem telepitett. "
                "HF state management nem aktiv."
            )

    def ensure_repo_exists(self) -> bool:
        """Ellenorzi a repozitorium letezeeset, letrehozza ha hiányzik.

        Returns:
            True ha a repo letezik (vagy sikeresen letrejott).
        """
        if self._api is None:
            return False

        try:
            self._api.repo_info(repo_id=self.repo_id, repo_type="model")
            logger.debug("HF repo letezik: %s", self.repo_id)
            return True
        except Exception:
            try:
                self._api.create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=True,
                )
                logger.info("HF repo letrehozva: %s (privat)", self.repo_id)
                return True
            except Exception as exc:
                logger.error("HF repo letrehozas sikertelen: %s", exc)
                return False

    def download_latest_state(self) -> str | None:
        """Letolti a teljes checkpoint strukturat a HF Hub-rol.

        A symlinkek letiltasa (local_dir_use_symlinks=False) kotelezo
        a Kaggle kornyezetben az irhato fizikai fajlok erdekeben.

        Returns:
            A letoltott konyvtar eleresi utja, vagy None ha ures/sikertelen.
        """
        if self._api is None:
            logger.warning("HF API nem elerheto, letoltes kihagyva.")
            return None

        if not self.ensure_repo_exists():
            return None

        try:
            # Ures repo ellenorzes
            files: list[str] = self._api.list_repo_files(repo_id=self.repo_id)
            if len(files) <= 1:  # Altalaban .gitattributes jelen van
                logger.info("HF repo ures. Training scratch-bol indul.")
                return None

            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

            downloaded_path: str = snapshot_download(
                repo_id=self.repo_id,
                repo_type="model",
                local_dir=self.local_dir,
                local_dir_use_symlinks=False,  # KRITIKUS: Kaggle kompatibilitas
                resume_download=True,
                ignore_patterns=["*.md", ".gitattributes"],
            )

            logger.info(
                "Checkpoint letoltve: %s -> %s",
                self.repo_id, downloaded_path,
            )
            return downloaded_path

        except Exception as exc:
            logger.error("Checkpoint letoltes hiba: %s", exc)
            return None

    def upload_current_state(
        self,
        commit_message: str = "Autosave checkpoint",
    ) -> bool:
        """Feltolti a teljes lokalis checkpoint konyvtarat a HF Hub-ra.

        Ez a szinkron (blokkolo) feltoltes a graceful shutdown eseten hasznalando.
        Normalis mukodes kozben az AsyncModelUploader (CommitScheduler) vegzi
        az aszinkron feltoltest.

        Args:
            commit_message: A Git commit uzenet.

        Returns:
            True ha a feltoltes sikeres.
        """
        if self._api is None:
            logger.warning("HF API nem elerheto, feltoltes kihagyva.")
            return False

        if not os.path.exists(self.local_dir):
            logger.warning("Lokalis konyvtar nem letezik: %s", self.local_dir)
            return False

        try:
            from huggingface_hub import upload_folder  # type: ignore[import-untyped]

            upload_folder(
                repo_id=self.repo_id,
                folder_path=self.local_dir,
                commit_message=commit_message,
                repo_type="model",
            )

            logger.info(
                "Checkpoint feltoltve: %s -> %s (%s)",
                self.local_dir, self.repo_id, commit_message,
            )
            return True

        except Exception as exc:
            logger.error("Checkpoint feltoltes hiba: %s", exc)
            return False

    def get_repo_info(self) -> dict[str, Any] | None:
        """Visszaadja a repozitorium informacioit.

        Returns:
            Dict a repo adataival, vagy None ha nem elerheto.
        """
        if self._api is None:
            return None

        try:
            info = self._api.repo_info(repo_id=self.repo_id, repo_type="model")
            return {
                "id": info.id,
                "private": info.private,
                "last_modified": str(info.last_modified) if info.last_modified else None,
            }
        except Exception as exc:
            logger.error("HF repo info hiba: %s", exc)
            return None
