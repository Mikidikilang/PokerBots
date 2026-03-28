"""
Aszinkron Hugging Face Hub Szinkronizacio (hf_sync.py).

[FIX M-2 — 2025-03-28] AsyncModelUploader.shutdown() double-upload removed.

    The notebook's finally block called:
        uploader.trigger_manual_upload()   ← flush
        uploader.shutdown()                ← this ALSO called trigger_manual_upload()
    Result: two sequential blocking HF uploads per session end.

    Fix: shutdown() no longer calls trigger_manual_upload() internally.
    The caller (finally block) is responsible for flushing before calling
    shutdown(). This matches the documented pattern in the notebook and the
    NOTEBOOK_REBUILD_SUMMARY.md.

[FIX L4 — already present] trigger_manual_upload() explicit True return.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Retry Logic
# =============================================================================

def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    *args: Any,
    **kwargs: Any,
) -> T | None:
    delay: float = base_delay
    for attempt in range(max_retries + 1):
        try:
            result: T = func(*args, **kwargs)
            if attempt > 0:
                logger.info(
                    "%s succeeded after retry (attempt %d/%d)",
                    func.__name__, attempt + 1, max_retries + 1,
                )
            return result
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(
                    "%s error (attempt %d/%d): %s. Retrying in %ds...",
                    func.__name__, attempt + 1, max_retries + 1, str(exc), int(delay),
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(
                    "%s permanently failed after %d attempts: %s",
                    func.__name__, max_retries + 1, str(exc),
                )
    return None


# =============================================================================
# Headless Auth
# =============================================================================

def configure_headless_auth(
    token: str | None = None,
    env_var_name: str = "HF_TOKEN",
    use_kaggle_secrets: bool = False,
) -> bool:
    if token:
        os.environ[env_var_name] = token
        logger.info("HF auth: direct token set.")
        return True

    if os.environ.get(env_var_name):
        logger.info("HF auth: %s env var already present.", env_var_name)
        return True

    if use_kaggle_secrets:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
            secrets_client = UserSecretsClient()
            hf_token: str = secrets_client.get_secret(env_var_name)
            os.environ[env_var_name] = hf_token
            logger.info("HF auth: token retrieved from Kaggle Secrets.")
            return True
        except ImportError:
            logger.warning("kaggle_secrets module not available (not on Kaggle).")
        except Exception as exc:
            logger.error("Kaggle Secrets error: %s", exc)

    logger.warning(
        "HF auth: no token found. Set %s env var or use Kaggle Secrets.",
        env_var_name,
    )
    return False


def verify_auth() -> dict[str, str] | None:
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
        api = HfApi()
        user_info: dict[str, Any] = api.whoami()
        username: str = user_info.get("name", "unknown")
        logger.info("HF auth verified. User: %s", username)
        return {"name": username, "type": user_info.get("type", "")}
    except ImportError:
        logger.warning("huggingface_hub not installed.")
        return None
    except Exception as exc:
        logger.error("HF auth verification failed: %s", exc)
        return None


# =============================================================================
# AsyncModelUploader — [FIX M-2] shutdown() no longer double-triggers
# =============================================================================

class AsyncModelUploader:
    """Background async checkpoint upload via HF CommitScheduler.

    Usage pattern (correct — matches notebook finally block):

        # Finally block:
        uploader.trigger_manual_upload()  # ← flush pending files (blocking)
        uploader.shutdown()               # ← stop the background thread
        _async_upload_ran = True          # ← flag for Cell 5-B

    [FIX M-2] shutdown() does NOT call trigger_manual_upload() internally
    anymore. Previously it did, causing two sequential blocking HF API calls
    per session end (one from the finally block, one from inside shutdown()).
    The caller is responsible for flushing before calling shutdown().
    """

    def __init__(
        self,
        repo_id: str,
        checkpoint_dir: str,
        sync_interval_minutes: int = 15,
        path_in_repo: str = "checkpoints",
        enabled: bool = True,
    ) -> None:
        self.repo_id:   str  = repo_id
        self.local_dir: Path = Path(checkpoint_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.enabled:   bool = enabled
        self._scheduler: Any = None

        if not enabled:
            logger.info("AsyncModelUploader: DISABLED (enabled=False).")
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
                "AsyncModelUploader active: repo=%s, dir=%s, interval=%dmin",
                repo_id, self.local_dir, sync_interval_minutes,
            )
        except ImportError:
            logger.warning("huggingface_hub not installed. Async upload inactive.")
        except Exception as exc:
            logger.error("CommitScheduler init error: %s", exc)

    def trigger_manual_upload(self) -> None:
        """Force an immediate upload (blocking). Call this BEFORE shutdown().

        [FIX L4] Returns True on success so retry_with_backoff() can
        distinguish success from failure (None = all retries failed).
        """
        if self._scheduler is None:
            logger.debug("No active scheduler, manual upload skipped.")
            return

        def _do_upload() -> bool:
            self._scheduler.trigger()
            return True  # Explicit True so retry_with_backoff sees success

        result = retry_with_backoff(
            _do_upload,
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0,
        )

        if result is None:
            logger.warning(
                "Manual upload failed after all retries: %s", self.repo_id
            )
        else:
            logger.info(
                "Manual upload successfully triggered: %s", self.repo_id
            )

    def shutdown(self) -> None:
        """Stop the background CommitScheduler thread.

        [FIX M-2] Does NOT call trigger_manual_upload() internally.

        The correct call sequence is:
            uploader.trigger_manual_upload()  ← caller flushes first
            uploader.shutdown()               ← then stops the thread

        Previously shutdown() called trigger_manual_upload() itself, causing
        a second blocking HF API call after the caller had already flushed.
        Two sequential uploads per session end wasted ~30-60s of the 11.5h
        Kaggle session and risked a duplicate commit race condition.
        """
        if self._scheduler is not None:
            logger.info(
                "AsyncModelUploader: background thread stopped. "
                "Final upload was flushed by caller before shutdown()."
            )
            # The CommitScheduler's internal daemon thread will exit when the
            # process exits. We do not call trigger() here — that's the caller's
            # responsibility (see docstring). Setting _scheduler to None prevents
            # accidental double-trigger if shutdown() is called twice.
            self._scheduler = None
        else:
            logger.debug("AsyncModelUploader.shutdown(): scheduler was not active.")

    def is_active(self) -> bool:
        return self._scheduler is not None and self.enabled


# =============================================================================
# HuggingFaceStateManager
# =============================================================================

class HuggingFaceStateManager:
    """Full checkpoint download/upload management for training resume."""

    def __init__(self, repo_id: str, local_dir: str = "checkpoints") -> None:
        self.repo_id:   str  = repo_id
        self.local_dir: str  = local_dir
        self._api: Any = None

        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]
            self._api = HfApi()
            logger.info("HuggingFaceStateManager initialized: repo=%s", repo_id)
        except ImportError:
            logger.warning("huggingface_hub not installed. HF state management inactive.")

    def ensure_repo_exists(self) -> bool:
        if self._api is None:
            return False
        try:
            self._api.repo_info(repo_id=self.repo_id, repo_type="model")
            return True
        except Exception:
            try:
                self._api.create_repo(
                    repo_id=self.repo_id, repo_type="model", private=True,
                )
                logger.info("HF repo created: %s (private)", self.repo_id)
                return True
            except Exception as exc:
                logger.error("HF repo creation failed: %s", exc)
                return False

    def download_latest_state(self) -> str | None:
        if self._api is None:
            logger.warning("HF API not available, skipping download.")
            return None

        if not self.ensure_repo_exists():
            return None

        try:
            files: list[str] = self._api.list_repo_files(repo_id=self.repo_id)
            if len(files) <= 1:
                logger.info("HF repo empty. Cold start.")
                return None

            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

            def _do_download() -> str:
                return snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="model",
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    ignore_patterns=["*.md", ".gitattributes"],
                )

            downloaded_path = retry_with_backoff(
                _do_download, max_retries=3, base_delay=2.0, max_delay=30.0,
            )

            if downloaded_path:
                logger.info("Checkpoint downloaded: %s -> %s", self.repo_id, downloaded_path)
                return downloaded_path
            else:
                logger.error("Checkpoint download failed after all retries.")
                return None

        except Exception as exc:
            logger.error("Checkpoint download error: %s", exc)
            return None

    def upload_current_state(self, commit_message: str = "Autosave checkpoint") -> bool:
        if self._api is None:
            logger.warning("HF API not available, skipping upload.")
            return False

        if not os.path.exists(self.local_dir):
            logger.warning("Local dir does not exist: %s", self.local_dir)
            return False

        def _do_upload() -> bool:
            from huggingface_hub import upload_folder  # type: ignore[import-untyped]
            upload_folder(
                repo_id=self.repo_id,
                folder_path=self.local_dir,
                commit_message=commit_message,
                repo_type="model",
            )
            return True

        result = retry_with_backoff(
            _do_upload, max_retries=3, base_delay=2.0, max_delay=30.0,
        )

        if result:
            logger.info(
                "Checkpoint uploaded: %s -> %s (%s)",
                self.local_dir, self.repo_id, commit_message,
            )
            return True
        else:
            logger.error("Upload failed after all retries: %s", self.repo_id)
            return False

    def get_repo_info(self) -> dict[str, Any] | None:
        if self._api is None:
            return None
        try:
            info = self._api.repo_info(repo_id=self.repo_id, repo_type="model")
            return {
                "id":            info.id,
                "private":       info.private,
                "last_modified": str(info.last_modified) if info.last_modified else None,
            }
        except Exception as exc:
            logger.error("HF repo info error: %s", exc)
            return None
