"""
Weights & Biases Integration (monitoring.py).

Provides fail-safe W&B monitoring for the training pipeline. If wandb is
unavailable, the API key is missing, or network fails, the training loop
continues without interruption. This module is decoupled from the core
training logic and gracefully degrades when W&B is unavailable.

Architecture:
    - WandbMonitor wraps wandb initialization and logging
    - If setup() fails for any reason, active flag is set to False
    - All subsequent log calls check active flag and return early
    - No exceptions are raised; failures are logged as warnings

Headless Authentication:
    - Local: Uses environment variable WANDB_API_KEY
    - Kaggle: Falls back to kaggle_secrets.UserSecretsClient() if env var missing
    - If authentication fails, active=False and training continues
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import wandb; set flag if unavailable
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.debug("wandb not installed; W&B monitoring will be disabled")


# =============================================================================
# Weights & Biases Monitor
# =============================================================================


class WandbMonitor:
    """Fail-safe Weights & Biases monitoring for training pipeline.

    If wandb initialization fails for any reason (missing API key, network
    error, authentication failure), the training loop continues without
    disruption. The active flag tracks whether monitoring is functional.

    Attributes:
        active: True if W&B is available and authenticated, False otherwise.
        run_id: The W&B run ID, or None if not active.
    """

    def __init__(self) -> None:
        """Initialize the W&B monitor in inactive state.

        The monitor remains inactive until setup() is successfully called.
        """
        self.active = False
        self.run_id: str | None = None
        self.project_name: str | None = None
        logger.info("WandbMonitor initialized (inactive until setup)")

    def setup(
        self,
        config: dict[str, Any],
        resume: bool = False,
        run_id: str | None = None,
    ) -> None:
        """Initialize W&B run with fail-safe error handling.

        If wandb is not available or initialization fails, sets active=False
        and continues execution without raising exceptions.

        Args:
            config: Full configuration dict (must contain project.name).
            resume: If True and run_id provided, resume existing run.
            run_id: Optional W&B run ID to resume (ignored if resume=False).

        Returns:
            None. Sets self.active and self.run_id based on success/failure.
        """
        # Check if wandb is available
        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed; W&B monitoring disabled")
            self.active = False
            return

        # Extract project name from config
        try:
            self.project_name = config.get("project", {}).get("name", "poker-ai")
        except (TypeError, AttributeError):
            self.project_name = "poker-ai"
            logger.warning("Could not extract project name from config, using default")

        # Attempt to authenticate with W&B
        try:
            self._authenticate()
        except Exception as e:
            logger.warning("W&B authentication failed: %s; monitoring disabled", e)
            self.active = False
            return

        # Attempt to initialize W&B run
        try:
            if resume and run_id:
                # Resume existing run
                logger.info("Resuming W&B run: %s", run_id)
                wandb.init(
                    project=self.project_name,
                    id=run_id,
                    resume="allow",
                    config=config,
                )
                self.run_id = run_id
            else:
                # Start new run with timestamp-based name
                run_name = self._generate_run_name()
                logger.info("Starting new W&B run: %s", run_name)
                wandb.init(
                    project=self.project_name,
                    name=run_name,
                    config=config,
                    resume="never",
                )
                self.run_id = wandb.run.id if wandb.run else None

            self.active = True
            logger.info(
                "W&B monitoring active: project=%s, run_id=%s",
                self.project_name,
                self.run_id,
            )

        except Exception as e:
            logger.warning(
                "W&B initialization failed: %s; monitoring disabled", e, exc_info=True
            )
            self.active = False
            self.run_id = None

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Log metrics to W&B if monitoring is active.

        Returns immediately if active=False. Never raises exceptions.

        Args:
            step: Global step/iteration for X-axis (e.g., training iteration).
            metrics: Dictionary of metric name -> value pairs.
                    Supports nested dicts; W&B flattens to dot notation.
        """
        if not self.active:
            return

        try:
            if not metrics:
                return

            # Add step information
            metrics_with_step = {**metrics, "step": step}

            # Log to W&B
            wandb.log(metrics_with_step, step=step)

        except Exception as e:
            logger.warning("W&B logging failed: %s; skipping this batch", e)
            # Continue execution; don't set active=False to allow recovery

    def finish(self) -> None:
        """Gracefully finish W&B monitoring.

        Safe to call even if active=False or W&B is unavailable.
        """
        if not self.active:
            return

        try:
            wandb.finish()
            logger.info("W&B run finished: %s", self.run_id)
        except Exception as e:
            logger.warning("Error finishing W&B run: %s", e)
        finally:
            self.active = False

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _authenticate(self) -> None:
        """Authenticate with W&B using environment variable or Kaggle secrets.

        Raises:
            RuntimeError: If authentication fails.
        """
        # Try environment variable first
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            logger.debug("W&B authenticated using WANDB_API_KEY environment variable")
            return

        # Try Kaggle secrets (for Kaggle kernel environment)
        try:
            from kaggle_secrets import UserSecretsClient

            secrets_client = UserSecretsClient()
            api_key = secrets_client.get_secret("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
                logger.debug("W&B authenticated using Kaggle secrets")
                return
        except ImportError:
            # kaggle_secrets not available (not running on Kaggle)
            pass
        except Exception as e:
            logger.debug("Could not retrieve WANDB_API_KEY from Kaggle secrets: %s", e)

        # No authentication available
        raise RuntimeError(
            "W&B authentication failed: WANDB_API_KEY not in environment or Kaggle secrets"
        )

    def _generate_run_name(self) -> str:
        """Generate a unique run name using project name and timestamp.

        Returns:
            Run name string (e.g., "poker-ai_2026-03-28_14-32-45").
        """
        if not self.project_name:
            self.project_name = "poker-ai"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{self.project_name}_{timestamp}"
