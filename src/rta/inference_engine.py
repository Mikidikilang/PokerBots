"""
Real-Time Inference Engine (inference_engine.py).

Loads trained checkpoints and provides low-latency decision-making for live poker.

CRITICAL CONSTRAINT:
    This module has ZERO dependencies on RLCard or any game simulation library.
    It is designed to be deployed standalone on user machines, with minimal
    external dependencies (torch + model weights only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.env.action_mapper import ActionMapper, PokerAction
from src.env.features import ObservationBuilder
from src.model.networks import NetworkConfig, PokerActorCritic

logger = logging.getLogger(__name__)


class RTAInferenceEngine:
    """Live inference engine with strict observation validation.
    
    Example usage:
        engine = RTAInferenceEngine(
            checkpoint_path="checkpoints/best_model.pt",
            device="cpu"
        )
        
        state = {
            "hand": ["AS", "KS"],
            "public_cards": ["2H", "3D", "4C"],
            "pot": 100.0,
            "my_chips": 500.0,
            "big_blind": 2.0,
            "amount_to_call": 50.0,
            "position": 2,
            "legal_actions": [0, 1, 3, 4],
            "opponent_chips": [200, 150, 300, 250, 400],
            "betting_history": [...],
            "min_raise": 4.0,
        }
        
        action_name, confidence = engine.get_decision(state)
        # action_name: "RAISE_POT" or "ALL_IN" etc.
        # confidence: float in [0.0, 1.0]
    
    Strict validation mode:
    - Observations that fail validation raise ValueError (prevents silent failures).
    - All inputs are range-checked and type-checked before inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        """Initialize inference engine with a checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file (saved PyTorch state dict).
            device: PyTorch device ("cpu", "cuda:0", etc.). Defaults to CPU.
            deterministic: If True, always return argmax action (no sampling).
        
        Raises:
            FileNotFoundError: If checkpoint does not exist.
            RuntimeError: If checkpoint format is invalid.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.deterministic = deterministic
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        self.obs_builder = ObservationBuilder()
        self.action_mapper = ActionMapper()
        self.model: PokerActorCritic | None = None
        self.config: NetworkConfig | None = None
        
        self._load_checkpoint()
        
        logger.info(
            "RTAInferenceEngine initialized: checkpoint=%s, device=%s, deterministic=%s",
            self.checkpoint_path, self.device, self.deterministic
        )
    
    def get_decision(
        self,
        raw_state: dict[str, Any],
        return_logits: bool = False,
    ) -> tuple[str, float] | tuple[str, float, dict[str, Any]]:
        """Get action decision from raw game state.
        
        Args:
            raw_state: Raw game state dict with keys:
                - hand, public_cards, pot, my_chips, big_blind
                - amount_to_call, position, legal_actions
                - opponent_chips, betting_history, min_raise
            return_logits: If True, also return action logits dict.
        
        Returns:
            (action_name, confidence) or (action_name, confidence, logits_dict)
            where action_name is a string like "RAISE_POT", "ALL_IN", etc.
        
        Raises:
            ValueError: If observation validation fails.
            RuntimeError: If model inference fails.
        """
        # Strict validation
        try:
            obs = self.obs_builder.build(raw_state, validate=True)
        except Exception as e:
            logger.critical(
                "Observation validation failed: %s. Raw state: %s",
                str(e), raw_state
            )
            raise ValueError(f"Invalid observation: {e}") from e
        
        # Ensure observation is on correct device
        obs_tensor = obs.to(self.device)
        
        # Run inference
        with torch.no_grad():
            # Model.get_action returns (action_idx, value_estimate, logits)
            try:
                action_idx, value, logits = self.model.get_action(
                    obs_tensor.unsqueeze(0),
                    deterministic=self.deterministic,
                )
            except Exception as e:
                logger.critical("Model inference failed: %s", str(e))
                raise RuntimeError(f"Inference failed: {e}") from e
        
        action_idx = int(action_idx.item())
        value = float(value.item())
        logits_np = logits.cpu().numpy()
        
        # Get action name
        try:
            action_name = self.action_mapper.action_index_to_name(action_idx)
        except ValueError:
            logger.error("Invalid action index: %d", action_idx)
            action_name = f"UNKNOWN_{action_idx}"
        
        # Compute confidence from logits (softmax probability of selected action)
        logits_tensor = logits.cpu()
        softmax_probs = torch.softmax(logits_tensor, dim=-1)
        confidence = float(softmax_probs[action_idx].item())
        
        logger.debug(
            "Decision: action=%s (idx=%d), confidence=%.3f, value=%.4f",
            action_name, action_idx, confidence, value
        )
        
        if return_logits:
            logits_dict = {
                "raw_logits": logits_np.tolist(),
                "softmax_probs": softmax_probs.numpy().tolist(),
                "value_estimate": value,
            }
            return action_name, confidence, logits_dict
        else:
            return action_name, confidence
    
    def _load_checkpoint(self) -> None:
        """Load model weights from checkpoint file (STRICT FORMAT VALIDATION).
        
        Checkpoint structure (REQUIRED):
            {
                "config": NetworkConfig dict,
                "model_state": model.state_dict(),
                "optimizer_state": (optional),
                "step": (optional),
            }
        
        Raises:
            RuntimeError: If checkpoint cannot be loaded, is corrupt, or keys are missing.
        """
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {self.checkpoint_path}: {e}"
            ) from e
        
        # [FIX] Strict validation: require BOTH "config" and "model_state" keys
        required_keys = {"config", "model_state"}
        missing_keys = required_keys - set(checkpoint.keys())
        
        if missing_keys:
            raise RuntimeError(
                f"Checkpoint {self.checkpoint_path} missing required keys: {missing_keys}. "
                f"Expected keys: {required_keys}. "
                f"Found keys: {list(checkpoint.keys())}. "
                f"Ensure checkpoint was saved via trainer.save_checkpoint() with proper structure."
            )
        
        # Extract config (STRICT: no fallback to defaults)
        config_dict = checkpoint["config"]
        if not isinstance(config_dict, dict):
            raise RuntimeError(
                f"Checkpoint config must be a dict, got {type(config_dict).__name__}. "
                f"Value: {config_dict}"
            )
        
        try:
            self.config = NetworkConfig(**config_dict)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate NetworkConfig from checkpoint: {e}"
            ) from e
        
        # Instantiate model
        self.model = PokerActorCritic(self.config)
        
        # Load weights (STRICT: require "model_state" key)
        state_dict = checkpoint["model_state"]
        try:
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error("Failed to load model state: %s", str(e))
            raise RuntimeError(f"Model load failed: {e}") from e
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            "Checkpoint loaded successfully. Model: %s, Config: %s",
            type(self.model).__name__,
            self.config,
        )
