"""
MLOps Module (mlops/).

Core infrastructure for training management, state persistence, and monitoring.

Currently includes:
  - state_manager.py: Atomic checkpoint save/load with RNG state capture
  - fault_tolerance.py: Preemption detection and handler
  - hf_sync.py: HuggingFace Hub synchronization
  - monitoring.py: Weights & Biases integration with fail-safe handling
"""

from __future__ import annotations

from src.mlops.monitoring import WandbMonitor
from src.mlops.state_manager import CheckpointManager, RNGStateManager, StateManager

__all__ = [
    "WandbMonitor",
    "CheckpointManager",
    "RNGStateManager",
    "StateManager",
]
