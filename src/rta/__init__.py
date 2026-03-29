"""
Real-Time Assistance (RTA) Module — Live Game Integration.

The RTA module connects the trained neural network to live poker environments
(online platforms, broadcast streams, or custom game logs). It handles:

1. Game state parsing:      Raw poker events → normalized state dicts
2. Inference:               State → action decision + confidence
3. Result logging:          Decision history for analysis/calibration

Usage (online):
    parser = LiveGameStateBuilder(num_players=6, initial_stack=200*BB, big_blind=2)
    engine = RTAInferenceEngine(checkpoint_path="checkpoints/best_model.pt")
    
    for event in live_event_stream:
        parser.process_event(event)
        state = parser.get_state()
        decision, confidence = engine.get_decision(state)
        # Use decision to guide play...

Critical constraint:
    NO RLCard dependencies in inference_engine.py (must be deployable standalone).
"""

from __future__ import annotations

from src.rta.game_state_parser import LiveGameStateBuilder
from src.rta.inference_engine import RTAInferenceEngine

__all__ = [
    "LiveGameStateBuilder",
    "RTAInferenceEngine",
]
