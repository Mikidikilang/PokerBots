"""
Evaluation Module (evaluation/).

Implements benchmarking and evaluation tools for the PokerAI agent.

Currently includes:
  - acpc_client.py: TCP socket-based ACPC protocol client for Slumbot
  - benchmark_runner.py: Orchestrates HUNL matches and tracks win rate
  - nash_evaluator.py: Local Best Response oracle for exploitability estimation
"""

from __future__ import annotations

from src.evaluation.acpc_client import AcpcClient, MatchState
from src.evaluation.benchmark_runner import SlumbotEvaluator, SlumbotEvalConfig
from src.evaluation.nash_evaluator import (
    LocalBestResponseEvaluator,
    NashEvalConfig,
    NashEvalResults,
)

__all__ = [
    "AcpcClient",
    "MatchState",
    "SlumbotEvaluator",
    "SlumbotEvalConfig",
    "LocalBestResponseEvaluator",
    "NashEvalConfig",
    "NashEvalResults",
]
