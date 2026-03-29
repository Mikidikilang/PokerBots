"""
Environment State Save/Restore for CFR Tree Traversal (cfr_env_state.py).

[PHASE 2] Game state persistence during MCCFR tree exploration.

PROBLEM
-------

Traditional MCCFR requires the ability to:
    1. Step environment forward (action a → new state)
    2. Recursively evaluate subtrees from new state
    3. Undo action, restore original state
    4. Try next action from original state

RLCard provides step() but NOT explicit undo(). Instead, we must:
    - Deep-copy the environment's game state before each action
    - Restore by replacing with saved copy

This module implements:
    - EnvStateSnapshot: captures complete game state via deep copy
    - EnvStateSavepoint: context manager (copy-on-enter, restore-on-exit)
    - EnvStateManager: manages snapshot stack during traversal

ALGORITHM
---------

    for each action a in legal_actions:
        with env.savepoint():  # Saves state
            obs, reward, done, info = env.step(a)
            # Recursively evaluate subtree
            value = traverse(state_prime, ...)
            # On __exit__, state auto-restores
        # Now back at original state, continue to next action

References:
    - Lanctot et al. (2009): "MCCFR traversal with external sampling"
    - Deep CFR tree search implementations (RL frameworks)
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

logger = logging.getLogger(__name__)


@dataclass
class EnvStateSnapshot:
    """Complete capture of RLCard game state via deep copy."""
    
    game_state: Any        # Deep copy of env._env.game (or equivalent)
    current_player: int    # Whose turn it is
    legal_actions: list[int]
    history: list[tuple[int, int]]  # (player, action) pairs
    payoffs: list[float]   # Terminal payoffs (if game ended)
    
    @classmethod
    def capture(cls, env: Any) -> EnvStateSnapshot:
        """
        Capture complete game state from RLCard environment.
        
        Args:
            env: RLCard environment instance
        
        Returns:
            EnvStateSnapshot with deep copies of all mutable state
        """
        # RLCard structure: env._env.game contains game-specific state
        game_state_copy = copy.deepcopy(env._env.game)
        history_copy = copy.deepcopy(env._env.history)
        payoffs_copy = copy.deepcopy(env._env.payoffs)
        
        return cls(
            game_state=game_state_copy,
            current_player=env._env.game.get_player_num() if hasattr(env._env.game, 'get_player_num') else 0,
            legal_actions=list(env._env.legal_actions) if hasattr(env._env, 'legal_actions') else [],
            history=history_copy,
            payoffs=payoffs_copy,
        )
    
    def restore(self, env: Any) -> None:
        """
        Restore RLCard environment to this snapshot state.
        
        Args:
            env: RLCard environment to restore into
        """
        env._env.game = copy.deepcopy(self.game_state)
        env._env.history = copy.deepcopy(self.history)
        env._env.payoffs = copy.deepcopy(self.payoffs)
        logger.debug(
            "Restored env state: current_player=%s, legal_actions=%d",
            self.current_player, len(self.legal_actions)
        )


class EnvStateManager:
    """Manages save/restore stack during tree traversal."""
    
    def __init__(self, env: Any):
        """
        Args:
            env: RLCard environment to manage
        """
        self.env = env
        self.snapshot_stack: list[EnvStateSnapshot] = []
    
    @contextmanager
    def savepoint(self) -> Generator[EnvStateSnapshot, None, None]:
        """
        Context manager: save state on enter, restore on exit.
        
        Usage:
            with env_manager.savepoint() as snapshot:
                env.step(action)
                # ... traverse subtree ...
            # State auto-restored on exit
        
        Yields:
            EnvStateSnapshot of state before context
        """
        snapshot = EnvStateSnapshot.capture(self.env)
        self.snapshot_stack.append(snapshot)
        
        try:
            yield snapshot
        finally:
            # Pop and restore
            if self.snapshot_stack:
                restored = self.snapshot_stack.pop()
                restored.restore(self.env)
                logger.debug(
                    "Savepoint exited: restored to snapshot (depth=%d)",
                    len(self.snapshot_stack)
                )
    
    def get_depth(self) -> int:
        """Returns current nesting depth of savepoints."""
        return len(self.snapshot_stack)
