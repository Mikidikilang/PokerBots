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
            env: RLCard environment instance (or wrapper with ._env.game)
        
        Returns:
            EnvStateSnapshot with deep copies of all mutable state
        """
        # Handle both wrapped and direct RLCard environments
        game_obj = env
        if hasattr(env, '_env'):
            # Wrapped environment (RLCardWrapper)
            game_obj = env._env
        
        # Deep copy game state
        game_state_copy = copy.deepcopy(game_obj.game if hasattr(game_obj, 'game') else game_obj)
        history_copy = copy.deepcopy(game_obj.history if hasattr(game_obj, 'history') else [])
        payoffs_copy = copy.deepcopy(game_obj.payoffs if hasattr(game_obj, 'payoffs') else [])
        
        # Get current player
        current_player = 0
        if hasattr(game_obj, 'get_player_num'):
            current_player = game_obj.get_player_num()
        
        # Get legal actions
        legal_actions = []
        if hasattr(game_obj, 'legal_actions'):
            legal_actions = list(game_obj.legal_actions)
        elif hasattr(env, 'get_legal_actions'):
            legal_actions = list(env.get_legal_actions())
        
        return cls(
            game_state=game_state_copy,
            current_player=current_player,
            legal_actions=legal_actions,
            history=history_copy,
            payoffs=payoffs_copy,
        )
    
    def restore(self, env: Any) -> None:
        """
        Restore RLCard environment to this snapshot state.
        
        Args:
            env: RLCard environment to restore into (or wrapper)
        """
        # Handle both wrapped and direct RLCard environments
        game_obj = env
        if hasattr(env, '_env'):
            game_obj = env._env
        
        # Restore game state
        if hasattr(game_obj, 'game'):
            game_obj.game = copy.deepcopy(self.game_state)
        else:
            # Direct game environment
            for attr_name in dir(self.game_state):
                if not attr_name.startswith('_') and not callable(getattr(self.game_state, attr_name, None)):
                    try:
                        setattr(game_obj, attr_name, copy.deepcopy(getattr(self.game_state, attr_name)))
                    except (AttributeError, TypeError):
                        pass
        
        # Restore history and payoffs
        if hasattr(game_obj, 'history'):
            game_obj.history = copy.deepcopy(self.history)
        if hasattr(game_obj, 'payoffs'):
            game_obj.payoffs = copy.deepcopy(self.payoffs)
        
        # Skip excessive debug logging during normal traversal
        # logger.debug("Restored env state")


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
                # Skip excessive debug logging
                # logger.debug("Savepoint exited")
    
    def get_depth(self) -> int:
        """Returns current nesting depth of savepoints."""
        return len(self.snapshot_stack)
