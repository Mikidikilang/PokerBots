"""Phase 2: Memory & Buffer Architecture for VR-DeepPDCFR+

================================================================================
ARCHITECTURAL OVERVIEW
================================================================================

The VR-DeepPDCFR+ algorithm requires two fundamentally different types of
experience buffers to achieve Nash convergence while avoiding "capacity-induced
reservoir dilution" (where early, noisy iterations contaminate later, accurate
strategy estimates).

BUFFER TYPES
============

1. EPHEMERAL ADVANTAGE BUFFER (B_V,i):
   - Lifespan: Single CFR iteration (batch of self-play games)
   - Wiped clean: At the START of each new iteration
   - Purpose: Feed the instantaneous advantage network (phi_i)
   - Data: (state_features, action_advantages, iteration_counter)
   - Why separate: Ensures advantage estimates reflect CURRENT strategy,
     not historical bias from earlier iterations

2. PERSISTENT STRATEGY BUFFER (B_Pi):
   - Lifespan: All iterations (entire training run)
   - Wiped clean: NEVER - data persists with time-decay weighting
   - Purpose: Feed the average strategy network (Pi)
   - Data: (state_features, action_strategy, iteration_stored, time_decay_weight)
   - Why time-decay: Later iterations have MORE information about true
     strategy, so weight recent data heavily (t^2 or t). This ensures
     the average strategy network converges to Nash equilibrium

TIME-DECAY MECHANICS
====================

When storing at iteration t:
  weight(t) = decay_fn(t) = t^p  (typically p=1 or p=2)

When sampling from iteration t in current iteration T:
  importance_weight = decay_fn(t) / decay_fn(T)

This ensures:
  - Early noisy data (t=1) has minimal influence
  - Recent high-quality data (t approximately T) has maximum influence
  - Smooth interpolation between iterations

CAPACITY & RESERVOIR SAMPLING
=============================

Both buffers have configurable capacity. When full:
  - Option 1: Uniform reservoir sampling (discard randomly)
  - Option 2: Priority-based sampling (keep recent data)

For strategy buffer: We prefer PRIORITY (keep recent, discard old)
For advantage buffer: Irrelevant since we wipe each iteration anyway

INTEGRATION WITH NEURAL NETWORKS
=================================

Advantage Network (phi_i):
  Input: state_features from EphemeralAdvantageBuffer
  Output: predicted advantages
  Loss: MSE between predicted and observed advantages
  
Strategy Network (Pi):
  Input: state_features from PersistentStrategyBuffer
  Output: predicted action probabilities
  Loss: Cross-entropy with time-decay weighted targets
  Note: Weights come from time_decay_weight field

---

References:
  - Koulis, Schvartzman et al. (2022): "VR-DeepPDCFR+" Section 3.2
  - Hart & Mas-Colell (2000): "A Simple Adaptive Procedure"
  - Lanctot et al. (2017): "A Unified Game-theoretic Approach to Multiagent RL"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Transition:
    """Single data point in a CFR buffer.
    
    Represents one complete sample: a decision state with computed advantages
    or strategy target, tagged with the iteration it was collected in.
    
    Attributes:
        infoset_features: State representation for neural network input
            Shape: (feature_dim,) - flattened for network
        action_probs: Counterfactual action probabilities or empirical strategy
            Shape: (num_actions,) - sum should be approximately 1.0
        advantages: Estimated action advantages (for advantage buffer)
            Shape: (num_actions,) or None for strategy samples
        legal_mask: Binary mask of legal actions (for behavior cloning masking)
            Shape: (num_actions,) - 1.0 for legal, 0.0 for illegal
            Used to prevent Π network from assigning probability to illegal actions
        iteration: CFR iteration when this data was generated (t >= 1)
        reach_prob: Player reach probability to this infoset
            Used for importance weighting in counterfactual computations
            Default: 1.0 (if not available)
            
    Notes:
        - action_probs and advantages should be numpy arrays
        - legal_mask should be binary np.ndarray (1.0 or 0.0)
        - iteration must be >= 1
        - reach_prob should be in (0, 1]
        - Frozen: immutable once created (safe for concurrent access)
    """
    infoset_features: np.ndarray
    action_probs: np.ndarray
    legal_mask: np.ndarray
    advantages: Optional[np.ndarray] = None
    iteration: int = 1
    reach_prob: float = 1.0
    
    def __post_init__(self):
        """Validate transition data."""
        # Check shapes
        if self.infoset_features.ndim != 1:
            raise ValueError(
                f"infoset_features must be 1D array, got shape {self.infoset_features.shape}"
            )
        if self.action_probs.ndim != 1:
            raise ValueError(
                f"action_probs must be 1D array, got shape {self.action_probs.shape}"
            )
        if self.legal_mask.ndim != 1:
            raise ValueError(
                f"legal_mask must be 1D array, got shape {self.legal_mask.shape}"
            )
        if self.advantages is not None and self.advantages.ndim != 1:
            raise ValueError(
                f"advantages must be 1D array, got shape {self.advantages.shape}"
            )
        
        num_actions = len(self.action_probs)
        if len(self.legal_mask) != num_actions:
            raise ValueError(
                f"legal_mask length {len(self.legal_mask)} != "
                f"action_probs length {num_actions}"
            )
        if self.advantages is not None and len(self.advantages) != num_actions:
            raise ValueError(
                f"advantages length {len(self.advantages)} != "
                f"action_probs length {num_actions}"
            )
        
        # Check legal_mask values (should be 0.0 or 1.0)
        unique_mask_values = np.unique(self.legal_mask)
        if not all(v in [0.0, 1.0] for v in unique_mask_values):
            raise ValueError(
                f"legal_mask must contain only 0.0 or 1.0, got {unique_mask_values}"
            )
        
        # Check iteration
        if self.iteration < 1:
            raise ValueError(f"iteration must be >= 1, got {self.iteration}")
        
        # Check reach probability
        if not 0 < self.reach_prob <= 1.0:
            raise ValueError(
                f"reach_prob must be in (0, 1], got {self.reach_prob}"
            )
        
        # Check action_probs sum (should be approximately 1.0, allow floating point error)
        prob_sum = np.sum(self.action_probs)
        if not 0.9 <= prob_sum <= 1.1:
            logger.warning(
                f"action_probs sum to {prob_sum:.4f}, expected approximately 1.0"
            )


# ============================================================================
# EPHEMERAL ADVANTAGE BUFFER
# ============================================================================

class EphemeralAdvantageBuffer:
    """Buffer for advantages in the current CFR iteration.
    
    This buffer stores advantage data ONLY for the current iteration.
    At the start of a new iteration, all data is discarded.
    
    Purpose: Feed instantaneous advantage network (phi_i) with fresh,
    unbiased advantage estimates from the current iteration's self-play.
    
    Key property: EPHEMERAL - data from previous iterations should NOT
    influence the advantage network, as it would bias toward old strategy.
    
    Attributes:
        capacity: Maximum number of transitions to store
        transitions: List of Transition objects
        feature_dim: Dimension of infoset features (inferred from first insert)
        
    Methods:
        insert: Add a transition to the buffer
        sample_minibatch: Sample random minibatch for training
        clear: Wipe all data (called at start of new iteration)
        size: Number of stored transitions
    """
    
    def __init__(self, capacity: int = 100_000):
        """Initialize ephemeral advantage buffer.
        
        Args:
            capacity: Maximum transitions to store before old data is discarded
        """
        self.capacity = capacity
        self.transitions: List[Transition] = []
        self.feature_dim: Optional[int] = None
    
    def insert(self, transition: Transition) -> None:
        """Add a transition to the buffer.
        
        If buffer is at capacity, discard a random old sample to make room.
        
        Args:
            transition: The Transition to add
            
        Raises:
            ValueError: If transition data is invalid
        """
        # Infer feature dimension from first insert
        if self.feature_dim is None:
            self.feature_dim = len(transition.infoset_features)
        
        # Validate consistency
        if len(transition.infoset_features) != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {len(transition.infoset_features)}"
            )
        
        self.transitions.append(transition)
        
        # Enforce capacity with random eviction (simple reservoir sampling)
        if len(self.transitions) > self.capacity:
            idx_to_remove = np.random.randint(0, len(self.transitions))
            self.transitions.pop(idx_to_remove)
    
    def sample_minibatch(
        self,
        batch_size: int,
        replace: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random minibatch from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            replace: Allow sampling with replacement (default: True)
            
        Returns:
            Tuple of (features, action_probs, advantages, iterations) where:
                - features: Shape (batch_size, feature_dim)
                - action_probs: Shape (batch_size, num_actions)
                - advantages: Shape (batch_size, num_actions)
                - iterations: Shape (batch_size,) - CFR iteration when each sample was generated
                
        Raises:
            ValueError: If buffer is empty or batch_size > buffer size
        """
        if len(self.transitions) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        actual_batch_size = min(batch_size, len(self.transitions))
        if not replace and batch_size > len(self.transitions):
            logger.warning(
                f"batch_size {batch_size} > buffer size {len(self.transitions)}, "
                f"sampling with replacement"
            )
            replace = True
        
        # Random sample indices
        indices = np.random.choice(
            len(self.transitions),
            size=actual_batch_size,
            replace=replace,
        )
        
        # Collect sampled transitions
        sampled = [self.transitions[i] for i in indices]
        
        # Stack into numpy arrays
        features = np.stack([t.infoset_features for t in sampled])
        action_probs = np.stack([t.action_probs for t in sampled])
        advantages = np.stack([t.advantages for t in sampled])
        iterations = np.array([t.iteration for t in sampled], dtype=np.int32)
        
        return features, action_probs, advantages, iterations
    
    def clear(self) -> None:
        """Wipe all transitions (called at start of new iteration).
        
        This is the CRITICAL OPERATION that prevents advantage network
        from being biased by old strategy iterations.
        """
        self.transitions = []
        logger.debug("EphemeralAdvantageBuffer cleared")
    
    def size(self) -> int:
        """Return number of stored transitions."""
        return len(self.transitions)
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return (f"EphemeralAdvantageBuffer(capacity={self.capacity}, "
                f"size={len(self.transitions)}, "
                f"feature_dim={self.feature_dim})")


# ============================================================================
# PERSISTENT STRATEGY BUFFER
# ============================================================================

class PersistentStrategyBuffer:
    """Buffer for strategy data across all CFR iterations.
    
    This buffer PERSISTS across iterations with time-decay weighting.
    Data is NEVER completely wiped, but older samples have less influence.
    
    Purpose: Feed average strategy network (Pi) with a balanced view of
    strategy evolution across training. Recent iterations have more accurate
    strategy estimates, so they receive higher weight.
    
    Time-Decay Mechanics:
        When storing at iteration t: weight(t) = t^p (p typically 1 or 2)
        When sampling: importance_weight = weight(t) / sum of all weights
        
    This ensures early noisy data (t approximately 1) has minimal impact while recent
    high-confidence data (t approximately T) dominates strategy network training.
    
    Attributes:
        capacity: Maximum number of transitions to store
        transitions: List of stored Transition objects
        iteration_stored: List of iterations when data was stored
        time_decay_power: Exponent p for weight(t) = t^p
        feature_dim: Dimension of infoset features
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        time_decay_power: float = 1.0,
    ):
        """Initialize persistent strategy buffer.
        
        Args:
            capacity: Maximum transitions to store
            time_decay_power: Exponent p for decay function t^p.
                p=1: Linear - recent iterations 2x heavier than early ones
                p=2: Quadratic - recent iterations 4x heavier (stronger decay)
        """
        self.capacity = capacity
        self.time_decay_power = time_decay_power
        self.transitions: List[Transition] = []
        self.iterations_stored: List[int] = []
        self.feature_dim: Optional[int] = None
    
    def insert(self, transition: Transition) -> None:
        """Add a transition to the buffer.
        
        When at capacity, remove the oldest sample (earliest iteration).
        This prioritizes keeping recent data.
        
        Args:
            transition: The Transition to add
        """
        # Infer feature dimension
        if self.feature_dim is None:
            self.feature_dim = len(transition.infoset_features)
        
        # Validate consistency
        if len(transition.infoset_features) != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {len(transition.infoset_features)}"
            )
        
        self.transitions.append(transition)
        self.iterations_stored.append(transition.iteration)
        
        # Enforce capacity: uniform random eviction (reservoir sampling)
        # This preserves the averaging property required for Nash convergence.
        # Time-decay weighting in sample_minibatch naturally down-weights old data.
        if len(self.transitions) > self.capacity:
            idx_to_remove = np.random.randint(0, len(self.transitions))
            self.transitions.pop(idx_to_remove)
            self.iterations_stored.pop(idx_to_remove)
    
    def sample_minibatch(
        self,
        batch_size: int,
        current_iteration: int,
        replace: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample minibatch with time-decay importance weighting.
        
        Recent data (current iteration) has weight 1.0.
        Older data has lower weight: weight(t) = (t / T)^p
        
        Args:
            batch_size: Number of transitions to sample
            current_iteration: Current CFR iteration (T)
            replace: Allow sampling with replacement
            
        Returns:
            Tuple of (features, action_probs, legal_masks, weights) where:
                - features: Shape (batch_size, feature_dim)
                - action_probs: Shape (batch_size, num_actions)
                - legal_masks: Shape (batch_size, num_actions) - binary masks
                - weights: Shape (batch_size,) - importance weights for loss
                
        Raises:
            ValueError: If buffer is empty
        """
        if len(self.transitions) == 0:
            raise ValueError("Cannot sample from empty strategy buffer")
        
        actual_batch_size = min(batch_size, len(self.transitions))
        if not replace and batch_size > len(self.transitions):
            logger.warning(
                f"batch_size {batch_size} > buffer size {len(self.transitions)}, "
                f"sampling with replacement"
            )
            replace = True
        
        # CRITICAL FIX: Use O(1) uniform sampling instead of O(N) weighted sampling.
        # OLD: Computed decay_weights for entire buffer (1M items!) on every call -> GPU starvation
        # NEW: Sample uniformly (O(1)), compute weights only for batch (O(batch_size))
        # Mathematical property: E_uniform[grad_L * w] ≈ E_proportional[grad_L]
        # The loss function will multiply cross-entropy by computed weights.
        indices = np.random.choice(
            len(self.transitions),
            size=actual_batch_size,
            replace=replace,
        )
        
        # Collect sampled transitions
        sampled = [self.transitions[i] for i in indices]
        
        # Compute time-decay weights ONLY for the sampled batch (O(batch_size))
        sampled_iterations = np.array([self.iterations_stored[i] for i in indices])
        sampled_weights = (sampled_iterations / current_iteration) ** self.time_decay_power
        
        # Normalize weights to average to 1.0 (for gradient stability)
        sampled_weights = sampled_weights / np.mean(sampled_weights)
        
        # Stack into arrays
        features = np.stack([t.infoset_features for t in sampled])
        action_probs = np.stack([t.action_probs for t in sampled])
        legal_masks = np.stack([t.legal_mask for t in sampled])
        
        return features, action_probs, legal_masks, sampled_weights
    
    def size(self) -> int:
        """Return number of stored transitions."""
        return len(self.transitions)
    
    def oldest_iteration(self) -> Optional[int]:
        """Return the oldest iteration stored in the buffer."""
        if len(self.transitions) == 0:
            return None
        return min(self.iterations_stored)
    
    def newest_iteration(self) -> Optional[int]:
        """Return the newest iteration stored in the buffer."""
        if len(self.transitions) == 0:
            return None
        return max(self.iterations_stored)
    
    def get_iteration_distribution(self) -> dict:
        """Return histogram of iterations stored in buffer.
        
        Useful for monitoring buffer composition over training.
        
        Returns:
            Dict mapping iteration -> count
        """
        from collections import Counter
        return dict(Counter(self.iterations_stored))
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        oldest = self.oldest_iteration()
        newest = self.newest_iteration()
        return (f"PersistentStrategyBuffer(capacity={self.capacity}, "
                f"size={len(self.transitions)}, "
                f"iterations=[{oldest}-{newest}], "
                f"decay_power={self.time_decay_power})")


# ============================================================================
# BUFFER MANAGEMENT UTILITIES
# ============================================================================

class BufferManager:
    """Convenience wrapper managing both advantage and strategy buffers.
    
    Handles the iteration lifecycle:
      1. Start iteration -> clear ephemeral buffer
      2. Collect data -> insert into both buffers
      3. Train networks -> sample from each buffer
      4. End iteration -> increment counter
      
    Attributes:
        advantage_buffer: EphemeralAdvantageBuffer instance
        strategy_buffer: PersistentStrategyBuffer instance
        current_iteration: Current CFR iteration
    """
    
    def __init__(
        self,
        advantage_capacity: int = 100_000,
        strategy_capacity: int = 1_000_000,
        time_decay_power: float = 1.0,
    ):
        """Initialize buffer manager.
        
        Args:
            advantage_capacity: Max size of advantage buffer
            strategy_capacity: Max size of strategy buffer
            time_decay_power: Time decay exponent for strategy buffer
        """
        self.advantage_buffer = EphemeralAdvantageBuffer(
            capacity=advantage_capacity
        )
        self.strategy_buffer = PersistentStrategyBuffer(
            capacity=strategy_capacity,
            time_decay_power=time_decay_power,
        )
        self.current_iteration = 1
    
    def start_iteration(self) -> None:
        """Mark start of new CFR iteration.
        
        Clears ephemeral advantage buffer (critical).
        Called at the beginning of each iteration's self-play.
        """
        self.advantage_buffer.clear()
        logger.debug(f"Started iteration {self.current_iteration}")
    
    def add_transition(
        self,
        infoset_features: np.ndarray,
        action_probs: np.ndarray,
        legal_mask: np.ndarray,
        advantages: np.ndarray,
        reach_prob: float = 1.0,
    ) -> None:
        """Add a transition to both buffers.
        
        Args:
            infoset_features: State representation
            action_probs: Action probabilities
            legal_mask: Binary mask of legal actions (1.0 for legal, 0.0 for illegal)
            advantages: Action advantages
            reach_prob: Reach probability (default 1.0)
        """
        transition = Transition(
            infoset_features=infoset_features,
            action_probs=action_probs,
            legal_mask=legal_mask,
            advantages=advantages,
            iteration=self.current_iteration,
            reach_prob=reach_prob,
        )
        
        self.advantage_buffer.insert(transition)
        self.strategy_buffer.insert(transition)
    
    def end_iteration(self) -> None:
        """Mark end of CFR iteration, prepare for next.
        
        Increments iteration counter.
        """
        self.current_iteration += 1
        logger.debug(
            f"Ended iteration. Next will be {self.current_iteration}. "
            f"Advantage buffer size: {self.advantage_buffer.size()}, "
            f"Strategy buffer size: {self.strategy_buffer.size()}"
        )
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return (f"BufferManager(iteration={self.current_iteration}, "
                f"{self.advantage_buffer}, {self.strategy_buffer})")
