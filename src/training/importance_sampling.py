"""
Importance Sampling Correction for CFR Buffer (Phase 3)

[PHASE 3] Corrects for visit frequency bias in reservoir sampling.

Problem:
    When sampling trajectories from a buffer filled during self-play, recent 
    game states have higher visit frequency (more stored trajectories).
    
    Example: If hero learns to always play fold vs weak hands, the dataset 
    becomes dominated by fold actions. Sampling uniformly biases learning 
    toward overfitting on fold scenarios.

Solution: Importance Sampling Weights
    1. Track visit count per state during collection: n(s)
    2. When sampling from buffer, weight by: w(s) = 1 / n(s)
    3. Normalize weights to sum to 1
    4. Weight loss: L = Σ_i w_i * |value_pred - value_target|
    
    This gives equal expected gradient to all states, regardless of visit frequency.

References:
    - Precup et al. (2000): "Eligibility Traces for Off-Policy Learning"
    - Silver et al. (2017): "Mastering the Game of Go without Human Knowledge"
    - Pohlen et al. (2016): "Observe and Look Further: Achieving Consistent 
      Performance on Atari"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class StateVisitTracker:
    """Track visit frequencies for importance sampling correction."""
    
    visit_counts: dict[str, int] = field(default_factory=dict)
    """state_hash -> visit count during data collection"""
    
    total_visits: int = 0
    """Total state visits across entire buffer"""
    
    max_visits: int = 0
    """Maximum visit count (for normalization)"""
    
    def record_visit(self, state_hash: str):
        """Record a state visit during trajectory collection."""
        if state_hash not in self.visit_counts:
            self.visit_counts[state_hash] = 0
        
        self.visit_counts[state_hash] += 1
        self.total_visits += 1
        self.max_visits = max(self.max_visits, self.visit_counts[state_hash])
    
    def get_importance_weight(self, state_hash: str) -> float:
        """
        Get importance weight for a state.
        
        w(s) = 1 / n(s), normalized
        
        Args:
            state_hash: State identifier
        
        Returns:
            Importance weight in (0, 1]
        """
        if state_hash not in self.visit_counts or self.visit_counts[state_hash] == 0:
            return 1.0  # Unseen state: uniform weight
        
        count = self.visit_counts[state_hash]
        # Inverse visit frequency (states visited less often get higher weight)
        weight = 1.0 / count
        
        return weight
    
    def get_importance_weights_batch(
        self,
        state_hashes: list[str],
    ) -> torch.Tensor:
        """
        Get importance weights for a batch of states.
        
        Args:
            state_hashes: List of state identifiers
        
        Returns:
            torch.Tensor [batch_size] of weights, normalized to sum to batch_size
        """
        weights = np.array(
            [self.get_importance_weight(h) for h in state_hashes],
            dtype=np.float32,
        )
        
        # Normalize to sum to batch size (preserve expected gradient magnitude)
        if len(state_hashes) > 0:
            weights = weights / weights.sum() * len(state_hashes)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def reset(self):
        """Clear tracker for next collection phase."""
        self.visit_counts.clear()
        self.total_visits = 0
        self.max_visits = 0
    
    def get_stats(self) -> dict:
        """Get statistics about visit distribution."""
        if not self.visit_counts:
            return {
                'unique_states': 0,
                'total_visits': 0,
                'avg_visits_per_state': 0.0,
                'max_visits': 0,
            }
        
        counts = list(self.visit_counts.values())
        return {
            'unique_states': len(counts),
            'total_visits': self.total_visits,
            'avg_visits_per_state': np.mean(counts),
            'max_visits': self.max_visits,
            'std_visits': np.std(counts),
        }


@dataclass
class ImportanceSampledBufferWrapper:
    """Wraps RolloutBuffer with importance sampling correction."""
    
    buffer: any  # RolloutBuffer instance
    visit_tracker: StateVisitTracker = field(default_factory=StateVisitTracker)
    enabled: bool = True
    
    def add_with_importance(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
        state_hash: Optional[str] = None,
    ):
        """
        Add to buffer and track visit frequency.
        
        Args:
            observation: State observation
            action: Action taken
            reward: Reward received
            log_prob: Log probability under policy
            value: Value estimate
            done: Episode termination
            state_hash: Optional state hash for tracking. 
                       If None, generated from observation.
        """
        # Generate state hash if not provided
        if state_hash is None and self.enabled:
            state_hash = self._hash_observation(observation)
        
        # Record visit
        if self.enabled and state_hash:
            self.visit_tracker.record_visit(state_hash)
        
        # Add to buffer normally
        self.buffer.add(observation, action, reward, log_prob, value, done)
    
    def get_mini_batches_with_weights(self):
        """
        Get mini-batches with importance weights applied.
        
        Yields:
            dict with keys:
                - All original batch keys from buffer
                - 'importance_weights': torch.Tensor [batch_size]
        """
        # Get original batches
        for batch in self.buffer.get_mini_batches():
            if self.enabled and self.visit_tracker.visit_counts:
                # Generate state hashes for observation batch
                state_hashes = [
                    self._hash_observation(obs)
                    for obs in batch['observations'].values()
                ]
                
                # Get importance weights
                weights = self.visit_tracker.get_importance_weights_batch(state_hashes)
                batch['importance_weights'] = weights
            else:
                # No importance sampling: uniform weights
                batch_size = len(batch['actions'])
                batch['importance_weights'] = torch.ones(batch_size, dtype=torch.float32)
            
            yield batch
    
    def _hash_observation(self, obs: dict[str, torch.Tensor]) -> str:
        """Generate hash of observation for visit tracking."""
        # Simple hash: concat tensor values and hash
        try:
            obs_str = ''.join(
                f"{k}:{v.sum().item()}"
                for k, v in sorted(obs.items())
            )
            import hashlib
            hash_obj = hashlib.sha256(obs_str.encode())
            return hash_obj.hexdigest()
        except Exception:
            return "unknown"
    
    def get_visit_stats(self) -> dict:
        """Get statistics about state visit distribution."""
        return self.visit_tracker.get_stats()
    
    def reset_tracking(self):
        """Reset visit tracker for next collection phase."""
        self.visit_tracker.reset()


# ============================================================================
# Loss Function with Importance Sampling
# ============================================================================

def compute_loss_with_importance_weights(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    importance_weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute loss with importance sampling correction.
    
    Args:
        predictions: Predicted values [batch_size, ...]
        targets: Target values [batch_size, ...]
        importance_weights: Optional weights [batch_size]. If None, uniform.
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Weighted loss scalar
    """
    # MSE per-sample
    loss_per_sample = (predictions - targets) ** 2
    
    if importance_weights is None:
        importance_weights = torch.ones_like(loss_per_sample[:, 0] if loss_per_sample.dim() > 1 else loss_per_sample)
    
    # Apply weights
    weighted_loss = importance_weights * loss_per_sample
    
    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:
        return weighted_loss


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test visit tracking
    tracker = StateVisitTracker()
    
    # Simulate repeated visits to same states
    states = ['state_A', 'state_B', 'state_C'] * 10  # state_A visited 10x, etc
    for state in states:
        tracker.record_visit(state)
    
    print("Visit Statistics:")
    print(tracker.get_stats())
    
    print("\nImportance Weights:")
    for state in set(states):
        weight = tracker.get_importance_weight(state)
        print(f"  {state}: weight = {weight:.4f}")
    
    print("\nBatch Weights (normalized):")
    batch = ['state_A', 'state_B', 'state_A', 'state_C']
    weights = tracker.get_importance_weights_batch(batch)
    print(f"  Batch: {batch}")
    print(f"  Weights: {weights}")
    print(f"  Sum: {weights.sum().item():.4f} (should ≈ {len(batch)})")
