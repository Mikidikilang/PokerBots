"""
Strategy Buffer & Average Strategy Network (cfr_strategy.py).

[PHASE 2C] Strategy memory and supervised learning head for behavioral cloning.

AVERAGE STRATEGY LEARNING
--------------------------

In traditional CFR, we compute the average strategy:
    σ̄_i(a|h) = (1/T) Σ_{t=1}^T σ^t_i(a|h)
    
where σ^t(a|h) = regret-matched strategy at iteration t.

This average strategy converges to Nash equilibrium over sufficient iterations.

In Deep CFR, we train a neural network to approximate this average strategy:
    network: observation → action_probabilities (via softmax)
    training: behavioral cloning (supervised learning on ground-truth σ̄)
    
STRATEGY BUFFER
---------------

Stores (observation, action_taken, strategy_probability) tuples weighted by iteration.
Later iterations have higher weight (they're closer to Nash equilibrium).

Vitter's reservoir sampling ensures uniform replacement (similar to RegretBuffer).

INFERENCE
---------

At inference time (playing against real opponents):
    1. Evaluate observation with average strategy network
    2. Sample action proportional to predicted probabilities
    3. Play that action
    
This is "online" play using the learned average strategy.

For superhuman performance (Phase 4), we'll add real-time subgame solving,
but the strategy network is the foundation.

---

References:
    - Hart & Mas-Colell (1999): "A Simple Adaptive Procedure..."
    - Lanctot et al. (2009): "An Introduction to CFR"
    - Brown & Sandholm (2019): "Solving Imperfect Information Games"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class StrategySample:
    """Single strategy sample for behavioral cloning."""
    
    infoset_id: str
    observation: torch.Tensor              # [obs_dim]
    legal_actions: list[int]               # Valid actions at this state
    action_probabilities: dict[int, float] # {action_idx: probability}
    iteration: int                          # Which CFR iteration (for weighting)
    
    def to_network_targets(self, num_actions: int = 12) -> torch.Tensor:
        """
        Convert action probabilities to target distribution.
        
        Output: tensor[num_actions] where:
            - illegal actions: 0 (cross-entropy will ignore via softmax)
            - legal actions: actual probability values
            - must sum to 1.0 (normalized)
        
        Args:
            num_actions: Total number of possible actions
        
        Returns:
            torch.Tensor of shape [num_actions] normalized to sum to 1.0
        """
        targets = torch.zeros(num_actions, dtype=torch.float32)
        
        total_prob = 0.0
        for action_idx, prob in self.action_probabilities.items():
            if action_idx in self.legal_actions:
                targets[action_idx] = float(prob)
                total_prob += prob
        
        # Normalize (should already be normalized, but ensure)
        if total_prob > 0:
            targets = targets / total_prob
        else:
            # Uniform if no probabilities
            for action in self.legal_actions:
                targets[action] = 1.0 / len(self.legal_actions)
        
        return targets


class StrategyBuffer:
    """
    Reservoir sampling buffer for average strategy learning.
    
    Stores (observation, action_probabilities) pairs for behavioral cloning.
    
    Key difference from RegretBuffer:
        - Stores probability distributions (not scalar regrets)
        - Can weight by iteration (later iterations more similar to Nash)
        - Used for supervised learning (predicting action probabilities)
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        num_actions: int = 12,
    ):
        """
        Args:
            buffer_size: Maximum number of strategy samples
            num_actions: Number of discrete actions
        """
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        
        self.samples: list[StrategySample] = []
        self.reservoir_count = 0  # Samples seen (for uniform probability)
        
        logger.info(f"StrategyBuffer initialized: size={buffer_size}, actions={num_actions}")
    
    def add_sample(
        self,
        infoset_id: str,
        observation: torch.Tensor,
        legal_actions: list[int],
        action_probabilities: dict[int, float],
        iteration: int,
    ) -> None:
        """
        Add strategy sample via reservoir sampling.
        
        Args:
            infoset_id: Information set identifier
            observation: Observation tensor [obs_dim]
            legal_actions: List of legal action indices
            action_probabilities: {action: probability}
            iteration: CFR iteration (for potential weighting)
        """
        sample = StrategySample(
            infoset_id=infoset_id,
            observation=observation.clone().detach(),
            legal_actions=legal_actions,
            action_probabilities=action_probabilities,
            iteration=iteration,
        )
        
        # Reservoir sampling
        j = np.random.randint(0, self.reservoir_count + 1)
        
        if j < self.buffer_size:
            if len(self.samples) < self.buffer_size:
                self.samples.append(sample)
            else:
                self.samples[j] = sample
        
        self.reservoir_count += 1
        
        if self.reservoir_count % 1000 == 0:
            logger.debug(
                f"StrategyBuffer: {self.reservoir_count} samples seen, "
                f"{len(self.samples)} in buffer"
            )
    
    def sample_batch(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> dict[str, torch.Tensor] | None:
        """
        Sample mini-batch for network training.
        
        Args:
            batch_size: Number of samples per batch
            device: PyTorch device
        
        Returns:
            {
                "observations": tensor[batch_size, obs_dim],
                "targets": tensor[batch_size, num_actions],
                "legal_action_masks": tensor[batch_size, num_actions],
            }
            or None if buffer is too small
        """
        if len(self.samples) < batch_size:
            logger.warning(f"Buffer too small: {len(self.samples)} < {batch_size}")
            return None
        
        # Sample indices uniformly
        indices = np.random.choice(len(self.samples), size=batch_size, replace=False)
        
        obs_list = []
        target_list = []
        mask_list = []
        
        for idx in indices:
            sample = self.samples[idx]
            
            # Observation
            obs_list.append(sample.observation)
            
            # Target probabilities
            targets = sample.to_network_targets(self.num_actions)
            target_list.append(targets)
            
            # Legal action mask
            mask = torch.zeros(self.num_actions, dtype=torch.bool)
            for action in sample.legal_actions:
                mask[action] = True
            mask_list.append(mask)
        
        batch = {
            "observations": torch.stack(obs_list).to(device),
            "targets": torch.stack(target_list).to(device),
            "legal_action_masks": torch.stack(mask_list).to(device),
        }
        
        return batch
    
    def get_summary(self) -> dict[str, any]:
        """Return buffer statistics."""
        return {
            "buffer_size": self.buffer_size,
            "samples_in_buffer": len(self.samples),
            "total_samples_seen": self.reservoir_count,
            "fill_ratio": len(self.samples) / self.buffer_size,
        }


class AverageStrategyNetwork(nn.Module):
    """
    [BEHAVIORAL CLONING HEAD]
    
    Learns to predict the average (Nash) strategy via supervised learning.
    
    Input:  observation (flattened state)
    Output: action_probabilities (softmax over 12 actions)
    
    Training: Cross-entropy loss on ground-truth average strategy
    
    Inference: Sample actions proportional to predicted probabilities
    
    This is the network you actually use to PLAY the game at inference time.
    The regret network is only for computing regrets during training.
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 12,
        hidden_dim: int = 512,
    ):
        """
        Args:
            obs_dim: Observation dimension
            num_actions: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        
        # MLP for strategy prediction
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            # Note: NO softmax here - applies in loss function
        )
        
        logger.info(
            f"AverageStrategyNetwork: obs_dim={obs_dim}, "
            f"actions={num_actions}, hidden={hidden_dim}"
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, obs_dim]
        
        Returns:
            logits: [batch_size, num_actions] (before softmax)
        """
        return self.mlp(observations)
    
    def get_action_probabilities(
        self,
        observation: torch.Tensor,
        legal_actions: list[int] | None = None,
    ) -> dict[int, float]:
        """
        Get strategy probabilities for a single observation (inference).
        
        Args:
            observation: Single obs tensor [obs_dim] or [1, obs_dim]
            legal_actions: If provided, zero out illegal actions
        
        Returns:
            {action_idx: probability} for legal actions only
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(observation)  # [1, num_actions]
            
            # Apply softmax
            probs = F.softmax(logits[0], dim=0)  # [num_actions]
            
            # Convert to dict
            strategy = {}
            for action_idx in range(self.num_actions):
                prob = float(probs[action_idx].item())
                
                # Only include legal actions
                if legal_actions is None or action_idx in legal_actions:
                    strategy[action_idx] = prob
            
            # Renormalize to sum to 1.0 (in case we filtered illegal)
            total = sum(strategy.values())
            if total > 0:
                strategy = {a: p / total for a, p in strategy.items()}
            
            return strategy


class StrategyNetworkTrainer:
    """
    Trains AverageStrategyNetwork via behavioral cloning.
    
    Loss: Cross-entropy between predicted and target (average) strategy.
    
    This is standard supervised learning (not RL).
    """
    
    def __init__(
        self,
        network: AverageStrategyNetwork,
        strategy_buffer: StrategyBuffer,
        learning_rate: float = 1e-4,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            network: AverageStrategyNetwork to train
            strategy_buffer: StrategyBuffer with samples
            learning_rate: Optimizer learning rate
            device: PyTorch device
        """
        self.network = network.to(device)
        self.strategy_buffer = strategy_buffer
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )
        
        self.total_updates = 0
    
    def train_epoch(
        self,
        batch_size: int = 32,
        num_batches: int = 100,
    ) -> dict[str, float]:
        """
        Train for one epoch on mini-batches from buffer.
        
        Args:
            batch_size: Size of each mini-batch
            num_batches: Number of batches to process
        
        Returns:
            {'loss': average_loss, 'updates': num_batches}
        """
        self.network.train()
        total_loss = 0.0
        
        for batch_idx in range(num_batches):
            # Sample mini-batch
            batch = self.strategy_buffer.sample_batch(batch_size, self.device)
            
            if batch is None:
                logger.warning("Buffer empty, skipping epoch")
                return {"loss": float('nan'), "updates": 0}
            
            # Forward pass
            obs = batch["observations"]
            targets = batch["targets"]
            masks = batch["legal_action_masks"]
            
            logits = self.network(obs)
            
            # Cross-entropy loss on legal actions
            loss = self._masked_cross_entropy_loss(logits, targets, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.total_updates += 1
        
        avg_loss = total_loss / num_batches
        
        logger.info(
            f"StrategyNetworkTrainer Epoch: "
            f"loss={avg_loss:.6f}, total_updates={self.total_updates}"
        )
        
        return {
            "loss": avg_loss,
            "updates": num_batches,
        }
    
    def _masked_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss computed only on legal actions.
        
        Args:
            logits: [batch, num_actions] (before softmax)
            targets: [batch, num_actions] (probability distribution)
            legal_masks: [batch, num_actions] (True for legal actions)
        
        Returns:
            Scalar loss tensor
        """
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=1)  # [batch, num_actions]
        
        # KL divergence: sum(target * (log(target) - log(pred)))
        # Simplified: sum(target * (-log(pred)))
        loss = -(targets * log_probs).sum(dim=1)  # [batch]
        
        # Average over batch
        loss = loss.mean()
        
        return loss
    
    def get_state(self) -> dict[str, any]:
        """Serialize trainer state for checkpointing."""
        return {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }
    
    def load_state(self, state: dict[str, any]) -> None:
        """Restore trainer state from checkpoint."""
        self.network.load_state_dict(state["network_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.total_updates = state.get("total_updates", 0)
        logger.info(f"StrategyNetworkTrainer restored: {self.total_updates} updates")
