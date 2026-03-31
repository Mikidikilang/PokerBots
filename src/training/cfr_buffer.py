"""
Regret Buffer & Value Network (cfr_buffer.py).

[PHASE 2] Regret storage and network training.

REGRET BUFFER
--------------

In traditional CFR, we store exact cumulative regrets for every infoset:
    regrets[infoset_id] = {action_idx: cumulative_regret}

In Deep CFR, we use a neural network to implicitly represent regrets:
    network: observation → action_regrets (via backprop)

The Regret Buffer bridges these approaches:
    1. **Reservoir Sampling:** Keep a fixed-size sample of (infoset, action, regret) tuples
    2. **Uniform Sampling:** All stored samples have equal probability
    3. **Bounded Memory:** O(buffer_size) space regardless of # infosets visited

VALUE NETWORK TRAINING
-----------------------

The value network is repurposed from the actor-critic architecture:
    input: observation (flattened state representation)
    output: action_regrets[num_actions] (one prediction per action)

Training:
    1. Collect trajectories via MCCFR traversal
    2. Compute true counterfactual regrets
    3. Store (obs, action_regrets) in buffer
    4. Sample mini-batches from buffer
    5. Train network: L = ||network_output - true_regrets||²

Convergence:
    - Function approximation error decreases over time
    - Network generalizes regrets across similar states (generalization)
    - Regret matching converges to Nash equilibrium

---

References
    - Brownlee et al. (2017): "Hybrid Computing using a Neural Network with Dynamic External Memory"
    - Lanctot et al. (2009): "An Introduction to CFR"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RegretSample:
    """Single sample in the regret experience buffer."""
    
    infoset_id: str
    observation: torch.Tensor          # Flattened observation [obs_dim]
    legal_actions: list[int]           # Valid actions at this infoset
    counterfactual_regrets: dict[int, float]  # {action_idx: regret_value}
    
    def to_network_targets(self, num_actions: int = 12) -> torch.Tensor:
        """
        Convert regrets dict to tensor for network training.
        
        Output: tensor[num_actions] where:
            - illegal actions: -∞ (masked out during training)
            - legal actions: actual counterfactual regret values
        
        Args:
            num_actions: Total number of possible actions (usually 12 in poker)
        
        Returns:
            torch.Tensor of shape [num_actions] with regret targets
        """
        targets = torch.full((num_actions,), float('-inf'), dtype=torch.float32)
        
        for action_idx, regret in self.counterfactual_regrets.items():
            if action_idx in self.legal_actions:
                targets[action_idx] = float(regret)
        
        return targets


class RegretBuffer:
    """
    Reservoir sampling buffer for counterfactual regrets.
    
    Stores (observation, action_regrets) pairs via uniform reservoir sampling.
    Used to train the value network (regret predictor).
    
    Algorithm (Vitter 1985):
        for each new sample i:
            j = random(0, i)
            if j < buffer_size:
                buffer[j] = sample_i
    
    Properties:
        - Each buffer[i] has equal probability of being sampled
        - New samples replace old ones uniformly
        - Memory is O(buffer_size) regardless of stream length
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        num_actions: int = 12,
    ):
        """
        Args:
            buffer_size: Maximum number of regret samples to store
            num_actions: Number of discrete actions (for tensor shaping)
        """
        self.buffer_size = buffer_size
        self.num_actions = num_actions
        
        self.samples: list[RegretSample] = []
        self.reservoir_count = 0  # Total samples seen (for uniform probability)
        
        logger.info(f"RegretBuffer initialized: size={buffer_size}, actions={num_actions}")
    
    def add_sample(
        self,
        infoset_id: str,
        observation: torch.Tensor,
        legal_actions: list[int],
        counterfactual_regrets: dict[int, float],
    ) -> None:
        """
        Add a new regret sample via reservoir sampling.
        
        Args:
            infoset_id: Information set identifier
            observation: Observation tensor [obs_dim]
            legal_actions: List of legal action indices
            counterfactual_regrets: {action: regret_value}
        """
        sample = RegretSample(
            infoset_id=infoset_id,
            observation=observation.clone().detach(),
            legal_actions=legal_actions,
            counterfactual_regrets=counterfactual_regrets,
        )
        
        # Reservoir sampling algorithm
        j = np.random.randint(0, self.reservoir_count + 1)
        
        if j < self.buffer_size:
            if len(self.samples) < self.buffer_size:
                self.samples.append(sample)
            else:
                self.samples[j] = sample
        
        self.reservoir_count += 1
        
        if self.reservoir_count % 1000 == 0:
            logger.debug(
                f"RegretBuffer: {self.reservoir_count} samples seen, "
                f"{len(self.samples)} in buffer"
            )
    
    def sample_batch(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> dict[str, torch.Tensor] | None:
        """
        Sample a mini-batch for network training.
        
        Args:
            batch_size: Number of samples per batch
            device: PyTorch device for tensors
        
        Returns:
            {
                "observations": tensor[batch_size, obs_dim],
                "targets": tensor[batch_size, num_actions],
                "legal_action_masks": tensor[batch_size, num_actions],
            }
            or None if buffer is too small
        """
        if len(self.samples) < batch_size:
            logger.warning(
                f"Buffer too small: {len(self.samples)} < {batch_size}"
            )
            return None
        
        # Sample indices uniformly
        indices = np.random.choice(len(self.samples), size=batch_size, replace=False)
        
        # Extract observations and targets
        obs_list = []
        target_list = []
        mask_list = []
        
        for idx in indices:
            sample = self.samples[idx]
            
            # Observation
            obs_list.append(sample.observation)
            
            # Target regrets
            targets = sample.to_network_targets(self.num_actions)
            target_list.append(targets)
            
            # Legal action mask (for loss computation)
            mask = torch.zeros(self.num_actions, dtype=torch.bool)
            for action in sample.legal_actions:
                mask[action] = True
            mask_list.append(mask)
        
        # Stack into batch tensors
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


class RegretValueNetwork(nn.Module):
    """
    [NETWORK ARCHITECTURE]
    
    Predicts counterfactual regrets for each action given an observation.
    
    Repurposes the actor-critic network's value head (which was predicting
    cumulative returns in PPO). Now it predicts action regrets instead.
    
    Input:  observation (flattened, ~350 dims for heads-up)
    Output: action_regret_predictions (12 dims, one per action)
    
    Training: MSE loss against true counterfactual regrets from MCCFR traversal
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 12,
        hidden_dim: int = 512,
    ):
        """
        Args:
            obs_dim: Observation dimension (from features.py)
            num_actions: Number of discrete actions (12 in poker)
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        
        # MLP for regret prediction
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        logger.info(
            f"RegretValueNetwork: obs_dim={obs_dim}, "
            f"actions={num_actions}, hidden={hidden_dim}"
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: tensor[batch_size, obs_dim]
        
        Returns:
            regret_predictions: tensor[batch_size, num_actions]
        """
        return self.mlp(observations)


class RegretNetworkTrainer:
    """
    Trains the value network to predict counterfactual regrets.
    
    Usage:
        >>> trainer = RegretNetworkTrainer(network, buffer, device)
        >>> for epoch in range(epochs):
        ...     stats = trainer.train_epoch()
        ...     print(f"Loss: {stats['loss']:.4f}")
    """
    
    def __init__(
        self,
        network: RegretValueNetwork,
        regret_buffer: RegretBuffer,
        learning_rate: float = 1e-4,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            network: RegretValueNetwork to train
            regret_buffer: RegretBuffer with samples
            learning_rate: Adam learning rate
            device: PyTorch device
        """
        self.network = network.to(device)
        self.regret_buffer = regret_buffer
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
            batch = self.regret_buffer.sample_batch(batch_size, self.device)
            
            if batch is None:
                logger.warning("Buffer empty, skipping epoch")
                return {"loss": float('nan'), "updates": 0}
            
            # Forward pass
            obs = batch["observations"]
            targets = batch["targets"]
            masks = batch["legal_action_masks"]
            
            predictions = self.network(obs)
            
            # MSE loss on legal actions only
            # Illegal actions (targets = -∞) are masked out
            loss = self._masked_mse_loss(predictions, targets, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.total_updates += 1
        
        avg_loss = total_loss / num_batches
        
        logger.info(
            f"RegretNetworkTrainer Epoch: "
            f"loss={avg_loss:.6f}, total_updates={self.total_updates}"
        )
        
        return {
            "loss": avg_loss,
            "updates": num_batches,
        }
    
    def _masked_mse_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss computed only on legal actions.
        
        Args:
            predictions: [batch, num_actions]
            targets: [batch, num_actions] (with -∞ for illegal actions)
            legal_masks: [batch, num_actions] (True for legal actions)
        
        Returns:
            Scalar loss tensor
        """
        # Mask out illegal actions
        masked_predictions = predictions * legal_masks.float()
        masked_targets = targets.clone()
        masked_targets[~legal_masks] = 0  # Zero out illegal targets
        
        # MSE on legal actions
        loss = torch.mean((masked_predictions - masked_targets) ** 2 * legal_masks.float())
        
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
        logger.info(f"RegretNetworkTrainer restored: {self.total_updates} updates")
