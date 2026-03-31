"""Phase 3: Neural Networks for VR-DeepPDCFR+

================================================================================
ARCHITECTURAL OVERVIEW
================================================================================

VR-DeepPDCFR+ requires 4 distinct neural networks per player, each trained on
different data with different objectives:

1. θ (Cumulative Advantage Network):
   - Learns cumulative advantage estimates across iterations
   - Bootstraps from frozen θ_{t-1} to compute targets
   - Input: state features (feature_dim,)
   - Output: action advantages (num_actions,)
   - Loss: MSE with bootstrapped targets

2. φ (Instantaneous Advantage Network):
   - Learns immediate advantages from ephemeral buffer (single iteration)
   - Fresh start each iteration (not bootstrapped)
   - Input: state features (feature_dim,)
   - Output: action advantages (num_actions,)
   - Loss: MSE with observed advantages

3. Q (Value Baseline Network):
   - Expected SARSA-style critic for variance reduction
   - Predicts state value (single scalar output)
   - Input: state features (feature_dim,)
   - Output: scalar baseline value (1,)
   - Loss: MSE with bootstrapped targets

4. Π (Average Strategy Network):
   - Nash-converging average strategy via time-weighted behavioral cloning
   - Trained on persistent strategy buffer with decay-weighted targets
   - Input: state features (feature_dim,)
   - Output: action probabilities (through softmax)
   - Loss: Cross-entropy with time-decay weighted targets

================================================================================
DESIGN PRINCIPLES
================================================================================

MLPBase:
  - Configurable architecture (hidden layers, sizes, activation)
  - No output layer (subclasses add specialized output heads)
  - Proper weight initialization using orthogonal or Xavier

AdvantageNetwork, ValueNetwork, StrategyNetwork:
  - All inherit from MLPBase
  - Add task-specific output layers
  - Mathematically pure: tensors in, tensors out
  - No mixing of game logic into network code

Freezing & Copying:
  - freeze_network(net) -> creates a frozen copy (for bootstrapping)
  - Used for θ_frozen = freeze_network(θ) at iteration transitions
  - Critical: Frozen network prevents circular gradient dependencies

Type System:
  - Full type hints for all methods and attributes
  - Enables better IDE support and easier refactoring

================================================================================
REFERENCES
================================================================================

- Koulis, Schvartzman et al. (2022): "VR-DeepPDCFR+" Section 3
- He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level Performance"
  (for Kaiming/He initialization)
- PyTorch Documentation: nn.Module, nn.Linear, nn.ReLU, etc.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, orthogonal_

logger = logging.getLogger(__name__)


# ============================================================================
# BASE MLP MODULE
# ============================================================================

class MLPBase(nn.Module):
    """Configurable multi-layer perceptron backbone.
    
    Provides a shared base for all network variants. Subclasses add
    specialized output heads for their task (advantages, value, strategy).
    
    Architecture:
        input_dim -> hidden[0] -> hidden[1] -> ... -> hidden[n-1] -> (subclass output)
    
    Attributes:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function class (nn.ReLU, nn.Tanh, etc.)
        use_layer_norm: Whether to apply LayerNorm after each layer
        dropout_p: Dropout probability after each hidden layer (0.0 = no dropout)
        hidden_layers: nn.Sequential containing the hidden layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: type = nn.ReLU,
        use_layer_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        """Initialize the MLP backbone.
        
        Args:
            input_dim: Size of input feature vector
            hidden_dims: List of hidden layer widths, e.g., [512, 256, 256]
            activation: Activation function class (not instance!)
            use_layer_norm: Apply LayerNorm after each layer (with dropout, before activation)
            dropout_p: Dropout probability (0.0 = disabled)
            
        Example:
            mlp = MLPBase(input_dim=100, hidden_dims=[256, 128, 64])
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.dropout_p = dropout_p
        
        # Build hidden layers
        layers: List[nn.Module] = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            linear = nn.Linear(current_dim, hidden_dim)
            # Initialize with Kaiming (He) initialization
            kaiming_uniform_(linear.weight, nonlinearity='relu' if activation == nn.ReLU else 'linear')
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            
            # Optional layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Optional dropout
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            
            # Activation function
            layers.append(activation())
            
            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hidden layers.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Tensor of shape (batch_size, hidden_dims[-1]) or (hidden_dims[-1],)
        """
        return self.hidden_layers(x)


# ============================================================================
# ADVANTAGE NETWORK (for θ and φ)
# ============================================================================

class AdvantageNetwork(MLPBase):
    """Neural network for learning action advantages.
    
    Used for both cumulative (θ) and instantaneous (φ) advantage networks.
    
    Architecture:
        features -> MLP -> output_dim (no activation, outputs raw real numbers)
    
    The output represents Q-values or advantages, which are inherently
    unbounded and should not be passed through softmax or other constraints.
    
    Attributes:
        output_dim: Number of actions (output dimension)
        output_layer: Final linear layer (no activation)
        
    Methods:
        forward: Computes advantage estimates for all actions
        
    Example:
        network = AdvantageNetwork(input_dim=64, output_dim=4, hidden_dims=[256, 128])
        state_features = torch.randn(batch_size, 64)
        advantages = network(state_features)  # Shape: (batch_size, 4)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128],
        **kwargs,
    ) -> None:
        """Initialize advantage network.
        
        Args:
            input_dim: State feature dimension
            output_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            **kwargs: Additional arguments passed to MLPBase
                     (activation, use_layer_norm, dropout_p)
        """
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, **kwargs)
        
        self.output_dim = output_dim
        
        # Output layer: no activation (unbounded real-valued advantages)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        kaiming_uniform_(self.output_layer.weight, nonlinearity='linear')
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute advantage estimates.
        
        Args:
            x: State features, shape (batch_size, input_dim)
            
        Returns:
            Action advantages, shape (batch_size, output_dim)
            Values are unbounded real numbers (not probabilities).
        """
        hidden = self.hidden_layers(x)
        advantages = self.output_layer(hidden)
        return advantages


# ============================================================================
# VALUE NETWORK (for Q)
# ============================================================================

class ValueNetwork(MLPBase):
    """Neural network for learning state value estimates (baseline).
    
    Implements Expected SARSA-style critic for variance reduction.
    Outputs a single scalar value estimate per state.
    
    Architecture:
        features -> MLP -> 1 (scalar value, no activation)
    
    The output is a state value estimate and should not be constrained
    to any particular range (can be negative, depends on reward scale).
    
    Attributes:
        output_layer: Final linear layer outputting single value
        
    Methods:
        forward: Computes value estimate for input states
        
    Example:
        network = ValueNetwork(input_dim=64, hidden_dims=[256, 128])
        state_features = torch.randn(batch_size, 64)
        values = network(state_features)  # Shape: (batch_size, 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        **kwargs,
    ) -> None:
        """Initialize value network.
        
        Args:
            input_dim: State feature dimension
            hidden_dims: Hidden layer dimensions
            **kwargs: Additional arguments passed to MLPBase
                     (activation, use_layer_norm, dropout_p)
        """
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, **kwargs)
        
        # Output layer: single scalar value (no activation)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        kaiming_uniform_(self.output_layer.weight, nonlinearity='linear')
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value estimate.
        
        Args:
            x: State features, shape (batch_size, input_dim)
            
        Returns:
            State values, shape (batch_size, 1)
            Single scalar value per state (unbounded real number).
        """
        hidden = self.hidden_layers(x)
        values = self.output_layer(hidden)
        return values


# ============================================================================
# STRATEGY NETWORK (for Π)
# ============================================================================

class StrategyNetwork(MLPBase):
    """Neural network for learning Nash-equilibrium average strategy.
    
    Outputs raw logits (no activation). The CFR Engine (Phase 4) handles:
      1. Dynamic legal action masking: logits[~legal_mask] = -inf
      2. Softmax application with masked logits
      3. Cross-entropy loss with time-decay weights
    
    CRITICAL: StrategyNetwork must NOT apply Softmax/LogSoftmax internally.
    In imperfect-information games like poker, legal actions change dynamically.
    If Softmax is applied over ALL actions, then masking breaks the probability
    distribution (will no longer sum to 1.0).
    
    Architecture:
        features -> MLP -> output_dim (raw logits, no activation)
    
    Attributes:
        output_dim: Number of actions
        output_layer: Final linear layer (no activation)
        
    Methods:
        forward: Computes raw action logits
        
    Example:
        network = StrategyNetwork(input_dim=64, output_dim=4, hidden_dims=[256, 128])
        state_features = torch.randn(batch_size, 64)
        logits = network(state_features)  # Shape: (batch_size, 4), unbounded
        # CFR Engine applies masking and softmax based on legal actions
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128],
        **kwargs,
    ) -> None:
        """Initialize strategy network.
        
        Args:
            input_dim: State feature dimension
            output_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            **kwargs: Additional arguments passed to MLPBase
                     (activation, use_layer_norm, dropout_p)
        """
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, **kwargs)
        
        self.output_dim = output_dim
        
        # Output layer: raw logits (no activation, no softmax)
        # The CFR Engine will apply masking and softmax based on legal actions
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        kaiming_uniform_(self.output_layer.weight, nonlinearity='linear')
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw action logits.
        
        Returns unbounded logits. The CFR Engine applies:
          1. Legal action masking: logits[~legal_mask] = -inf
          2. Softmax to convert to probabilities
          3. Cross-entropy loss with time-decay weights
        
        Args:
            x: State features, shape (batch_size, input_dim)
            
        Returns:
            Raw logits, shape (batch_size, output_dim)
            Unbounded real numbers (NOT normalized).
        """
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        return logits


# ============================================================================
# NETWORK UTILITIES
# ============================================================================

def freeze_network(network: nn.Module) -> None:
    """Permanently freeze a network for inference-only use.
    
    Sets requires_grad=False for all parameters and eval mode.
    This is more efficient than deepcopy for persistent frozen networks
    like cumulative_advantage_frozen.
    
    Args:
        network: The network to freeze in-place
        
    Example:
        network_frozen = AdvantageNetwork(input_dim=64, output_dim=4)
        freeze_network(network_frozen)  # Now frozen for target network use
        
        # Later, update weights via state_dict:
        network_frozen.load_state_dict(network.state_dict())
    """
    # Disable gradient computation
    for param in network.parameters():
        param.requires_grad = False
    
    # Set to evaluation mode
    network.eval()
    
    logger.debug(f"Froze {network.__class__.__name__}")


# ============================================================================
# NETWORK BUNDLE (Convenience wrapper)
# ============================================================================

class VRDeepPDCFRNetworks:
    """Container for all 4 networks of VR-DeepPDCFR+ per player.
    
    Manages the complete set of networks with convenience methods for
    freezing, unfreezing, and parameter synchronization.
    
    Attributes:
        cumulative_advantage: θ (cumulative advantage network)
        cumulative_advantage_frozen: Frozen copy of θ for bootstrapping
        instantaneous_advantage: φ (instantaneous advantage network)
        value: Q (state value baseline)
        strategy: Π (average strategy)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: type = nn.ReLU,
        use_layer_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        """Initialize all networks for one player.
        
        Args:
            input_dim: State feature dimension
            output_dim: Number of actions
            hidden_dims: Hidden layer dimensions for all networks
            activation: Activation function class
            use_layer_norm: Whether to use layer normalization
            dropout_p: Dropout probability
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize trainable networks
        self.cumulative_advantage = AdvantageNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        )
        
        # Create permanent frozen copy for cumulative advantage bootstrapping
        # CRITICAL: Use persistent instance + state_dict synchronization,
        # NOT deepcopy, for GPU efficiency and device placement safety.
        self.cumulative_advantage_frozen = AdvantageNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        )
        freeze_network(self.cumulative_advantage_frozen)
        # Initialize frozen weights to match trainable network
        self.cumulative_advantage_frozen.load_state_dict(
            self.cumulative_advantage.state_dict()
        )
        
        self.instantaneous_advantage = AdvantageNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        )
        
        self.value = ValueNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        )
        
        self.strategy = StrategyNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        )
    
    def update_cumulative_frozen(self) -> None:
        """Synchronize frozen cumulative network with current network.
        
        Called at iteration boundaries to update bootstrap targets.
        Uses state_dict (not deepcopy) for GPU efficiency and device safety.
        """
        self.cumulative_advantage_frozen.load_state_dict(
            self.cumulative_advantage.state_dict()
        )
    
    def to_device(self, device: torch.device) -> None:
        """Move all networks to specified device (CPU or GPU).
        
        Args:
            device: torch.device instance
        """
        self.cumulative_advantage = self.cumulative_advantage.to(device)
        self.cumulative_advantage_frozen = self.cumulative_advantage_frozen.to(device)
        self.instantaneous_advantage = self.instantaneous_advantage.to(device)
        self.value = self.value.to(device)
        self.strategy = self.strategy.to(device)
    
    def train_mode(self) -> None:
        """Set all networks to training mode."""
        self.cumulative_advantage.train()
        self.instantaneous_advantage.train()
        self.value.train()
        self.strategy.train()
        # Note: cumulative_advantage_frozen stays in eval mode
    
    def eval_mode(self) -> None:
        """Set all trainable networks to evaluation mode."""
        self.cumulative_advantage.eval()
        self.instantaneous_advantage.eval()
        self.value.eval()
        self.strategy.eval()
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all parameters that require gradients.
        
        Returns:
            List of parameters from trainable networks
        """
        params = []
        params.extend(self.cumulative_advantage.parameters())
        params.extend(self.instantaneous_advantage.parameters())
        params.extend(self.value.parameters())
        params.extend(self.strategy.parameters())
        return params
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"VRDeepPDCFRNetworks(\n"
            f"  cumulative_advantage={self.cumulative_advantage.hidden_dims},\n"
            f"  instantaneous_advantage={self.instantaneous_advantage.hidden_dims},\n"
            f"  value={self.value.hidden_dims},\n"
            f"  strategy={self.strategy.hidden_dims}\n"
            f")"
        )
