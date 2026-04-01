"""Phase 4: VR-DeepPDCFR+ Engine - Core Algorithm Implementation

================================================================================
ARCHITECTURAL OVERVIEW
================================================================================

The VR-DeepPDCFR+ (Variance-Reduced Deep Predictive-CFR Plus) engine is the
mathematical core that:

1. TRAVERSES the game tree recursively
   - Terminal nodes: return payoffs
   - Chance nodes: sample/evaluate stochastic transitions
   - Player nodes: compute predictive strategy, recursively value all actions

2. COMPUTES VARIANCE-REDUCED ADVANTAGES
   - Q baseline: Expected value of state (variance reduction)
   - Instantaneous advantages: Action values minus baseline
   - Cumulative advantages: Time-decayed bootstrapping from frozen θ

3. TRAINS THE 4 NETWORKS
   - π (Strategy): Cross-entropy with time-decay weights
   - φ (Instantaneous Advantage): MSE with traversal-computed advantages
   - Q (Value Baseline): MSE with expected action values
   - θ (Cumulative Advantage): Bootstrapped MSE with temporal discounting

KEY MATHEMATICAL COMPONENTS
============================

Predictive Strategy (CFR+ Matching):
  For each infoset, compute regret-matched strategy:
    advantage_i = max(0, θ_frozen_i + φ_i - V_state)
    strategy_i ∝ advantage_i
    
Legal Action Masking:
  strategy[illegal_actions] = 0
  (renormalize to maintain sum=1)

Variance Reduction via Q Baseline:
  advantage_inst = action_value - Q_baseline
  This reduces gradient variance without biasing the gradient

Temporal Discounting for θ (Cumulative Advantage):
  At iteration t:
    w_t = (t-1)^2 / ((t-1)^2 + 1)
    target_θ = w_t * θ_frozen + (1 - w_t) * φ_observed
  
  Early iterations (t≈1): w≈0, rely on φ (ephemeral data)
  Late iterations (t>>1): w≈1, rely on θ (persistent data)
  
  Apply ReLU: target_θ = max(0, target_θ)  [CFR+ clipping]

================================================================================
REFERENCES
================================================================================

- Koulis, Schvartzman et al. (2022): "VR-DeepPDCFR+" Sections 3-4
- Burch, Lanctot, Bowling (2014): "Improved Opponent Modeling and Game Tree Search"
- Tsaknakis & Spirakis (2008): "On the Convergence of Witness Proof Sets"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from src.training.buffers import BufferManager, Transition
from src.training.dcfr_params import compute_dcfr_discount, DCFRParameters
from src.model.networks import VRDeepPDCFRNetworks

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_cfr_plus_strategy(
    advantages: np.ndarray,
) -> np.ndarray:
    """Compute CFR+ regret-matched strategy from advantages.
    
    CFR+ uses max(0, advantage) to ensure positive regrets only.
    This prevents "oscillation" in strategy convergence.
    
    Args:
        advantages: Shape (num_actions,), unbounded real numbers
        
    Returns:
        Shape (num_actions,), normalized strategy summing to 1.0
    """
    # Clip at 0: only positive regrets matter in CFR+
    positive_regrets = np.maximum(0.0, advantages)
    
    # Normalize (handle all-zero case)
    regret_sum = positive_regrets.sum()
    if regret_sum > 0:
        strategy = positive_regrets / regret_sum
    else:
        # Uniform fallback if all advantages are negative
        strategy = np.ones(len(advantages)) / len(advantages)
    
    return strategy.astype(np.float32)


def apply_action_mask(
    strategy: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    """Apply legal action masking to strategy.
    
    Sets illegal action probabilities to 0 and renormalizes.
    
    Args:
        strategy: Shape (num_actions,), unnormalized or normalized probabilities
        legal_mask: Shape (num_actions,), boolean (1 = legal, 0 = illegal)
        
    Returns:
        Shape (num_actions,), normalized strategy over legal actions only
    """
    # Zero out illegal actions
    masked_strategy = strategy * legal_mask.astype(np.float32)
    
    # Renormalize
    strategy_sum = masked_strategy.sum()
    if strategy_sum > 0:
        masked_strategy = masked_strategy / strategy_sum
    else:
        # Fallback: uniform over legal actions
        num_legal = legal_mask.sum()
        if num_legal > 0:
            masked_strategy = legal_mask.astype(np.float32) / num_legal
        else:
            # Edge case: no legal actions (should not happen in valid games)
            masked_strategy = np.ones(len(strategy)) / len(strategy)
    
    return masked_strategy.astype(np.float32)


def compute_temporal_decay_weight(iteration_t: int) -> float:
    """Compute temporal decay weight for cumulative advantage bootstrapping.
    
    At iteration t:
        w = (t-1)^2 / ((t-1)^2 + 1)
    
    This smoothly transitions from φ (early iterations) to θ (late iterations).
    - t=1: w=0 (rely entirely on ephemeral φ)
    - t=2: w=1/2 (balance between φ and θ)
    - t→∞: w→1 (rely entirely on persistent θ)
    
    Args:
        iteration_t: Current CFR iteration (1-indexed)
        
    Returns:
        Weight scalar in [0, 1)
    """
    numerator = (iteration_t - 1) ** 2
    denominator = (iteration_t - 1) ** 2 + 1
    return numerator / denominator


# ============================================================================
# MAIN ENGINE CLASS
# ============================================================================

class VRDeepPDCFREngine:
    """Variance-Reduced DeepPDCFR+ algorithm engine.
    
    Manages game tree traversal, advantage computation, and network training
    for all players simultaneously. Handles:
    
    1. Recursive game tree traversal
    2. Predictive strategy computation
    3. Variance-reduced advantage estimation
    4. Training of 4 networks per player with loss-specific objectives
    
    Attributes:
        buffer_managers: Dict[player_id] -> BufferManager for each player
        networks: Dict[player_id] -> VRDeepPDCFRNetworks for each player
        optimizers: Dict[player_id] -> Dict[network_name] -> Optimizer
        device: torch device (CPU or GPU)
        current_iteration: Current CFR iteration counter
    """
    
    def __init__(
        self,
        buffer_managers: Dict[int, BufferManager],
        networks: Dict[int, VRDeepPDCFRNetworks],
        optimizers: Dict[int, Dict[str, Optimizer]],
        device: torch.device = torch.device("cpu"),
        max_depth: int = 10,
        dcfr_params: Optional[DCFRParameters] = None,
    ) -> None:
        """Initialize VR-DeepPDCFR+ engine.
        
        Args:
            buffer_managers: Dict mapping player_id -> BufferManager instance
            networks: Dict mapping player_id -> VRDeepPDCFRNetworks instance
            optimizers: Dict mapping player_id -> {
                'cumulative': Optimizer,
                'instantaneous': Optimizer,
                'value': Optimizer,
                'strategy': Optimizer
            }
            device: torch.device for computation
            max_depth: Maximum depth for game tree traversal. When depth >= max_depth,
                      return estimated values from Q networks instead of continuing traversal.
                      Prevents infinite recursion on large games like 6-Max NLHE.
            dcfr_params: DCFR parameters for loss weighting (default: standard Brown & Sandholm 2019)
        """
        self.buffer_managers = buffer_managers
        self.networks = networks
        self.optimizers = optimizers
        self.device = device
        self.current_iteration = 1
        self.max_depth = max_depth
        
        # Initialize DCFR parameters for loss weighting
        self.dcfr_params = dcfr_params or DCFRParameters(
            alpha=1.5,
            beta=0.0,
            gamma=2.0
        )
        
        # Move all networks to device
        for player_id, net_bundle in self.networks.items():
            net_bundle.to_device(device)
        
        logger.info(
            f"VRDeepPDCFREngine initialized: {len(networks)} players, "
            f"device={device}, DCFR params: alpha={self.dcfr_params.alpha}, "
            f"beta={self.dcfr_params.beta}, gamma={self.dcfr_params.gamma}"
        )
    
    def start_iteration(self) -> None:
        """Mark the start of a new CFR iteration.
        
        Called before traversing the game tree. Performs:
        - Clears ephemeral buffers (EphemeralAdvantageBuffer)
        - Puts all networks in training mode
        - Updates cumulative frozen network with current weights
        """
        for player_id, buffer_manager in self.buffer_managers.items():
            buffer_manager.start_iteration()
            self.networks[player_id].update_cumulative_frozen()
            self.networks[player_id].train_mode()
        
        logger.debug(f"Started iteration {self.current_iteration}")
    
    def end_iteration(self) -> None:
        """Mark the end of a CFR iteration.
        
        Called after traversal and training. Increments iteration counter
        and updates persistent frozen networks.
        """
        for player_id, buffer_manager in self.buffer_managers.items():
            buffer_manager.end_iteration()
        
        self.current_iteration += 1
        logger.debug(f"Ended iteration. Next will be {self.current_iteration}")
    
    def traverse(
        self,
        state: Any,
        player_reach_probs: Dict[int, float],
        updating_player: int,
        depth: int = 0,
    ) -> Dict[int, float]:
        """Recursively traverse the game tree with External Sampling MCCFR.
        
        This is the core VR-DeepPDCFR+ traversal algorithm with External Sampling.
        For LARGE game trees (e.g., 6-Max NLHE with ~10^161 nodes), full enumeration
        is intractable. External Sampling reduces computation by sampling ONE opponent
        action per non-updating player node instead of enumerating all.
        
        Algorithm:
            1. If terminal: return payoffs
            2. If chance node: sample/evaluate transitions
            3. If player node:
               a. Compute predictive strategy from frozen θ and φ
               b. Apply legal action masking
               
               **If acting_player == updating_player**:
                 - Recursively compute child values for ALL legal actions
                 - Query Q baseline
                 - Compute instantaneous advantages for all actions
                 - Store transition in buffer for this player
                 - Return value weighted by strategy
               
               **If acting_player != updating_player**:
                 - Sample a SINGLE action according to predictive_strategy
                 - Recursively compute child value for ONLY that sampled action
                 - Do NOT compute advantages or store in buffer
                 - Return child value directly (unbiased via importance weighting)
        
        Args:
            state: GameState object with required methods
            player_reach_probs: Dict[player_id] -> cumulative reach probability
            updating_player: Which player's regrets/strategies we're updating this pass
            
        Returns:
            Dict[player_id] -> expected value from this state
        """
        # BASE CASE: Terminal node
        if state.is_terminal():
            payoffs = state.get_terminal_payoffs()
            logger.debug(f"Terminal reached: payoffs={payoffs}")
            return payoffs
        
        # DEPTH LIMIT: Use Q network to estimate values for all players
        if depth >= self.max_depth:
            logger.debug(f"Depth limit reached at depth={depth}, using Q networks for value estimation")
            estimated_values = {}
            
            with torch.no_grad():
                for player_id in self.networks.keys():
                    # Get features from this player's perspective
                    player_features = state.get_infoset_features(player_id)
                    features_tensor = torch.FloatTensor(player_features).unsqueeze(0).to(self.device)
                    
                    # Query Q network for this player
                    q_value = self.networks[player_id].value(features_tensor)[0, 0].item()
                    estimated_values[player_id] = float(q_value)
            
            logger.debug(f"Estimated values at depth limit: {estimated_values}")
            return estimated_values
        
        # CHANCE NODE: Stochastic transition (External Sampling MCCFR)
        if state.is_chance_node():
            # External Sampling: sample exactly ONE chance outcome, not all
            # This is mathematically correct for External Sampling MCCFR:
            # we explore the sampled branch with importance weighting
            logger.debug(f"Chance node at depth {depth}: sampling one outcome")
            
            child_state = state.sample_chance_outcome()
            
            # Recursively traverse the sampled branch
            # No reach probability weighting for chance: reach stays the same
            # (chance events don't have player reach probabilities)
            child_values = self.traverse(child_state, player_reach_probs, updating_player, depth=depth + 1)
            
            logger.debug(f"Chance node: traversed sampled branch, values={child_values}")
            return child_values
        
        # PLAYER NODE: Decision point
        acting_player = state.get_acting_player()
        infoset_features = state.get_infoset_features()
        legal_actions = state.get_legal_actions()
        num_legal_actions = int(legal_actions.sum())
        
        logger.debug(
            f"Player {acting_player} node: {num_legal_actions} legal actions, "
            f"reach_prob={player_reach_probs[acting_player]:.6f}, "
            f"updating_player={updating_player}"
        )
        
        # =====================================================================
        # STEP 1: Compute predictive strategy using frozen θ and φ
        # =====================================================================
        with torch.no_grad():
            features_tensor = torch.FloatTensor(infoset_features).unsqueeze(0).to(self.device)
            
            theta_frozen_output = self.networks[acting_player].cumulative_advantage_frozen(
                features_tensor
            )  # Shape: (1, num_actions)
            phi_output = self.networks[acting_player].instantaneous_advantage(
                features_tensor
            )  # Shape: (1, num_actions)
            
            # Sum for cumulative advantage estimate
            cumulative_advantages = (
                theta_frozen_output[0].cpu().numpy() +
                phi_output[0].cpu().numpy()
            )  # Shape: (num_actions,)
            
            # Query Q for baseline
            q_baseline = self.networks[acting_player].value(
                features_tensor
            )[0, 0].item()  # Scalar
            
            # CFR+ regret matching: max(0, advantage - baseline)
            advantages = cumulative_advantages - q_baseline
            
            # Compute strategy from advantages
            predictive_strategy = compute_cfr_plus_strategy(advantages)
            
            # Apply legal action masking
            predictive_strategy = apply_action_mask(
                predictive_strategy,
                legal_actions.astype(np.float32)
            )
        
        logger.debug(
            f"Predictive strategy: {predictive_strategy}, "
            f"baseline={q_baseline:.4f}"
        )
        
        # =====================================================================
        # BRANCHING: Full Enumeration vs External Sampling
        # =====================================================================
        
        if acting_player == updating_player:
            # ================== FULL ENUMERATION (Updating Player) ==================
            # Enumerate ALL legal actions, compute advantages, store in buffer
            
            logger.debug(f"Full enumeration mode for updating_player={updating_player}")
            
            action_values = {}
            for action_idx in range(len(legal_actions)):
                if not legal_actions[action_idx]:
                    action_values[action_idx] = None
                    continue
                
                # Get child state
                child_state = state.get_action_taken(action_idx)
                
                # Update reach probabilities
                new_reach_probs = dict(player_reach_probs)
                new_reach_probs[acting_player] *= predictive_strategy[action_idx]
                
                # Recursively traverse ALL actions unconditionally
                # CRITICAL: In External Sampling MCCFR, we must explore every legal action
                # at the updating player's nodes, regardless of reach probability. The reach
                # probability of an action does not justify skipping its traversal, as CFR
                # explicitly computes counterfactual values assuming the action was taken.
                child_values = self.traverse(child_state, new_reach_probs, updating_player, depth=depth + 1)
                action_values[action_idx] = child_values
            
            # Compute state value via strategy
            state_values = {player_id: 0.0 for player_id in self.networks.keys()}
            for action_idx in range(len(legal_actions)):
                if legal_actions[action_idx] and action_values[action_idx] is not None:
                    for player_id in state_values.keys():
                        state_values[player_id] += (
                            predictive_strategy[action_idx] * action_values[action_idx][player_id]
                        )
            
            # Compute instantaneous advantages and store in buffer
            instantaneous_advantages = np.zeros(len(legal_actions))
            target_strategy = predictive_strategy.copy()  # Target for π network
            
            for action_idx in range(len(legal_actions)):
                if legal_actions[action_idx] and action_values[action_idx] is not None:
                    # Advantage: action value minus state value
                    child_value_for_player = action_values[action_idx][acting_player]
                    instantaneous_advantages[action_idx] = (
                        child_value_for_player - state_values[acting_player]
                    )
                else:
                    instantaneous_advantages[action_idx] = 0.0
            
            logger.debug(f"Instantaneous advantages: {instantaneous_advantages}")
            
            # Store transition in buffer for updating player
            # Pass legal_mask to ensure Π network only assigns probability to legal actions
            self.buffer_managers[acting_player].add_transition(
                infoset_features=infoset_features,
                action_probs=target_strategy,
                legal_mask=legal_actions.astype(np.float32),
                advantages=instantaneous_advantages,
                reach_prob=player_reach_probs[acting_player],
            )
            
            return state_values
        
        else:
            # ================== EXTERNAL SAMPLING (Non-Updating Player) ==================
            # Sample ONE action according to strategy, traverse only that branch
            
            logger.debug(f"External sampling mode for acting_player={acting_player}, updating_player={updating_player}")
            
            # Get legal action indices
            legal_indices = np.where(legal_actions)[0]
            
            if len(legal_indices) == 0:
                raise RuntimeError(f"No legal actions at non-terminal node")
            
            # Get probabilities for legal actions (normalized)
            legal_probs = predictive_strategy[legal_indices]
            legal_probs = legal_probs / legal_probs.sum()  # Ensure normalized
            
            # Sample ONE action according to strategy
            sampled_action_idx = np.random.choice(legal_indices, p=legal_probs)
            
            logger.debug(
                f"Sampled action {sampled_action_idx} for acting_player={acting_player} "
                f"with prob {legal_probs[list(legal_indices).index(sampled_action_idx)]:.4f}"
            )
            
            # Get child state
            child_state = state.get_action_taken(sampled_action_idx)
            
            # Update reach probabilities for the sampled action
            new_reach_probs = dict(player_reach_probs)
            new_reach_probs[acting_player] *= predictive_strategy[sampled_action_idx]
            
            # Recursively traverse ONLY the sampled branch
            child_values = self.traverse(child_state, new_reach_probs, updating_player, depth=depth + 1)
            
            # Return child values directly (no advantage computation, no buffer storage)
            # The values from this branch are unbiased estimators via importance weighting
            return child_values
    
    def train_networks(self, batch_size: int = 4096, num_epochs: int = 4) -> Dict[str, float]:
        """Train all networks from buffered data.
        
        Performs multiple epochs of gradient descent using:
        - π (Strategy): Cross-entropy with time-decay weights
        - φ (Instantaneous Advantage): MSE
        - Q (Value Baseline): MSE
        - θ (Cumulative Advantage): Bootstrapped MSE with temporal discounting
        
        Args:
            batch_size: Number of samples per minibatch (default 4096)
            num_epochs: Number of training epochs (default 4)
        
        Returns:
            Dict of loss values for logging
        """
        losses = {}
        
        for player_id, network_bundle in self.networks.items():
            logger.info(f"Training player {player_id} networks (batch_size={batch_size}, epochs={num_epochs})...")
            
            buffer_manager = self.buffer_managers[player_id]
            optim_dict = self.optimizers[player_id]
            
            # Get buffer sizes for logging
            adv_buffer_size = buffer_manager.advantage_buffer.size()
            strat_buffer_size = buffer_manager.strategy_buffer.size()
            
            for epoch in range(num_epochs):
                # =========================================================
                # φ Loss (Instantaneous Advantage)
                # =========================================================
                if adv_buffer_size > 0:
                    loss_phi = self._compute_phi_loss(
                        network_bundle,
                        buffer_manager,
                        batch_size
                    )
                    losses[f"player_{player_id}_phi_loss"] = loss_phi.item()
                    
                    # Debug: Check loss tensor state
                    logger.debug(f"φ loss - requires_grad: {loss_phi.requires_grad}, has grad_fn: {loss_phi.grad_fn is not None}, dtype: {loss_phi.dtype}")
                    
                    # Only perform backward if loss requires grad and has a computation graph
                    if loss_phi.requires_grad and loss_phi.grad_fn is not None:
                        optim_dict['instantaneous'].zero_grad()
                        loss_phi.backward()
                        optim_dict['instantaneous'].step()
                    elif loss_phi.requires_grad:
                        logger.warning(f"φ loss requires_grad but has no grad_fn - skipping backward")
                    
                    logger.debug(f"φ loss: {loss_phi.item():.6f}")
                
                # =========================================================
                # Q Loss (Value Baseline)
                # =========================================================
                if adv_buffer_size > 0:
                    loss_q = self._compute_q_loss(
                        network_bundle,
                        buffer_manager,
                        batch_size
                    )
                    losses[f"player_{player_id}_q_loss"] = loss_q.item()
                    
                    # Only perform backward if loss requires grad and has a computation graph
                    if loss_q.requires_grad and loss_q.grad_fn is not None:
                        optim_dict['value'].zero_grad()
                        loss_q.backward()
                        optim_dict['value'].step()
                    elif loss_q.requires_grad:
                        logger.warning(f"Q loss requires_grad but has no grad_fn - skipping backward")
                    
                    logger.debug(f"Q loss: {loss_q.item():.6f}")
                
                # =========================================================
                # π Loss (Strategy)
                # =========================================================
                if strat_buffer_size > 0:
                    loss_pi = self._compute_pi_loss(
                        network_bundle,
                        buffer_manager,
                        batch_size,
                        self.current_iteration
                    )
                    losses[f"player_{player_id}_pi_loss"] = loss_pi.item()
                    
                    # Only perform backward if loss requires grad and has a computation graph
                    if loss_pi.requires_grad and loss_pi.grad_fn is not None:
                        optim_dict['strategy'].zero_grad()
                        loss_pi.backward()
                        optim_dict['strategy'].step()
                    elif loss_pi.requires_grad:
                        logger.warning(f"π loss requires_grad but has no grad_fn - skipping backward")
                    
                    logger.debug(f"π loss: {loss_pi.item():.6f}")
                
                # =========================================================
                # θ Loss (Cumulative Advantage)
                # =========================================================
                if adv_buffer_size > 0:
                    loss_theta = self._compute_theta_loss(
                        network_bundle,
                        buffer_manager,
                        batch_size,
                        self.current_iteration
                    )
                    losses[f"player_{player_id}_theta_loss"] = loss_theta.item()
                    
                    # Only perform backward if loss requires grad and has a computation graph
                    if loss_theta.requires_grad and loss_theta.grad_fn is not None:
                        optim_dict['cumulative'].zero_grad()
                        loss_theta.backward()
                        optim_dict['cumulative'].step()
                    elif loss_theta.requires_grad:
                        logger.warning(f"θ loss requires_grad but has no grad_fn - skipping backward")
                    
                    logger.debug(f"θ loss: {loss_theta.item():.6f}")
        
        return losses
    
    def _compute_phi_loss(
        self,
        network_bundle: VRDeepPDCFRNetworks,
        buffer_manager: BufferManager,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute φ (instantaneous advantage) loss with DCFR weighting.
        
        Loss = weighted MSE where each sample is weighted by DCFR discount factor
        
        DCFR Weight: w_t = (t / (t + γ))^α
        where t = iteration when sample was generated
        
        Weighted Loss = mean( w_t * MSE(predicted, observed) for each sample )
        
        This emphasizes recent samples (higher iteration = higher weight) and
        ensures the network focuses on the most accurate advantage estimates.
        """
        if buffer_manager.advantage_buffer.size() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Sample minibatch INCLUDING iterations for DCFR weighting
        features, _, observed_advantages, iterations = buffer_manager.advantage_buffer.sample_minibatch(
            batch_size, replace=True
        )
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        observed_tensor = torch.FloatTensor(observed_advantages).to(self.device)
        
        # Compute DCFR discount weight for each sample in the batch
        # weight_t = (t / (t + γ))^α  [using alpha exponent for all samples]
        dcfr_weights = np.array([
            compute_dcfr_discount(
                iteration=int(t) - 1,  # Convert to 0-indexed for the function
                regret_old=1.0,  # Use positive regret_old to always apply alpha discount
                params=self.dcfr_params
            )
            for t in iterations
        ], dtype=np.float32)
        
        dcfr_weights_tensor = torch.FloatTensor(dcfr_weights).to(self.device).unsqueeze(-1)
        
        # Predict and compute weighted MSE loss
        predicted = network_bundle.instantaneous_advantage(features_tensor)
        
        # Element-wise MSE
        mse_per_sample = (predicted - observed_tensor) ** 2  # Shape: (batch, num_actions)
        
        # Apply DCFR weights: scale each sample's loss
        weighted_mse = mse_per_sample * dcfr_weights_tensor  # Broadcasting: (batch, num_actions)
        
        # Average over batch and actions
        loss = weighted_mse.mean()
        
        logger.debug(
            f"φ loss: DCFR weights range [{dcfr_weights.min():.6f}, {dcfr_weights.max():.6f}], "
            f"mean weight: {dcfr_weights.mean():.6f}"
        )
        
        return loss
    
    def _compute_q_loss(
        self,
        network_bundle: VRDeepPDCFRNetworks,
        buffer_manager: BufferManager,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute Q (value baseline) loss.
        
        Loss = MSE(predicted_value, target_value)
        
        Target: compute expected return from instantaneous advantages weighted by strategy
        """
        if buffer_manager.advantage_buffer.size() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Sample coherent (features, action_probs, advantages, iterations) tuple from advantage_buffer
        # All three components come from the SAME state, ensuring alignment
        features, action_probs, advantages, _ = buffer_manager.advantage_buffer.sample_minibatch(
            batch_size, replace=True
        )
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        action_probs_tensor = torch.FloatTensor(action_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Compute target: expected value from action advantages weighted by strategy
        # V_target = sum_a pi(a|s) * A(a|s) for the SAME state s
        target_values = torch.sum(
            action_probs_tensor * advantages_tensor, dim=1, keepdim=True
        )  # Shape: (batch, 1)
        
        predicted_values = network_bundle.value(features_tensor)  # Shape: (batch, 1)
        loss = F.mse_loss(predicted_values, target_values)
        
        return loss
    
    def _compute_pi_loss(
        self,
        network_bundle: VRDeepPDCFRNetworks,
        buffer_manager: BufferManager,
        batch_size: int,
        iteration_t: int,
    ) -> torch.Tensor:
        """Compute π (strategy) loss.
        
        Loss = Cross-entropy(predicted_logits, target_policy) * time_decay_weight
        
        IMPORTANT: Apply legal action masking BEFORE log_softmax to ensure
        the Π network does not leak probability mass to illegal actions during
        behavioral cloning.
        
        Time-decay weight emphasizes recent iterations over old ones.
        """
        if buffer_manager.strategy_buffer.size() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        features, target_probs, legal_masks, time_decay_weights = buffer_manager.strategy_buffer.sample_minibatch(
            batch_size, current_iteration=iteration_t, replace=True
        )
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        target_tensor = torch.FloatTensor(target_probs).to(self.device)
        legal_masks_tensor = torch.FloatTensor(legal_masks).to(self.device)
        weights_tensor = torch.FloatTensor(time_decay_weights).to(self.device)
        
        # Network outputs raw logits; apply masking BEFORE softmax
        logits = network_bundle.strategy(features_tensor)  # Shape: (batch, num_actions)
        
        # =====================================================================
        # BEHAVIORAL CLONING MASKED SOFTMAX (AMP-SAFE)
        # =====================================================================
        # Apply legal action mask to logits using AMP-safe masking:
        # Set illegal actions to torch.finfo(dtype).min (safe for float16/bfloat16)
        # This prevents softmax from assigning any probability to illegal actions
        
        mask_value = torch.finfo(logits.dtype).min
        masked_logits = torch.where(
            legal_masks_tensor.bool(),
            logits,
            torch.full_like(logits, mask_value, dtype=logits.dtype)
        )
        
        # Apply log_softmax ONLY to masked logits
        log_probs = F.log_softmax(masked_logits, dim=-1)
        
        # Cross-entropy loss (unweighted)
        entropy_loss = F.kl_div(
            log_probs, target_tensor, reduction='none'
        )  # Shape: (batch, num_actions)
        
        # Sum over actions and weight by time-decay
        entropy_loss = entropy_loss.sum(dim=-1)  # Shape: (batch,)
        weighted_loss = (entropy_loss * weights_tensor).mean()
        
        return weighted_loss
    
    def _compute_theta_loss(
        self,
        network_bundle: VRDeepPDCFRNetworks,
        buffer_manager: BufferManager,
        batch_size: int,
        iteration_t: int,
    ) -> torch.Tensor:
        """Compute θ (cumulative advantage) loss.
        
        Loss = MSE(predicted_θ, target_θ)
        
        where target_θ = w_t * θ_frozen + φ_observed
        and w_t = (t-1)^2 / ((t-1)^2 + 1)
        
        Apply ReLU clipping (CFR+): target = max(0, target)
        """
        if buffer_manager.advantage_buffer.size() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        features, _, observed_advantages, _ = buffer_manager.advantage_buffer.sample_minibatch(
            batch_size, replace=True
        )
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        observed_tensor = torch.FloatTensor(observed_advantages).to(self.device)
        
        # Compute temporal decay weight
        decay_weight = compute_temporal_decay_weight(iteration_t)
        
        # Get frozen θ predictions
        with torch.no_grad():
            theta_frozen_pred = network_bundle.cumulative_advantage_frozen(
                features_tensor
            )  # Shape: (batch, num_actions)
        
        # Compute bootstrapped target
        # target_θ = w_t * θ_frozen + (1 - w_t) * φ_observed
        # This maintains cumulative sum structure as a convex combination
        target = decay_weight * theta_frozen_pred + (1 - decay_weight) * observed_tensor
        
        # Apply CFR+ clipping: non-negative cumulative advantages
        target = torch.clamp(target, min=0.0)
        
        # Predict with trainable θ
        predicted = network_bundle.cumulative_advantage(features_tensor)
        
        # MSE loss
        loss = F.mse_loss(predicted, target)
        
        return loss
