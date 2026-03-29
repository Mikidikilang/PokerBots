"""
Counterfactual Value Computation (cfr_valuator.py).

[PHASE 2] Core mathematics of Deep CFR.

This module computes COUNTERFACTUAL VALUES from game trajectories:

    V^t_i(h) = Expected value to player i from infoset h under current strategy

Unlike traditional RL rewards (scalar outcome), CFR valuates ACTIONS:

    Counterfactual Value of Action a:
    V̂^t_i(h, a) = Expected value IF we always played action a at this point,
                    but both players still followed σ^t elsewhere

    Counterfactual Regret:
    R^t(h, a) = V̂^t(h, a) - V^t(h)
    
    (What we would have gained by playing a instead of our actual action)

---

HEADS-UP SIMPLIFICATION (Phase 1 constraint)
----------------------------------------------

In 2-player zero-sum poker:
    V^t_0(h) = Expected value to hero
    V^t_1(h) = Expected value to villain = -V^t_0(h)  (zero-sum)

This simplifies counterfactual computation:
    - Only need to compute one player's value per state
    - Alternating player perspective handling is straightforward
    - No "phantom" player issues (multi-way gets complex)

Multi-way extension (Phase 2+):
    - Track value for each player separately
    - Requires "reach probability" weighting for absent players
    - Exponential state explosion risk

---

NEURAL NETWORK INTEGRATION
-----------------------------

Network outputs TWO quantities (actor-critic):
    1. action_logits[h] → policy π(a|h) via softmax
    2. value[h] → approximate V̂^t(h) (critic prediction)

Training procedure:
    1. Rollout trajectory with frozen network (importance sampling)
    2. Compute true value via bootstrap: V̂_target = reward + γ * V̂(s')
    3. Backprop: minimize (V̂_network(h) - V̂_target)^2
    4. Use V̂_network outputs to form counterfactual regrets
    5. Regret matching → next iteration's strategy

---

References
    - Brown & Sandholm (2019): "Solving Imperfect Information Games via
      Discounted Regret Minimization"
    - Srinivasan et al. (2018): "Actor-Critic Policy Optimization in Partially
      Observable Multiagent Environments"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GameNode:
    """Single decision node in game tree (infoset perspective)."""
    
    infoset_id: str                      # Unique identifier for this infoset
    player: int                          # 0 = hero, 1 = opponent (heads-up)
    current_player_reached: bool         # Did current player play to this node?
    obs_tensor: torch.Tensor             # Normalized observation [obs_dim]
    legal_actions: list[int]             # Valid action indices
    action_taken: int                    # Which action was actually played
    
    # For counterfactual value computation
    reach_prob_hero: float = 1.0         # P(reach this node | hero's strategy)
    reach_prob_opponent: float = 1.0     # P(reach this node | opponent's strategy)


def compute_counterfactual_values(
    trajectory: list[GameNode],
    final_reward: float,
    network: nn.Module,
    device: torch.device,
    discount_factor: float = 1.0,
) -> dict[str, dict[int, float]]:
    """
    [CORE CFR COMPUTATION]
    
    Compute counterfactual values for all (infoset, action) pairs in trajectory.
    
    Algorithm:
        1. Forward pass: network outputs V̂(h) for each node h
        2. Backward pass: compute counterfactual values via game tree rollback
        3. For each action a at node h:
           - Compute V̂(h, a) = value IF we had taken action a (hypothetically)
           - Regret(h, a) = V̂(h, a) - V̂(h)
    
    Args:
        trajectory: List of GameNode for this hand (first to last decision)
        final_reward: Chip EV outcome (signed, e.g., +50 or -30)
        network: Actor-critic network (outputs action_logits, value)
        device: PyTorch device
        discount_factor: γ for bootstrapping (usually 1.0 for episodic poker)
    
    Returns:
        {infoset_id: {action_idx: counterfactual_regret_value}}
    """
    
    if not trajectory:
        return {}
    
    # Step 1: Forward pass — get network value estimates for all nodes
    obs_batch = torch.stack([node.obs_tensor for node in trajectory]).to(device)
    with torch.no_grad():
        action_logits, value_estimates = network(obs_batch)
        # action_logits: [len(trajectory), 12]
        # value_estimates: [len(trajectory), 1]
        value_estimates = value_estimates.squeeze(-1)  # [len(trajectory)]
    
    counterfactual_regrets: dict[str, dict[int, float]] = {}
    
    # Step 2: Backward pass — compute counterfactual values
    # Start from end of trajectory and work backward
    
    # Bootstrap value at terminal node
    next_value = final_reward  # Terminal: realized reward is the value
    
    for node_idx in range(len(trajectory) - 1, -1, -1):
        node = trajectory[node_idx]
        current_value = value_estimates[node_idx].item()
        
        # For each legal action, compute counterfactual value
        # V̂(h, a) approximates: "what if we took action a here?"
        
        # In practice, we use the network's output as V̂(h, a)
        # More sophisticated approaches would:
        #   1. Compute separate values for each action via tree search
        #   2. Use on-policy/off-policy corrections (importance sampling)
        
        # For Phase 2 MVP: use network value directly
        # TODO: Implement proper counterfactual tree evaluation
        
        if node.infoset_id not in counterfactual_regrets:
            counterfactual_regrets[node.infoset_id] = {}
        
        # Placeholder counterfactual value for each legal action
        # Real implementation: evaluate each branch of game tree
        for action_idx in node.legal_actions:
            if action_idx == node.action_taken:
                # Action we actually played: use realized value
                counterfactual_value = next_value
            else:
                # Action we didn't play: approximate via network
                # In reality, would need full game tree evaluation
                counterfactual_value = current_value
            
            regret = counterfactual_value - current_value
            counterfactual_regrets[node.infoset_id][action_idx] = regret
        
        # Prepare for next iteration (moving backward through tree)
        next_value = discount_factor * current_value + (1 - discount_factor) * final_reward
    
    return counterfactual_regrets


def compute_value_targets(
    trajectory: list[GameNode],
    final_reward: float,
    discount_factor: float = 0.99,
) -> torch.Tensor:
    """
    Compute bootstrapped value targets for value function training.
    
    Used to train the critic (value head) of the network.
    
    Standard N-step return:
        V_target(h_t) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n*V̂(h_{t+n})
    
    For poker (episodic, discount=1.0):
        V_target(h_t) = final_reward (all N-step returns collapse to same terminal value)
    
    Args:
        trajectory: Game nodes
        final_reward: Realized chip EV
        discount_factor: γ
    
    Returns:
        torch.Tensor [len(trajectory)] of value targets
    """
    targets = []
    
    for _ in trajectory:
        # In episodic setting (poker hand), all nodes see same terminal reward
        targets.append(final_reward)
    
    # Could use bootstrapping for multi-step returns, but for episodic poker
    # (one hand = one episode) the target is always just final_reward
    return torch.tensor(targets, dtype=torch.float32)


def bootstrap_counterfactual_values(
    node: GameNode,
    network_value: float,
    next_node_value: float,
    discount_factor: float = 1.0,
) -> float:
    """
    Bootstrap counterfactual value using Bellman equation.
    
    V^t(h) ≈ E[r + γ*V^t(h')]  (temporal difference)
    
    Args:
        node: Current game node
        network_value: Network's value estimate V̂(h)
        next_node_value: Value of next node in trajectory
        discount_factor: γ
    
    Returns:
        Bootstrapped value estimate
    """
    # Simple TD(0) bootstrap: use network value as approximation
    return network_value + discount_factor * (next_node_value - network_value)


def importance_weight_counterfactual_value(
    infoset_id: str,
    action_taken: int,
    reach_prob_off_policy: float,
    reach_prob_target: float,
    counterfactual_value: float,
) -> float:
    """
    Apply importance weighting for off-policy counterfactual value estimates.
    
    Used when network was trained on different (off-policy) data distribution.
    
    Weight = P(reach in target policy) / P(reach in sampling policy)
    
    Args:
        infoset_id: Hash of game state
        action_taken: Which action was sampled
        reach_prob_off_policy: P(reach this node | behavior policy)
        reach_prob_target: P(reach this node | target policy)
        counterfactual_value: Value before weighting
    
    Returns:
        Importance-weighted counterfactual value
    """
    if reach_prob_off_policy < 1e-6:
        logger.warning(
            "Near-zero reach probability (%.2e) for infoset %s action %d: "
            "off-policy samples may be numerically unstable",
            reach_prob_off_policy, infoset_id, action_taken
        )
        return 0.0
    
    importance_weight = reach_prob_target / reach_prob_off_policy
    
    # Truncate weights to prevent extreme variance
    #importance_weight = min(importance_weight, 5.0)  # Max weight = 5x
    
    return importance_weight * counterfactual_value


def compute_exploitability_lower_bound(
    regrets: dict[str, dict[int, float]],
    infoset_reach_probs: dict[str, float],
) -> float:
    """
    [CONVERGENCE METRIC]
    
    Compute lower bound on exploitability (distance from Nash equilibrium).
    
    Theorem (Hart & Mas-Colell 1999):
        Exploitability ≤ √[Σ_i (sum of squared regrets at h) / T]
    
    where T = number of iterations.
    
    Args:
        regrets: All accumulated regrets across infosets
        infoset_reach_probs: P(reach each infoset under current strategy)
    
    Returns:
        Exploitability bound (in chips per hand)
    """
    total_squared_regret = 0.0
    
    for infoset_id, action_regrets in regrets.items():
        reach_prob = infoset_reach_probs.get(infoset_id, 0.0)
        
        for action_idx, regret in action_regrets.items():
            # Regret is scaled by reach probability
            # (regrets at unlikely infosets matter less)
            total_squared_regret += (regret ** 2) * reach_prob
    
    # Bound is O(√ regret_sum / T)
    # Compute exploitability ≈ √ sum
    exploitability = (total_squared_regret ** 0.5) if total_squared_regret > 0 else 0.0
    
    return exploitability
