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

from .cfr_env_state import EnvStateManager

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
    env: Any,
    env_state_manager: EnvStateManager,
    infoset_id: str,
    player_to_update: int,
    legal_actions: list[int],
    network: nn.Module,
    infoset_storage: Any,
    device: torch.device,
    obs_builder: Any = None,
    depth: int = 0,
    max_depth: int = 50,
) -> float:
    """
    [CORE CFR COMPUTATION — PROPER GAME TREE TRAVERSAL]
    
    Recursively traverse game tree, computing counterfactual values for each
    (infoset, action) pair by actually stepping the environment.
    
    **Algorithm: External Sampling MCCFR**
    
    For each legal action at this infoset:
        1. Save environment state
        2. Step environment with action
        3. Recursively evaluate resulting subtree
        4. Restore environment state
        5. Accumulate regrets based on subtree value
    
    Args:
        env: RLCard environment
        env_state_manager: EnvStateManager for copy-on-enter, restore-on-exit
        infoset_id: Hash of current information set
        player_to_update: Which player's regrets to update (0 or 1)
        legal_actions: Valid action indices from this state
        network: Actor-critic network (for value estimates)
        infoset_storage: InformationSetStorage for regret accumulation
        device: PyTorch device
        obs_builder: ObservationBuilder for observation generation
        depth: Current depth in tree (for logging)
        max_depth: Maximum depth to traverse (avoids infinite loops)
    
    Returns:
        float: Counterfactual value of this node from player_to_update's perspective
    """
    
    # Terminal condition: max depth reached
    if depth > max_depth:
        logger.debug(f"Max depth {max_depth} reached; returning 0.0")
        return 0.0
    
    # Get current player and check if game is terminal
    current_player = env._env.get_player_num()
    done = env._env.is_over()
    
    if done:
        # Terminal node: return payoff
        payoffs = env._env.get_payoffs()
        payoff = payoffs[player_to_update] if player_to_update < len(payoffs) else 0.0
        logger.debug(f"Terminal node at depth {depth}: payoff={payoff}")
        return float(payoff)
    
    # Build observation tensor for network value estimate
    if obs_builder is not None:
        try:
            obs_dict = obs_builder.build(env._env)  # type: ignore
            obs_tensor = obs_dict.get("obs_tensor", torch.zeros(1)).to(device)
        except Exception as e:
            logger.warning(f"ObservationBuilder.build() failed: {e}; using zeros")
            obs_tensor = torch.zeros(1).to(device)
    else:
        obs_tensor = torch.zeros(1).to(device)
    
    # Get network's value estimate for this node
    with torch.no_grad():
        batch_obs = obs_tensor.unsqueeze(0) if obs_tensor.dim() == 1 else obs_tensor
        try:
            action_logits, value_estimate = network(batch_obs)
            value_self = value_estimate.squeeze().item()
        except Exception as e:
            logger.warning(f"Network forward pass failed: {e}; using 0.0")
            value_self = 0.0
    
    # =========================================================================
    # MCCFR Tree Traversal: For each legal action
    # =========================================================================
    
    counterfactual_regrets_at_node: dict[int, float] = {}
    total_node_value = 0.0
    
    # Get current player's strategy from regrets (regret matching)
    infoset = infoset_storage.get_infoset(infoset_id)
    if infoset:
        strategy = infoset.get_strategy(legal_actions)
    else:
        # Unseen infoset: uniform strategy
        strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}
    
    subtree_values: dict[int, float] = {}
    
    for action_idx in legal_actions:
        # Save environment state
        with env_state_manager.savepoint():
            # Step environment with this action
            obs, reward, done, info = env.step(action_idx)
            
            # Recursively evaluate subtree
            subtree_value = compute_counterfactual_values(
                env=env,
                env_state_manager=env_state_manager,
                infoset_id=f"{infoset_id}@{action_idx}",  # Unique ID for subtree
                player_to_update=player_to_update,
                legal_actions=list(env._env.legal_actions) if hasattr(env._env, 'legal_actions') else [],
                network=network,
                infoset_storage=infoset_storage,
                device=device,
                obs_builder=obs_builder,
                depth=depth + 1,
                max_depth=max_depth,
            )
        
        subtree_values[action_idx] = subtree_value
        
        # Accumulate weighted value
        action_prob = strategy.get(action_idx, 1.0 / len(legal_actions))
        total_node_value += action_prob * subtree_value
    
    # =========================================================================
    # Compute Counterfactual Regrets
    # =========================================================================
    
    # If this is a node where player_to_update acts, compute regrets
    if current_player == player_to_update:
        for action_idx in legal_actions:
            action_value = subtree_values[action_idx]
            counterfactual_regret = action_value - total_node_value
            counterfactual_regrets_at_node[action_idx] = counterfactual_regret
            
            # Add regret to storage (for regret matching in next iteration)
            infoset_storage.add_regret(infoset_id, action_idx, counterfactual_regret)
        
        logger.debug(
            f"Depth {depth}: Updated regrets at {infoset_id}, "
            f"node_value={total_node_value:.2f}, regrets={counterfactual_regrets_at_node}"
        )
    else:
        # Opponent's node: just return expected value (no regret update)
        logger.debug(f"Depth {depth}: Opponent node (no regret update), value={total_node_value:.2f}")
    
    return total_node_value



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
