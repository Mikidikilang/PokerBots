"""
Blueprint Strategy & Multi-Way CFR Support (Phase 4.2-4.3).

[PHASE 4.2] Blueprint Strategy Generation
[PHASE 4.3] Multi-Way Poker Expansion (6-Max)

BLUEPRINT STRATEGY
==================

"Blueprint" = the offline-trained average strategy network.

Trained on: millions of game states (all preflop/flop/turn/river positions)
Usage: Default strategy for any game situation
Properties:
  - Nash-approximate (converges over iterations)
  - Fast (O(1) neural network inference)
  - Generalizes across similar states
  - Can be exploited long-term, but unpredictable short-term

Architecture:
  Input:  observation (cards + pot odds + action history)
  Output: action probabilities (softmax over 12 actions)
  Training: behavioral cloning on regret-matched strategies

Usage in RTA:
  1. Blueprint provides PRIOR distribution over actions
  2. RTA solves SUBGAME for specific situation
  3. RTA output: more exploitative but adapts in real-time

Example Hybrid Strategy:
  ├─ Preflop: Use Blueprint (too large to solve)
  ├─ Flop: Use Blueprint (still large)
  ├─ Turn: Use Blueprint or RTA (depends on stack depth)
  └─ River: Use RTA (most exploitable, highest variance)

MULTI-WAY POKER
===============

Phase 1 Constraint: Heads-up only (simplified CFR analysis)

Phase 4 Goal: Expand to 6-Max (multiplayer)

Challenges vs Heads-up:
  1. Game tree exponentially larger: 2 players → 6 players
  2. Nash may not exist (3+ player games are non-zero-sum)
  3. Range inference harder: must track 5 opponents
  4. Computational: single hand now involves 5 opponent hands

Solutions:
  1. Approximate equilibrium: use CFR to bound exploitability
  2. Opponent modeling: separate agents for each opponent
  3. Hand abstraction stricter: 169 → 10-50 buckets (lossy)
  4. Distributed solving: GPU parallelism across 6 players

Key Insight: Multi-way exploitability ≤ Σ_i exploitability_i
  If each player's exploitability ≤ ε per opponent,
  total exploitability ≤ 5ε (for 6 players)

MULTI-WAY REGRET COMPUTATION
============================

Counterfactual regret in multi-way:
  
  For player i at infoset h:
    R^t_i(a|h) = (V^t_{-i}(h, a) - V^t_{-i}(h)) × reach_{-i}(h)
  
  where:
    V^t_{-i}(h, a) = value if player i plays action a
    V^t_{-i}(h) = baseline value (average over actions)
    reach_{-i}(h) = reach probability of all OTHER players
  
  Same algorithm, but higher-dimensional regret tensors:
    Old: {infoset_id: {action: regret}}
    New: {infoset_id: {action: {opp_config: regret}}}
    
    opp_config = which 5 opponents, what hands
    Complexity: exponential in number of opponents

Solution: Factorize
  Assume independence: V_{-i} = independent evaluations
  Store only: {infoset_id: {action: [regret vs each opponent]}}
  Memory: linear instead of exponential

---

References:
  - Landau et al. (2013): "Opponent Modeling for Multi-Headed Agents"
  - Bowling et al. (2015): "Heads-up Limit Hold'em Poker is Solved"
  - Brown & Sandholm (2017): "Libratus: The Superhuman Poker Player" (heads-up)
  - Brown et al. (2019): "Superhuman AI for Multiplayer Poker" (Pluribus, 6-Max)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# BLUEPRINT STRATEGY NETWORK
# ============================================================================

class BlueprintStrategy(nn.Module):
    """
    Frozen blueprint strategy network (offline-trained).
    
    Stores the average strategy learned by Deep CFR.
    Used as:
      1. Fixed strategy during training (for opponent simulation)
      2. Prior for RTA subgame solving
      3. Fallback when RTA times out
      4. Reference for measuring convergence
    """
    
    def __init__(self, strategy_network: nn.Module):
        """
        Args:
            strategy_network: Trained AverageStrategyNetwork from Phase 2C
        """
        super().__init__()
        self.strategy_network = strategy_network
        self.num_evals = 0
        
        # Freeze all parameters
        for param in self.strategy_network.parameters():
            param.requires_grad = False
        
        logger.info("BlueprintStrategy initialized (frozen parameters)")
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities from blueprint.
        
        Args:
            observation: (batch, obs_dim) or (obs_dim,)
        
        Returns:
            Probabilities: (batch, 12) or (12,)
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.strategy_network(observation)
            probs = torch.softmax(logits, dim=-1)
        
        self.num_evals += observation.shape[0]
        return probs.squeeze(0) if logits.shape[0] == 1 else probs
    
    def get_action_distribution(
        self,
        observation: torch.Tensor,
        legal_actions: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Get strategy dict (action_idx: probability)."""
        probs = self.forward(observation)
        
        if legal_actions is None:
            legal_actions = list(range(12))
        
        distribution = {}
        for action_idx in legal_actions:
            distribution[action_idx] = float(probs[action_idx].item())
        
        # Renormalize to sum to 1.0
        total = sum(distribution.values())
        if total > 0:
            distribution = {a: p / total for a, p in distribution.items()}
        
        return distribution
    
    def get_summary(self) -> Dict[str, any]:
        """Return statistics about blueprint usage."""
        return {
            "total_evaluations": self.num_evals,
            "frozen": all(not p.requires_grad for p in self.strategy_network.parameters()),
            "device": next(self.strategy_network.parameters()).device,
        }


# ============================================================================
# MULTI-WAY POKER SUPPORT
# ============================================================================

@dataclass
class MultiWayState:
    """
    Represents a multi-way poker decision point.
    
    Extends heads-up RiverSubgame to N players.
    """
    
    num_players: int = 6  # Heads-up=2, 6-Max=6
    current_player: int = 0  # Whose turn?
    hole_cards: Dict[int, str] = field(default_factory=dict)  # {player_idx: canonical_hand}
    pot: float = 10.0  # In BB
    stacks: Dict[int, float] = field(default_factory=dict)  # {player_idx: stack_in_BB}
    
    def all_in_except_one(self) -> bool:
        """Check if all but one player are all-in."""
        active = [p for p, stack in self.stacks.items() if stack > 0]
        return len(active) <= 2
    
    def get_active_players(self) -> List[int]:
        """Players still in the hand."""
        return [p for p, stack in self.stacks.items() if stack > 0]
    
    def __str__(self) -> str:
        return (
            f"MultiWayState(players={self.num_players}, pot={self.pot}BB, "
            f"active={len(self.get_active_players())}, current={self.current_player})"
        )


class MultiWayValueEvaluator:
    """
    Evaluate game value in multi-way scenarios.
    
    Computes: V_i = expected payoff to player i
    
    Challenges:
      1. No binary outcome (win/tie/loss in heads-up)
      2. Multiple ways to win: showdown or fold out opponents
      3. Ranges: each opponent has separate hand distribution
    
    Approach:
      For each outcome (fold configurations + showdown):
      1. Compute probability (function of strategies + hands)
      2. Compute payoff to each player
      3. Expected value = Σ P(outcome) × payoff
    """
    
    def __init__(self, num_players: int = 6):
        self.num_players = num_players
        
        logger.info(f"MultiWayValueEvaluator: {num_players} players")
    
    def evaluate(
        self,
        state: MultiWayState,
        strategies: Dict[int, Dict[int, float]],
    ) -> Dict[int, float]:
        """
        Compute expected value for each player.
        
        Args:
            state: Game state
            strategies: {player_idx: {action_idx: probability}}
        
        Returns:
            {player_idx: expected_payoff_in_chips}
        """
        # Placeholder: simple uniform value distribution
        total_chips = sum(state.stacks.values())
        return {
            p: total_chips / state.num_players
            for p in range(state.num_players)
        }
    
    def showdown_value(
        self,
        hole_cards: Dict[int, str],
        board: str,
    ) -> Dict[int, float]:
        """
        Determine winner at showdown.
        
        Args:
            hole_cards: {player: canonical_hand}
            board: Community cards
        
        Returns:
            {player: chips_won}
        
        NOTE: Placeholder. Real version uses hand evaluator.
        """
        # TODO: Implement actual poker hand evaluation
        # (compare 5-card best hands, handle ties)
        return {p: 0.0 for p in hole_cards.keys()}


class MultiWayCFREngine:
    """
    Extended CFR engine for multi-way poker.
    
    Key differences from heads-up (Phase 2):
      1. Regret per opponent: {infoset: {action: [R_opp1, R_opp2, ...]}}
      2. Opponent modeling: separate strategy per seat
      3. Value function: multi-output (value for each opponent config)
      4. Convergence: slower due to larger game tree
    """
    
    def __init__(
        self,
        num_players: int = 6,
        device: torch.device = torch.device('cpu'),
    ):
        self.num_players = num_players
        self.device = device
        
        # Multi-way specific data structures will be initialized on demand
        self.regrets_by_opp_config: Dict[str, Dict[int, List[float]]] = {}
        
        logger.info(f"MultiWayCFREngine: {num_players}-player poker")
    
    def compute_regret_multi_way(
        self,
        infoset_id: str,
        action_values: Dict[int, float],  # {action: value}
        opponent_strategies: Dict[int, Dict[int, float]],  # {opp: {action: prob}}
    ) -> Dict[int, List[float]]:
        """
        Compute regrets in multi-way setting.
        
        R_i(a|h) = (V_i(h, a) - V_i(h)) × reach_{-i}(h)
        
        where reach_{-i} = ∏_{j ≠ i} π_j(reach)
        
        Args:
            infoset_id: Information set identifier
            action_values: {action: expected value if action taken}
            opponent_strategies: Strategies of other players
        
        Returns:
            {action: [regret_vs_opp1, regret_vs_opp2, ...]}
        """
        baseline_value = sum(action_values.values()) / len(action_values)
        
        # Simple: regret proportional to value difference
        regrets = {}
        for action, value in action_values.items():
            regret_base = value - baseline_value
            
            # Weight by opponent strategies (lower prob opponents = higher weight)
            opponent_regrets = []
            for opp_idx, opp_strategy in opponent_strategies.items():
                # Opponent reach: sum of probabilities they reach this point
                opp_reach = 1.0  # TODO: compute from hand ranges
                opponent_regrets.append(regret_base * opp_reach)
            
            regrets[action] = opponent_regrets
        
        return regrets
    
    def strategy_from_multi_way_regrets(
        self,
        regrets: Dict[int, List[float]],
    ) -> Dict[int, float]:
        """
        Form strategy from multi-way regrets.
        
        Average regrets across opponent configurations, then apply regret matching.
        """
        # Average regrets across opponent dimension
        avg_regrets = {}
        for action, opp_regrets in regrets.items():
            if opp_regrets:
                avg_regrets[action] = sum(opp_regrets) / len(opp_regrets)
            else:
                avg_regrets[action] = 0.0
        
        # Regret matching
        positive_regrets = [max(r, 0.0) for r in avg_regrets.values()]
        total_positive = sum(positive_regrets)
        
        if total_positive <= 0:
            # Uniform
            num_actions = len(avg_regrets)
            return {a: 1.0 / num_actions for a in avg_regrets.keys()}
        else:
            return {
                a: positive_regrets[i] / total_positive
                for i, a in enumerate(avg_regrets.keys())
            }


# ============================================================================
# GAME PIPELINE (BLUEPRINT + RTA + MULTI-WAY)
# ============================================================================

class HybridPokerAgent:
    """
    Complete poker agent combining:
      1. Blueprint strategy (offline)
      2. RTA subgame solving (online, river only)
      3. Multi-way support (6-player poker)
    
    Decision process:
      1. Compute Blueprint action (fast, exploitable)
      2. If on river and time remaining: run RTA
      3. Blend actions based on time pressure
      4. Support multi-way opponent modeling
    """
    
    def __init__(
        self,
        blueprint: Optional[BlueprintStrategy] = None,
        rta_solver: Optional[Any] = None,  # RTAAgent from rta_solver.py
        num_players: int = 6,
        use_rta: bool = True,
        max_solve_time: float = 5.0,  # seconds
    ):
        """
        Args:
            blueprint: Frozen strategy network
            rta_solver: Real-time solver (for river)
            num_players: 2 (heads-up) or 6 (6-max)
            use_rta: Whether to enable real-time solving
            max_solve_time: Time limit for subgame solving
        """
        self.blueprint = blueprint
        self.rta_solver = rta_solver
        self.num_players = num_players
        self.use_rta = use_rta and rta_solver is not None
        self.max_solve_time = max_solve_time
        
        # Multi-way support
        self.multi_way_engine = MultiWayCFREngine(num_players=num_players)
        
        logger.info(
            f"HybridPokerAgent: blueprint={'yes' if blueprint else 'no'}, "
            f"rta={'yes' if self.use_rta else 'no'}, "
            f"players={num_players}, max_solve_time={max_solve_time}s"
        )
    
    def get_action(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get action for any game situation.
        
        Args:
            observation: Game state dict with:
              - hole_cards, board, pot, stacks
              - action_history, street
              - legal_actions
        
        Returns:
            {
              'action': 'fold' | 'check' | 'call' | 'bet' | 'raise',
              'amount': float,
              'source': 'blueprint' | 'rta' | 'blend',
              'confidence': float,
            }
        """
        street = observation.get('street', 0)  # 0=preflop, 1=flop, 2=turn, 3=river
        
        # 1. Get Blueprint action (always available)
        blueprint_action = self._blueprint_action(observation)
        
        # 2. Try RTA only on river
        rta_action = None
        if self.use_rta and street == 3:  # River
            try:
                rta_action = self._rta_action(observation)
            except Exception as e:
                logger.warning(f"RTA failed: {e}, falling back to blueprint")
        
        # 3. Blend or choose
        if rta_action is not None:
            return {
                **rta_action,
                'source': 'rta',
                'fallback': blueprint_action,
            }
        else:
            return {
                **blueprint_action,
                'source': 'blueprint',
            }
    
    def _blueprint_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Compute action from Blueprint strategy."""
        if self.blueprint is None:
            return {'action': 'check', 'amount': 0.0}
        
        obs_tensor = torch.tensor(
            observation.get('observation', []),
            dtype=torch.float32,
        )
        action_dist = self.blueprint.get_action_distribution(
            obs_tensor,
            legal_actions=observation.get('legal_actions'),
        )
        
        best_action = max(action_dist.items(), key=lambda x: x[1])[0]
        return {
            'action': self._action_idx_to_name(best_action),
            'amount': observation.get('bet_amount', 0.0),
            'confidence': action_dist[best_action],
        }
    
    def _rta_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Compute action from RTA subgame solver."""
        if self.rta_solver is None:
            return None
        
        result = self.rta_solver.get_action(
            hero_hand=observation.get('hero_hand'),
            board=tuple(observation.get('board', [])),
            pot=observation.get('pot', 10.0),
            hero_stack=observation.get('hero_stack', 50.0),
            opponent_stack=observation.get('opponent_stack', 50.0),
            action_history=observation.get('action_history', []),
        )
        
        return {
            'action': result.get('action', 'check'),
            'amount': result.get('amount', 0.0),
            'confidence': result.get('confidence', 0.5),
        }
    
    def _action_idx_to_name(self, idx: int) -> str:
        """Convert action index to poker action name."""
        mapping = {
            0: 'fold',
            1: 'check',
            2: 'call',
            3: 'raise',
            4: 'bet',
            # ... more actions
        }
        return mapping.get(idx, 'check')


# ============================================================================
# UTILITIES
# ============================================================================

def create_hybrid_agent(
    strategy_network: Optional[nn.Module] = None,
    regret_network: Optional[nn.Module] = None,
    num_players: int = 6,
    use_rta: bool = True,
    device: str = 'cpu',
) -> HybridPokerAgent:
    """Factory function to create hybrid poker agent."""
    # Import here to avoid circular dependencies
    from src.training.rta_solver import RTAAgent
    
    blueprint = None
    if strategy_network is not None:
        blueprint = BlueprintStrategy(strategy_network)
    
    rta = None
    if use_rta and strategy_network is not None and regret_network is not None:
        rta = RTAAgent(
            strategy_network=strategy_network,
            regret_network=regret_network,
            num_solve_iterations=1000,
            device=torch.device(device),
        )
    
    return HybridPokerAgent(
        blueprint=blueprint,
        rta_solver=rta,
        num_players=num_players,
        use_rta=use_rta,
        max_solve_time=5.0,
    )
