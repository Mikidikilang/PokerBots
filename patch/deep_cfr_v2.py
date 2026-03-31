"""
Intervention 2: Correct Deep CFR with Neural Feedback Loop
============================================================

The fundamental wiring error in the original codebase:
    - Traversal queries tabular infoset_storage for opponent strategies
    - Neural regret network is trained but never consulted during traversal
    - This is tabular MCCFR + unused neural network, NOT Deep CFR

Correct Deep CFR data flow:
    ┌──────────────────────────────────────────────────────────────────┐
    │  For iteration t = 1, 2, ..., T:                                 │
    │                                                                  │
    │  ① TRAVERSAL (uses regret network from iteration t-1)           │
    │     For each player i:                                           │
    │       traverse(root, player=i, π_regret_net_{t-1})              │
    │       → collects (obs, legal, counterfactual_regrets) tuples     │
    │       → stored in ReservoirBuffer M_i                            │
    │                                                                  │
    │  ② REGRET NETWORK UPDATE                                        │
    │     Sample batch B from M_i                                      │
    │     Train V_θ: obs → predicted_regrets                          │
    │     Loss: Σ (a∈legal) [V_θ(obs)[a] - true_CF_regret[a]]²       │
    │                                                                  │
    │  ③ STRATEGY ACCUMULATION                                        │
    │     For each infoset visited in ①:                              │
    │       σ^t(a|h) = regret_match(V_θ^t(h))                        │
    │       M_σ.add(obs, legal, σ^t)  ← weighted by t (linear avg)   │
    │                                                                  │
    │  ④ STRATEGY NETWORK UPDATE                                      │
    │     Sample batch from M_σ (weighted by iteration t)             │
    │     Train Π_φ: obs → action_probs (behavioral cloning on σ̄)    │
    │                                                                  │
    │  KEY: Step ① uses V_θ from PREVIOUS iteration                   │
    │       to compute action probs π(a|h) ∝ max(V_θ(h)[a], 0)       │
    └──────────────────────────────────────────────────────────────────┘

The critical invariant: the regret network IS the traversal policy.
Without this feedback loop, the network cannot generalize.

References
----------
- Brown, Lerer, Gross & Sandholm (2019): "Deep Counterfactual Regret Minimization"
- Steinberg, Ganzfried & Sandholm (2021): "Deep CFR in Theory"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .regret_store import RegretStore, NUM_ACTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural Network Definitions
# ---------------------------------------------------------------------------

class RegretNetwork(nn.Module):
    """
    V_θ: obs → counterfactual_regrets[NUM_ACTIONS]

    This network defines the traversal policy via regret matching:
        π(a|h) = max(V_θ(h)[a], 0) / Σ_{a'} max(V_θ(h)[a'], 0)

    Architecture: wide MLP with layer normalization (critical for stability
    when regret magnitudes vary by 3–4 orders of magnitude across the tree).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...] = (1024, 1024, 512, 256),
        num_actions: int = NUM_ACTIONS,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_actions))
        # No activation — regrets can be positive or negative
        self.net = nn.Sequential(*layers)

        # Initialize output layer near zero (uniform initial strategy)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            regrets: (batch, num_actions) — raw counterfactual regret predictions
        """
        return self.net(obs)

    def get_strategy(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert regret predictions to strategy via regret matching.

        Args:
            obs:        (batch, obs_dim)
            legal_mask: (batch, num_actions) — 1.0 for legal, 0.0 for illegal
        Returns:
            probs: (batch, num_actions) — strategy probabilities
        """
        with torch.no_grad():
            regrets = self.forward(obs)

        # Mask illegal actions
        masked = regrets * legal_mask
        # CFR+: clamp to zero (only positive regrets drive strategy)
        positive = F.relu(masked)

        # Normalize
        total = positive.sum(dim=-1, keepdim=True)
        uniform = legal_mask / (legal_mask.sum(dim=-1, keepdim=True) + 1e-9)

        # Use uniform when all regrets ≤ 0
        use_uniform = (total < 1e-9).float()
        strat = (1.0 - use_uniform) * positive / (total + 1e-9) + use_uniform * uniform

        return strat


class AverageStrategyNetwork(nn.Module):
    """
    Π_φ: obs → σ̄(a|h)  (the Nash-converging average strategy)

    This is the network actually used at inference / deployment time.
    Trained via behavioral cloning on iteration-weighted strategy samples.

    Uses a separate, wider architecture from the regret network because
    it needs to memorize the average strategy across ALL iterations,
    while the regret network only needs to fit the current iteration's
    counterfactual values.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...] = (2048, 2048, 1024, 512),
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits (before softmax)."""
        return self.net(obs)

    def get_probs(self, obs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        logits = logits + (legal_mask - 1.0) * 1e9  # Mask illegal with -inf
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Reservoir Buffers
# ---------------------------------------------------------------------------

@dataclass
class RegretSample:
    obs: np.ndarray           # flattened observation
    legal_mask: np.ndarray    # binary [NUM_ACTIONS]
    regrets: np.ndarray       # counterfactual regrets [NUM_ACTIONS]
    iteration: int


@dataclass
class StrategySample:
    obs: np.ndarray           # flattened observation
    legal_mask: np.ndarray    # binary [NUM_ACTIONS]
    strategy: np.ndarray      # σ^t(a|h) [NUM_ACTIONS]
    weight: float             # linear weight = iteration t (for iteration-weighted avg)
    iteration: int


class ReservoirBuffer:
    """
    Reservoir sampling buffer ensuring uniform coverage of game tree.

    Vitter's Algorithm R: each new sample replaces an existing sample
    with probability buffer_size / (samples_seen + 1).

    This gives uniform distribution over ALL samples ever seen,
    regardless of game tree visit frequency — critical for convergence.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.n_seen: int = 0

    def add(self, sample: Any) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            j = np.random.randint(0, self.n_seen + 1)
            if j < self.capacity:
                self.buffer[j] = sample
        self.n_seen += 1

    def sample_batch(self, batch_size: int) -> Optional[List[Any]]:
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Deep CFR Engine (Correct Implementation)
# ---------------------------------------------------------------------------

class DeepCFREngine:
    """
    Correct Deep CFR with neural feedback loop.

    The regret network from iteration t-1 is used to define action
    probabilities during iteration t's traversal. This is the feedback
    loop that was missing from the original codebase.

    Key difference from original:
        ORIGINAL: traverse() → tabular_storage.get_strategy()
        THIS:     traverse() → regret_network.get_strategy()

    This enables generalization to unseen infosets — the core value
    proposition of Deep CFR over tabular CFR.
    """

    def __init__(
        self,
        obs_dim: int,
        env_factory: Any,           # callable() → env instance
        obs_builder: Any,           # ObservationBuilder
        regret_buffer_size: int = 4_000_000,
        strategy_buffer_size: int = 4_000_000,
        regret_hidden: Tuple[int, ...] = (1024, 1024, 512, 256),
        strategy_hidden: Tuple[int, ...] = (2048, 2048, 1024, 512),
        lr_regret: float = 1e-3,
        lr_strategy: float = 1e-3,
        batch_size: int = 10_000,
        n_regret_train_steps: int = 4000,
        n_strategy_train_steps: int = 4000,
        n_traversals_per_iter: int = 1500,
        device: str = "cpu",
        regret_store: Optional[RegretStore] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.obs_dim = obs_dim
        self.env_factory = env_factory
        self.obs_builder = obs_builder
        self.batch_size = batch_size
        self.n_regret_train_steps = n_regret_train_steps
        self.n_strategy_train_steps = n_strategy_train_steps
        self.n_traversals_per_iter = n_traversals_per_iter
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        # Neural networks
        self.regret_net = RegretNetwork(obs_dim, regret_hidden).to(self.device)
        self.strategy_net = AverageStrategyNetwork(obs_dim, strategy_hidden).to(self.device)

        # Separate buffers per player (indexed 0, 1 for heads-up)
        self.regret_buffers: List[ReservoirBuffer] = [
            ReservoirBuffer(regret_buffer_size),
            ReservoirBuffer(regret_buffer_size),
        ]
        self.strategy_buffer = ReservoirBuffer(strategy_buffer_size)

        # Optimizers
        self.regret_opt = torch.optim.Adam(
            self.regret_net.parameters(), lr=lr_regret
        )
        self.strategy_opt = torch.optim.Adam(
            self.strategy_net.parameters(), lr=lr_strategy
        )

        # Optional persistent regret store (for 10+TB scale)
        self.regret_store = regret_store

        self.iteration: int = 0
        self._traversal_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, n_iterations: int) -> Dict[str, float]:
        """
        Run n_iterations of Deep CFR.

        Returns final training statistics.
        """
        all_stats: Dict[str, float] = {}

        for t in range(n_iterations):
            iter_start = time.monotonic()
            self.iteration = t + 1
            logger.info("Deep CFR Iteration %d / %d", self.iteration, n_iterations)

            # ① TRAVERSAL: collect regret samples using current regret net
            trav_stats = self._run_traversals()

            # ② REGRET NETWORK UPDATE
            regret_losses = []
            for player in range(2):
                loss = self._train_regret_network(player)
                if loss is not None:
                    regret_losses.append(loss)
            mean_regret_loss = float(np.mean(regret_losses)) if regret_losses else 0.0

            # ③ + ④ STRATEGY ACCUMULATION + NETWORK UPDATE
            # Strategy samples were accumulated during traversal
            strategy_loss = self._train_strategy_network()

            elapsed = time.monotonic() - iter_start
            stats = {
                "iteration": float(self.iteration),
                "traversals": float(trav_stats.get("total", 0)),
                "regret_loss": mean_regret_loss,
                "strategy_loss": strategy_loss or 0.0,
                "regret_buf_0": float(len(self.regret_buffers[0])),
                "regret_buf_1": float(len(self.regret_buffers[1])),
                "strategy_buf": float(len(self.strategy_buffer)),
                "iter_time_s": elapsed,
            }
            all_stats.update(stats)

            logger.info(
                "  Iter %d: regret_loss=%.5f, strat_loss=%.5f, "
                "buf_sizes=[%d, %d], time=%.1fs",
                self.iteration, mean_regret_loss, stats["strategy_loss"],
                len(self.regret_buffers[0]), len(self.regret_buffers[1]),
                elapsed,
            )

            if self.checkpoint_dir and (t + 1) % 10 == 0:
                self._save_checkpoint()

        return all_stats

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _run_traversals(self) -> Dict[str, int]:
        """Run external sampling traversals for both players."""
        env = self.env_factory()
        total = 0

        for player in range(2):
            for _ in range(self.n_traversals_per_iter):
                state = env.reset()
                self._traverse(
                    env=env,
                    state=state,
                    player_to_update=player,
                    reach_p0=1.0,
                    reach_p1=1.0,
                    iteration=self.iteration,
                )
                total += 1

        return {"total": total}

    def _traverse(
        self,
        env: Any,
        state: Dict[str, Any],
        player_to_update: int,
        reach_p0: float,
        reach_p1: float,
        iteration: int,
        depth: int = 0,
    ) -> float:
        """
        External Sampling MCCFR traversal using regret NETWORK for strategies.

        THE KEY CHANGE: We call self.regret_net.get_strategy(obs, legal_mask)
        instead of infoset_storage.get_strategy() — the neural network IS
        the traversal policy.

        Args:
            reach_p0, reach_p1: Reach probabilities for each player
        Returns:
            Counterfactual value from player_to_update's perspective
        """
        if env.is_over():
            return self._terminal_payoff(env, player_to_update)

        if depth > 60:
            return 0.0  # Safety depth limit

        current_player = env._current_player_id
        legal_actions = state.get("legal_actions", list(range(NUM_ACTIONS)))
        if isinstance(legal_actions, dict):
            legal_actions = list(legal_actions.keys())

        # Build observation and legal mask
        obs_tensor, legal_mask = self._build_obs_and_mask(state, legal_actions)
        obs_np = obs_tensor.cpu().numpy()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRITICAL: Use regret NETWORK to get traversal strategy
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)
            strategy_t = self.regret_net.get_strategy(obs_t, mask_t)
            strategy_np = strategy_t.squeeze(0).cpu().numpy()

        if current_player == player_to_update:
            # ── Updating player: evaluate ALL actions ────────────────
            action_values: Dict[int, float] = {}
            saved_state = env.get_full_state()

            for i, action in enumerate(legal_actions):
                if i > 0:
                    env.set_full_state(saved_state)

                next_state, _ = env.step(action)

                new_reach_p0 = reach_p0 * strategy_np[action] if current_player == 0 else reach_p0
                new_reach_p1 = reach_p1 * strategy_np[action] if current_player == 1 else reach_p1

                v = self._traverse(
                    env, next_state, player_to_update,
                    new_reach_p0, new_reach_p1,
                    iteration, depth + 1,
                )
                action_values[action] = v

            env.set_full_state(saved_state)

            # Compute baseline value
            baseline = sum(strategy_np[a] * action_values[a] for a in legal_actions)

            # Compute counterfactual regrets
            opponent_reach = reach_p1 if current_player == 0 else reach_p0
            cf_regrets = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for a in legal_actions:
                cf_regrets[a] = opponent_reach * (action_values[a] - baseline)

            # ── Store regret sample in reservoir buffer ───────────────
            sample = RegretSample(
                obs=obs_np.copy(),
                legal_mask=legal_mask.copy(),
                regrets=cf_regrets.copy(),
                iteration=iteration,
            )
            self.regret_buffers[player_to_update].add(sample)

            # ── Accumulate strategy sample (for average strategy) ─────
            my_reach = reach_p0 if current_player == 0 else reach_p1
            strat_sample = StrategySample(
                obs=obs_np.copy(),
                legal_mask=legal_mask.copy(),
                strategy=strategy_np.copy(),
                weight=float(my_reach * iteration),  # iteration-weighted
                iteration=iteration,
            )
            self.strategy_buffer.add(strat_sample)

            # Also update persistent regret store (if configured)
            if self.regret_store is not None:
                infoset_key = self._state_to_key(state, current_player)
                action_regrets = {a: float(cf_regrets[a]) for a in legal_actions}
                self.regret_store.add_regrets_batch(
                    infoset_key, action_regrets, legal_actions, iteration
                )

            return baseline

        else:
            # ── Opponent: sample ONE action (external sampling) ───────
            action_probs = np.array([strategy_np[a] for a in legal_actions])
            action_probs = action_probs / (action_probs.sum() + 1e-9)
            sampled_action = np.random.choice(legal_actions, p=action_probs)
            sampled_prob = float(strategy_np[sampled_action])

            new_reach_p0 = reach_p0 * sampled_prob if current_player == 0 else reach_p0
            new_reach_p1 = reach_p1 * sampled_prob if current_player == 1 else reach_p1

            next_state, _ = env.step(sampled_action)

            return self._traverse(
                env, next_state, player_to_update,
                new_reach_p0, new_reach_p1,
                iteration, depth + 1,
            )

    # ------------------------------------------------------------------
    # Network training
    # ------------------------------------------------------------------

    def _train_regret_network(self, player: int) -> Optional[float]:
        """
        Train the regret network on samples from player's reservoir buffer.

        Loss: Σ_{a legal} (V_θ(obs)[a] - true_CF_regret[a])²
              (masked to legal actions only, following Brunner et al. 2019)
        """
        batch = self.regret_buffers[player].sample_batch(self.batch_size)
        if batch is None:
            return None

        self.regret_net.train()
        total_loss = 0.0
        n_steps = min(self.n_regret_train_steps, max(1, len(self.regret_buffers[player]) // self.batch_size * 2))

        for step in range(n_steps):
            if step > 0:
                batch = self.regret_buffers[player].sample_batch(self.batch_size)
                if batch is None:
                    break

            obs_np = np.stack([s.obs for s in batch])
            mask_np = np.stack([s.legal_mask for s in batch])
            regrets_np = np.stack([s.regrets for s in batch])

            obs_t = torch.from_numpy(obs_np).float().to(self.device)
            mask_t = torch.from_numpy(mask_np).float().to(self.device)
            target_t = torch.from_numpy(regrets_np).float().to(self.device)

            predicted = self.regret_net(obs_t)

            # MSE loss on legal actions only
            loss = ((predicted - target_t) ** 2 * mask_t).sum(dim=-1).mean()

            self.regret_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.regret_net.parameters(), 1.0)
            self.regret_opt.step()

            total_loss += loss.item()

        self.regret_net.eval()
        return total_loss / max(n_steps, 1)

    def _train_strategy_network(self) -> Optional[float]:
        """
        Train the average strategy network via iteration-weighted behavioral cloning.

        Samples are weighted by their iteration index t (linear weighting),
        implementing the time-average: σ̄ = (2/T(T+1)) Σ_t t·σ^t

        Loss: - Σ_{a legal} σ̄(a|h) · log Π_φ(a|h)  (cross-entropy)
        """
        batch = self.strategy_buffer.sample_batch(self.batch_size)
        if batch is None:
            return None

        self.strategy_net.train()
        total_loss = 0.0
        n_steps = min(self.n_strategy_train_steps, max(1, len(self.strategy_buffer) // self.batch_size * 2))

        for step in range(n_steps):
            if step > 0:
                batch = self.strategy_buffer.sample_batch(self.batch_size)
                if batch is None:
                    break

            obs_np = np.stack([s.obs for s in batch])
            mask_np = np.stack([s.legal_mask for s in batch])
            strat_np = np.stack([s.strategy for s in batch])
            weights_np = np.array([s.weight for s in batch], dtype=np.float32)

            # Normalize weights within batch
            weights_np = weights_np / (weights_np.sum() + 1e-9) * len(batch)

            obs_t = torch.from_numpy(obs_np).float().to(self.device)
            mask_t = torch.from_numpy(mask_np).float().to(self.device)
            target_t = torch.from_numpy(strat_np).float().to(self.device)
            weights_t = torch.from_numpy(weights_np).float().to(self.device)

            logits = self.strategy_net(obs_t)
            # Mask illegal with -inf
            logits_masked = logits + (mask_t - 1.0) * 1e9
            log_probs = F.log_softmax(logits_masked, dim=-1)

            # Weighted cross-entropy
            ce = -(target_t * log_probs * mask_t).sum(dim=-1)
            loss = (ce * weights_t).mean()

            self.strategy_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), 1.0)
            self.strategy_opt.step()

            total_loss += loss.item()

        self.strategy_net.eval()
        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _terminal_payoff(self, env: Any, player: int) -> float:
        """Extract real terminal payoff for player from environment."""
        try:
            payoffs = env._env.get_payoffs()
            bb = getattr(env, 'config', None)
            bb = bb.big_blind if bb is not None else 2.0
            return float(payoffs[player]) / bb
        except Exception:
            return 0.0

    def _build_obs_and_mask(
        self, state: Dict[str, Any], legal_actions: List[int]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Build flat observation tensor and binary legal action mask."""
        try:
            obs_dict = self.obs_builder.build(state)
            flat = self.obs_builder.flatten(obs_dict)
            obs = flat.float()
        except Exception:
            obs = torch.zeros(self.obs_dim, dtype=torch.float32)

        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in legal_actions:
            if 0 <= a < NUM_ACTIONS:
                mask[a] = 1.0

        return obs, mask

    def _state_to_key(self, state: Dict[str, Any], player: int) -> bytes:
        """Convert game state dict to infoset key bytes."""
        hand = sorted(state.get("hand", []))
        board = list(state.get("public_cards", []))
        history = [str(h.get("action", "")) for h in state.get("betting_history", [])]
        parts = [str(player), "|", ",".join(hand), "|", ",".join(board), "|", ",".join(history)]
        return "".join(parts).encode("utf-8")

    def get_action(self, state: Dict[str, Any], deterministic: bool = True) -> int:
        """
        Inference: query average strategy network for action.
        This is the network used during actual play, NOT the regret network.
        """
        legal_actions = state.get("legal_actions", list(range(NUM_ACTIONS)))
        if isinstance(legal_actions, dict):
            legal_actions = list(legal_actions.keys())

        obs_t, mask_np = self._build_obs_and_mask(state, legal_actions)
        obs_bt = obs_t.unsqueeze(0).to(self.device)
        mask_bt = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.strategy_net.get_probs(obs_bt, mask_bt)

        probs_np = probs.squeeze(0).cpu().numpy()

        if deterministic:
            return int(np.argmax(probs_np))
        else:
            return int(np.random.choice(NUM_ACTIONS, p=probs_np))

    def _save_checkpoint(self) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"deep_cfr_iter_{self.iteration:06d}.pt"
        torch.save({
            "iteration": self.iteration,
            "regret_net": self.regret_net.state_dict(),
            "strategy_net": self.strategy_net.state_dict(),
            "regret_opt": self.regret_opt.state_dict(),
            "strategy_opt": self.strategy_opt.state_dict(),
        }, path)
        logger.info("Deep CFR checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.iteration = ckpt["iteration"]
        self.regret_net.load_state_dict(ckpt["regret_net"])
        self.strategy_net.load_state_dict(ckpt["strategy_net"])
        self.regret_opt.load_state_dict(ckpt["regret_opt"])
        self.strategy_opt.load_state_dict(ckpt["strategy_opt"])
        logger.info("Loaded Deep CFR checkpoint: iter=%d", self.iteration)

    def compute_exploitability_bound(self) -> float:
        """
        Regret-based exploitability bound: exploitability ≤ Σ_i R^max_i / √T
        where R^max_i = max_{h,a} |cumulative_regret(h,a)|

        Only an upper bound — actual exploitability can be much lower.
        Use proper BR computation (LBR or ISMCTS) for tight estimates.
        """
        # Sample regret magnitudes from buffer
        max_r = 0.0
        for buf in self.regret_buffers:
            for sample in buf.buffer[:1000]:  # sample up to 1000
                max_r = max(max_r, float(np.abs(sample.regrets).max()))

        t = max(self.iteration, 1)
        return max_r / np.sqrt(t)
