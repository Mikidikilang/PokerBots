# VR-DeepPDCFR+ 6-Max NLHE — Architectural & Mathematical Audit Report

**Project:** VR-DeepPDCFR+ 6-Max No-Limit Texas Hold'em  
**Audit Date:** April 1, 2026  
**Files Reviewed:** `vr_deep_pdcfr_engine.py`, `buffers.py`, `networks.py`, `runner.py`, `train_6max_vr_deep.py`, `action_mapper.py`, `card_abstraction.py`, `dcfr_params.py`, `config_production.yaml`

---

## Executive Verdict: No — not as written.

This codebase has several mathematically fatal bugs that would cause it to converge to garbage regardless of GPU budget. Most are fixable in a week of engineering work. Bugs are ranked below by kill-radius.

---

## Critical Vulnerabilities

### Bug 1 — θ Bootstrap Target Formula is Wrong *(convergence killer)*

**File:** `vr_deep_pdcfr_engine.py`, `_compute_theta_loss`, line 781.

```python
# WHAT THE CODE DOES:
target = decay_weight * theta_frozen_pred + observed_tensor
# = w_t * θ_frozen + φ_obs

# WHAT IT SHOULD DO (per its own docstring):
target = decay_weight * theta_frozen_pred + (1 - decay_weight) * observed_tensor
# = w_t * θ_frozen + (1 - w_t) * φ_obs
```

As `t→∞`, `w_t→1`, so the actual target becomes `θ_frozen + φ` — cumulative and instantaneous advantages are *summed*, not interpolated. θ's targets grow without bound as training progresses. The `(1 - w_t)` coefficient on `observed_tensor` is simply missing.

**Fix:** Change line 781 to:
```python
target = decay_weight * theta_frozen_pred + (1 - decay_weight) * observed_tensor
```

---

### Bug 2 — Q-Loss Mixes States from Two Independent Samples *(corrupts the baseline entirely)*

**File:** `vr_deep_pdcfr_engine.py`, `_compute_q_loss`, lines 657–676.

The Q target is computed as:

```
V_target(s_A) = Σ_a  π(s_A, a) × A(s_B, a)
```

`s_A` comes from the strategy buffer and `s_B` comes from a completely independent random draw from the advantage buffer. These are two different game states. The Q network will learn to predict a nonsensical scalar and the variance-reduction baseline will add variance rather than reduce it.

**Fix:** Sample a single batch from the advantage buffer, which already stores `action_probs` (via `Transition.action_probs`) alongside `advantages`. Compute Q targets within a single cohesive batch:

```python
features, action_probs, _, _ = buffer_manager.advantage_buffer.sample_minibatch(batch_size, replace=True)
# advantages stored in the same transitions:
_, _, advantages = buffer_manager.advantage_buffer.sample_minibatch(batch_size, replace=True)
# ... but use a single sample pass so features and advantages align
```

Or restructure `sample_minibatch` on the advantage buffer to return all four fields together.

---

### Bug 3 — DCFR Module is Dead Code *(entire `dcfr_params.py` is never called)*

`compute_dcfr_discount` and `apply_dcfr_update` in `dcfr_params.py` are never imported or called anywhere in the engine or training loop. The config values `dcfr_alpha`, `dcfr_beta`, `dcfr_gamma` are present in the YAML but no code reads them. What is actually implemented is a custom temporal-decay bootstrapping scheme with no connection to the Brown & Sandholm (2019) update rule. The DCFR convergence proof does not apply.

**Fix (option A):** Wire DCFR discounting into advantage buffer weighting — weight each sample's loss contribution by its per-iteration discount factor `(t / (t + γ))^α`.

**Fix (option B):** Remove `dcfr_params.py` and stop naming the algorithm DCFR.

---

### Bug 4 — Batch Size Hardcoded at 32; Epochs Hardcoded at 1

**File:** `vr_deep_pdcfr_engine.py`, `train_networks`, lines 520–521.

```python
batch_size = min(32, ...)   # Config says 4096. This is 128× smaller.
num_epochs = 1              # Config says num_network_epochs: 4. Ignored.
```

Neither config value is read or passed into `train_networks()`. Training with batch size 32 gives catastrophically noisy gradient estimates for a [1024, 1024, 512, 256] network.

**Fix:** Add `batch_size` and `num_network_epochs` as parameters to `train_networks()` and pass them from the config in the training loop.

---

### Bug 5 — Chance Nodes Are Suppressed *(breaks the entire tree structure)*

**File:** `runner.py`, `GameStateAdapter.is_chance_node()`, line 137.

```python
def is_chance_node(self) -> bool:
    return False   # Always
```

In 6-Max NLHE, dealing the flop (3 community cards), turn, and river are explicit stochastic events in the game tree. External Sampling MCCFR computes *expected* values by averaging over all chance outcomes. By hardcoding `False`, every traversal operates on a single fixed board realization — the one dealt at `env.reset()`. The traversal computes "regrets given this specific board," not "regrets averaged over all boards." The resulting strategy overfits to a single board runout. CFR theory requires chance nodes to be handled correctly; skipping them breaks the convergence guarantee entirely.

**Fix:** Expose flop/turn/river dealing as chance nodes with uniform probability over all valid card combinations. At minimum, implement chance-sampling inside `traverse()` at the correct points in the betting tree rather than relying on a single `env.reset()` per traversal.

---

### Bug 6 — Only 1 Traversal Per Player Per Iteration (Config Says 100)

**File:** `train_6max_vr_deep.py`, `train()`, lines 252–269.

Config specifies `traversals_per_iteration: 100`. The training loop does exactly 1 traversal per updating player (6 total per iteration). The config value is never read. At 10M iterations × 6 traversals = 60M traversals total instead of the intended 600M. The advantage buffer will be severely underfilled each iteration, making network updates nearly meaningless.

**Fix:**
```python
traversals_per_iter = self.config["cfr"].get("traversals_per_iteration", 100)
for updating_player in range(self.num_players):
    for _ in range(traversals_per_iter):
        self.env.reset()
        root_state = GameStateAdapter(self.env, self.obs_builder)
        self.engine.traverse(root_state, initial_reach_probs, updating_player, depth=0)
```

---

### Bug 7 — Reach Probability of Updating Player Incorrectly Prunes the Tree

**File:** `vr_deep_pdcfr_engine.py`, lines 408–412.

```python
new_reach_probs[acting_player] *= predictive_strategy[action_idx]
if new_reach_probs[acting_player] > 1e-10:   # ← prunes based on OWN reach
    child_values = self.traverse(...)
```

In External Sampling MCCFR, the updating player's own reach probability should not gate traversal of counterfactual arms — the updating player evaluates all arms regardless of their own strategy probability. This pruning will skip low-probability actions, introducing bias into regret estimates for exactly the actions where strategic deviations matter most (rare, high-impact plays like large overbets).

**Fix:** Remove or replace with opponent-only reach probability pruning.

---

### Bug 8 — LBR Evaluator Uses 10 Hands, Not 50,000

**File:** `train_6max_vr_deep.py`, line 217.

```python
eval_config = NashEvalConfig(eval_hands=10)  # Config says oracle_hands: 50,000
```

Every exploitability measurement during training is computed over 10 hands. The monitoring dashboard will show pure noise throughout the 14-day run.

**Fix:**
```python
oracle_hands = self.config["evaluation"].get("oracle_hands", 50000)
eval_config = NashEvalConfig(eval_hands=oracle_hands)
```

---

## Theoretical Bottlenecks

### 1. Blueprint + Real-Time Search vs. Pure Neural Approach

Pluribus's key insight that this architecture abandons: **blueprint + real-time subgame solving**. The blueprint (offline MCCFR) computes a coarse approximation; real-time search then refines it for the specific hand being played. This means Pluribus's depth-limited errors are corrected at inference time. This system has no inference-time search — the Π network's output is the final action, and any approximation error baked in during training is permanent.

### 2. 12-Action Abstraction is Very Coarse for 6-Max

The config defines 12 total discrete actions across all streets. In 6-Max, preflop vs. postflop structure is fundamentally different: preflop involves 3-bets, 4-bets, 5-bets, and limp/fold dynamics with 5 opponents. With only 4 preflop bet-size buckets, the agent cannot represent many strategically critical sizings. Pluribus used a much richer abstraction and then performed real-time solving to refine it.

### 3. External Sampling Variance in Multi-Way Pots

External Sampling in 6-player games means 5 opponent nodes are sampled per traversal and only 1 player's regrets are updated per pass. The effective sample efficiency is much lower than heads-up, and estimated values at the updating player's nodes have higher variance due to the multiplicative sampling over 5 opponents. The VR baseline (Q network) was the correct solution in principle — but Bug 2 means it is currently non-functional.

### 4. Infoset Collision Risk

The architecture has no explicit infoset key computation. Two different game histories sharing the same feature vector are treated identically by all four networks. If `ObservationBuilder` doesn't encode betting round, pot size, and position perfectly, infosets will collide and regrets will be incorrectly aggregated. This creates an exploitability floor that training cannot breach regardless of iteration count.

---

## Code-Level Recommendations (Priority Order)

### Fix Immediately Before Any GPU Spend

| # | File | Line(s) | Fix |
|---|------|---------|-----|
| 1 | `vr_deep_pdcfr_engine.py` | 781 | Add `(1 - decay_weight)` coefficient on `observed_tensor` in `_compute_theta_loss` |
| 2 | `vr_deep_pdcfr_engine.py` | 657–676 | Sample advantages and strategy from the same transition in `_compute_q_loss` |
| 3 | `runner.py` | 137 | Implement proper chance node handling for flop/turn/river deals |
| 4 | `train_6max_vr_deep.py` | 252–269 | Read `traversals_per_iteration` from config and loop accordingly |
| 5 | `vr_deep_pdcfr_engine.py` | 520–521 | Pass `batch_size: 4096` and `num_network_epochs: 4` from config |
| 6 | `train_6max_vr_deep.py` | 217 | Read `oracle_hands: 50000` from config into `NashEvalConfig` |

### Fix Before Evaluating Convergence

| # | File | Line(s) | Fix |
|---|------|---------|-----|
| 7 | `vr_deep_pdcfr_engine.py` | 408–412 | Remove updating-player reach probability pruning |
| 8 | `dcfr_params.py` | entire file | Wire into advantage weighting or remove and rename the algorithm |
| 9 | `ObservationBuilder` | — | Audit that every distinct infoset maps to a unique feature vector |

### Before Production Cluster Launch

| # | File | Area | Fix |
|---|------|------|-----|
| 10 | `runner.py` | `get_infoset_features` | Verify `env._env.get_state(pid)` isolates only that player's hole cards and exposes no global deck state |
| 11 | `train_6max_vr_deep.py` | curriculum | Connect the three-phase UCB/FSP curriculum from config to the actual training loop (currently ignored) |

---

## Comparison to Pluribus

| Property | Pluribus | This system |
|----------|----------|-------------|
| Core algorithm | Tabular MCCFR + depth-limited subgame solving | External Sampling MCCFR + neural approximation |
| Inference-time search | Yes (real-time subgame solving) | No (Π network output is final) |
| Action abstraction | Rich (street-specific, ~14 sizes) | Coarse (12 actions total) |
| Infoset representation | Exact tabular keys | Feature vector (collision risk) |
| Regret storage | Memory-mapped tabular | Neural network (generalization + approximation error) |
| Blueprint training | ~12,400 CPU core-days | 14-day GPU cluster (proposed) |
| Depth-limited error correction | Corrected at inference | Permanent (baked into Π) |

The 4-network-per-player design is theoretically interesting, but the added complexity of θ/φ separation only pays off if both the bootstrapping (Bug 1) and the Q baseline (Bug 2) are correct. Currently the codebase carries the memory overhead of 24 networks while getting the convergence properties of none of them working correctly.

---

## Recommended Next Steps

1. Apply the 6 immediate fixes above.
2. Run a smoke test: 2-player heads-up, 10K iterations, measure exploitability every 500 iterations.
3. Confirm exploitability is monotonically decreasing before scaling to 6-Max.
4. If exploitability curves look healthy, scale to 6-Max with a small cluster (4–8 GPUs) for 100K iterations before committing the full 14-day reservation.
5. Consider adding real-time depth-limited subgame solving at inference time — this single addition has historically been the difference between "good bot" and "superhuman" in every published system.

---

*Audit conducted by architectural review. All line numbers refer to the uploaded source files as of March 31, 2026.*
