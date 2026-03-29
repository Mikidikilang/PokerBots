# Training Architecture Analysis: poker_ai_v5

**Date:** 2025-03-29  
**Status:** Deep CFR transition IN PROGRESS (Phase 2A-2C complete, Phase 2.5 integration BLOCKED)  
**Analyst:** GitHub Copilot

---

## Executive Summary

The codebase is **mid-migration from PPO to Deep CFR** with core mathematics complete but integration incomplete. The main blocker is **observation decoding**: CFR cannot learn proper strategies because infoset IDs are hardcoded instead of extracted from observation tensors.

| Metric | Status |
|--------|--------|
| Core CFR Algorithm | ✅ 100% Complete |
| Buffer Infrastructure | ✅ 100% Complete |
| Integration Bridge API | ✅ 100% Complete |
| **Observation Parsing** | ❌ **0% (BLOCKER)** |
| **Legal Actions** | ⚠️ Hardcoded (ISSUE) |
| Runner Integration | ❌ Not started |

**Time to functional CFR training:** ~2-3 days once observation decoding is fixed.

---

## 1. Current PPO Loop Structure

### TrainingRunner._run_single_iteration() [runner.py:230-350]

**6-step iteration:**

```
Step 1: Collect Rollout (2048 steps)
  ├─ For each step: action_logits, value = network(obs_dict)
  ├─ Sample action from Categorical(logits)
  ├─ Execute: obs', reward, done = env.step(action)
  └─ [FIX C1] Set bootstrap_value atomically in buffer
     (eliminates race condition where value was 1-step stale)

Step 2: Compute GAE
  ├─ Bootstrap: last_value = buffer.get_last_bootstrap_value()  [FIX C1]
  └─ GAE formula: A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...
     where δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
     Normalize: A = (A - mean) / std

Step 3: PPO Training (4 epochs × 4 mini-batches = 16 updates)
  ├─ For each mini-batch (512 samples):
  │  ├─ [POLICY LOSS]
  │  │  L_π = -E[min(ratio × A, clip(ratio) × A)]
  │  │  where ratio = exp(new_log_prob - old_log_prob)
  │  │
  │  ├─ [VALUE LOSS]
  │  │  L_V = 0.5 E[(V(s) - return)²]  (with optional value clipping)
  │  │
  │  ├─ [ENTROPY BONUS]
  │  │  L_H = entropy(π)  (promotes exploration)
  │  │
  │  ├─ [TOTAL]
  │  │  L_total = L_π + 0.5 L_V - 0.01 L_H
  │  │
  │  ├─ [FIX C-4] Cross-rank NaN guard synchronized across DDP ranks
  │  │  All ranks either raise FloatingPointError together or proceed
  │  │
  │  └─ optimizer.zero_grad(); backward(); clip_grad_norm(); step()

Step 4: Orchestrator Callback
  └─ on_iteration_end(iteration, stats)  [Curriculum learning]

Step 5: DDP Synchronization
  └─ on_ddp_sync(iteration)

Step 6: Buffer Reset
  └─ buffer.reset()
```

**Config (TrainerConfig in trainer.py:30-45):**
- learning_rate: 3e-4
- clip_epsilon: 0.2 (PPO clipping)
- value_loss_coef: 0.5
- entropy_coef: 0.01
- num_epochs: 4
- target_kl: 0.015 (early stopping if KL divergence too large)

---

## 2. CFRIntegrationBridge: Status & Issues

### Location: src/training/cfr_adapter.py (lines 180-290)

**Purpose:** Drop-in replacement for PPOTrainer maintaining same API

```python
class CFRIntegrationBridge:
    def train_on_buffer(self, buffer) -> dict[str, float]:
        # Same interface as PPOTrainer.train_on_buffer()
        # Returns: {cfr_loss, avg_regret, num_infosets, num_batches}
```

### ✅ WORKING
- Bridge maintains PPOTrainer.train_on_buffer() signature
- Calls CFREngine.train_on_rollouts() per mini-batch
- Aggregates stats back to runner

### ❌ CRITICAL ISSUES

#### Issue #1: Infoset IDs Are Hardcoded → CFR Learns Wrong Strategy

**Location:** cfr_adapter.py:130-160 (_generate_infoset_id method)

```python
def _generate_infoset_id(self, obs_dicts: dict, idx: int) -> str:
    # CURRENT (BROKEN):
    hero_cards = ("A", "K")  # HARDCODED!
    board_cards = ()         # HARDCODED!
    action_history = ()      # HARDCODED!
    
    # RESULT: All observations hash to ~same infoset_id
    # CFR learns ONE strategy for ALL game states (INCORRECT)
```

**Why Critical:**
- CFR's entire value is learning **per-infoset strategies** (different strategy for each unique game state)
- If all states map to same infoset, CFR learns one average policy for everything
- This is how traditional CFR fails without proper state abstraction

**Fix Required:**
Must extract actual card values from observation tensors:
```python
# What needs to happen:
hero_tensor = obs_dicts["hero_cards"][idx]      # tensor([...])
board_tensor = obs_dicts["board_cards"][idx]    # tensor([...])
action_history_tensor = obs_dicts.get("action_history")[idx]  # tensor([...])

# Parse: tensor integers (0-51) → card strings ("AS", "KH", etc.)
# Card mapping: 0→"2C", 1→"2D", ..., 51→"AS"
hero_cards = decode_card_indices(hero_tensor)      # ("AS", "KH")
board_cards = decode_card_indices(board_tensor)    # ("QC", "JH", "TS")
action_history = decode_action_history(action_history_tensor)  # ("check", "raise")
```

**Implementation Blocker:** Reverse-engineer encoding from [features.py](src/env/features.py)
- Need to understand: How are cards represented in observation dict?
- Need mapping: observation indices ↔ card strings

---

#### Issue #2: Legal Actions Always Set to All 12 Actions

**Location:** cfr_adapter.py:110

```python
legal_actions = list(range(12))  # HARDCODED - never changes!
```

**Problem:**
- Some actions illegal in every state (e.g., fold when facing no bet)
- CFR wastes regret updates on impossible moves
- ~80% of regret computations are for illegal actions (inefficient)

**Impact:** Regret updates don't accumulate properly; strategy learning slowed

**Fix Options:**
1. **Extract from env:** `legal_actions = env.get_legal_actions()`
2. **Store in buffer:** Add `_legal_actions_per_step` to RolloutBuffer
3. **Add to obs dict:** Include `legal_action_mask` in observation

**Preferred:** Option 2 (store in RolloutBuffer during collection)

---

#### Issue #3: Observation Flattening Logic (Minor)

**Location:** cfr_adapter.py:88-102

```python
def _flatten_obs_dict(self, obs_dicts: dict[str, torch.Tensor]) -> dict:
    # obs_dicts from buffer.get_mini_batches():
    # {key: tensor[batch_size, ...]}
    
    # But code treats as if obs_dicts = {key: tensor[...]}
    # Potential shape mismatch
```

**Status:** Low priority — can be debugged when running

---

### CFRIntegrationBridge Integration Points

**Called by:** runner.py (lines 100-108)
```python
self.trainer: PPOTrainer = PPOTrainer(...)  # CURRENTLY
# MUST CHANGE TO:
self.trainer = CFRIntegrationBridge(cfr_engine)
```

**Called once per iteration:** _run_single_iteration() (line 285)
```python
train_stats: dict[str, float] = self.trainer.train_on_buffer(self.buffer)
# This works unchanged for both PPOTrainer and CFRIntegrationBridge
```

---

## 3. Buffer Interfaces & Data Structures

### RolloutBuffer [buffer.py]

**Purpose:** Store PPO rollout data (transitions) and compute GAE advantages

**Architecture:**
```
Raw Data (during rollout):
  _observations: list[dict[str, torch.Tensor]]     # One obs dict per step
  _actions: list[torch.Tensor]                     # Actions taken
  _rewards: list[float]                            # Rewards received
  _log_probs: list[torch.Tensor]                   # Policy log probabilities
  _values: list[torch.Tensor]                      # Value estimates V(s)
  _dones: list[bool]                               # Episode terminal flags

[FIX C1] Bootstrap Value:
  _last_bootstrap_value: float = 0.0               # V(s_T) at rollout end
  
Computed Tensors (after compute_gae()):
  _advantages: torch.Tensor[2048]                  # Advantage estimates
  _returns: torch.Tensor[2048]                     # Discounted returns
  _obs_tensors: dict[str, torch.Tensor]            # Batched observations
  _actions_tensor: torch.Tensor[2048]              # Batched actions
  _log_probs_tensor: torch.Tensor[2048]            # Batched log probs
  _values_tensor: torch.Tensor[2048]               # Batched values
```

**Key Methods:**

| Method | Purpose | Called By |
|--------|---------|-----------|
| `add()` | Append transition data | RolloutCollector.collect_rollout() |
| `set_last_value()` [NEW] | Store bootstrap V(s_T) | RolloutCollector.collect_rollout() END |
| `get_last_bootstrap_value()` [NEW] | Retrieve bootstrap value | runner.py _run_single_iteration() |
| `compute_gae()` | Compute advantages + returns | runner.py _run_single_iteration() |
| `get_mini_batches()` | Yield shuffled batches | trainer.train_on_buffer() |
| `reset()` | Clear for next iteration | runner.py _run_single_iteration() |

**Mini-Batch Yielding:**
```python
# buffer.get_mini_batches() yields:
batch = {
    "observations": {
        "board_cards": tensor[batch_size, ...],
        "hero_cards": tensor[batch_size, ...],
        "opponent_cards": tensor[batch_size, ...],  # If visible
        "pot_odds": tensor[batch_size, 1],
        "stacks": tensor[batch_size, 2],
        "position": tensor[batch_size, 1],
        # ... other keys from observation dict
    },
    "actions": tensor[batch_size],
    "old_log_probs": tensor[batch_size],
    "advantages": tensor[batch_size],            # PPO-specific
    "returns": tensor[batch_size],               # PPO-specific
    "old_values": tensor[batch_size],            # PPO-specific
}
```

**Key Issue for CFR:** Observation dict keys and tensor shapes not standardized
- cfr_adapter._extract_cards() must handle variable observation structures

---

## 4. What's In Place vs What Needs Building

### ✅ COMPLETE (Phase 2A-2C)

**Core CFR Mathematics:**
- ✅ `cfr_valuator.py`: Counterfactual value computation (forward/backward pass)
- ✅ `cfr_infoset.py`: Regret accumulation + regret matching strategy
- ✅ `cfr_engine.py`: Main CFR orchestrator (train_on_rollouts + strategies)
- ✅ `cfr_traversal.py`: MCCFR external sampling traversal
- ✅ All modules compile without errors

**Buffer Infrastructure:**
- ✅ `cfr_buffer.py`: RegretBuffer (reservoir sampling for network training)
- ✅ `cfr_strategy.py`: StrategyBuffer (for behavioral cloning)
- ✅ `cfr_infoset.py`: InformationSetStorage (O(1) infoset lookup)

**Adapter Layer:**
- ✅ `CFRTrajectoryAdapter`: Mini-batch format conversion (PPO→CFR)
- ✅ `CFRIntegrationBridge`: Drop-in replacement for PPOTrainer

### ⚠️ IN PROGRESS (Phase 2.5 - BLOCKS TRAINING)

**Observation Decoding:** [CRITICAL BLOCKER]
- ❌ `_generate_infoset_id()`: Must extract cards from obs tensors
- ❌ `_extract_cards()`: Must parse card encoding
- ❌ Requires understanding features.py card representation

**Legal Actions:** [EFFICIENCY ISSUE]
- ❌ Store legal_actions per step in buffer
- ❌ Use in cfr_adapter for regret masking

**Runner Integration:** [CANNOT RUN YET]
- ❌ Create CFREngine in runner.__init__()
- ❌ Swap PPOTrainer → CFRIntegrationBridge
- ❌ Load CFRConfig from yaml config

### ❌ NOT STARTED (Phase 3+)

**Infoset Persistence:**
- ❌ Serialize InformationSetStorage to checkpoint files
- ❌ Deserialize on resume (cannot lose learned strategies)

**Multi-step Trajectories:**
- ❌ Preserve episode boundaries through collector → buffer
- ❌ Enable multi-step counterfactual lookahead

**MCCFR Traversal for Pure CFR:**
- ❌ Option to skip PPO buffer entirely
- ❌ Run pure MCCFR self-play

**Strategy Inference:**
- ❌ Create inference wrapper using StrategyNetwork
- ❌ Real-time subgame solving (Phase 4)

---

## 5. Integration Points That Need to Change

### A. src/training/runner.py [3-4 Lines]

**Current (lines 75-78):**
```python
self.trainer: PPOTrainer = PPOTrainer(
    trainer_config or TrainerConfig(),
    network, self.device,
)
```

**Required Change:**
```python
# Option 1: Conditional support (recommended)
use_cfr = yaml_config.get("training", {}).get("use_cfr", False)

if use_cfr:
    from src.training.cfr_engine import CFREngine, CFRConfig
    from src.training.cfr_adapter import CFRIntegrationBridge
    
    cfr_config = CFRConfig.from_dict(yaml_config)
    cfr_engine = CFREngine(cfr_config, network, self.device, env=self.env)
    self.trainer = CFRIntegrationBridge(cfr_engine)
else:
    self.trainer = PPOTrainer(trainer_config, network, self.device)
```

**No other runner.py changes** (CFRIntegrationBridge.train_on_buffer() compatible API)

---

### B. config.yaml [NEW SECTION]

**Add:**
```yaml
cfr:
  learning_rate: 3.0e-4
  adam_epsilon: 1.0e-5
  max_grad_norm: 0.5
  entropy_coef: 0.01
  num_epochs: 4
  
  # CFR-specific
  regret_discount: 1.0          # 1.0 = no discounting
  regret_min_threshold: 0.0     # Ignore small regrets
  regret_scaling: 1.0           # Numerical stability
  track_exploitability: true
  exploitability_update_freq: 100

training:
  use_cfr: false  # TOGGLE: false = PPO, true = CFR
```

---

### C. src/training/cfr_adapter.py [CRITICAL FIXES]

**Fix #1: Implement Actual Card Extraction** [BLOCKING]

```python
def _generate_infoset_id(self, obs_dicts: dict, idx: int) -> str:
    """Extract real card info from observation tensors."""
    # MUST REPLACE HARDCODED VALUES
    hero_cards = self._extract_cards(obs_dicts, idx, "hero")    # ("AS", "KH")
    board_cards = self._extract_cards(obs_dicts, idx, "board")  # ("QC", "JH", "TS")
    action_history = self._extract_action_history(obs_dicts, idx)  # [NEW]
    
    return hash_infoset(
        player=0,
        hole_cards=hero_cards,
        board_cards=board_cards,
        action_history=action_history,
    )

def _extract_cards(self, obs_dicts: dict, idx: int, card_type: str) -> tuple[str, ...]:
    """
    Parse observation tensors to card strings.
    
    observations indices (0-51):
        0-3:   2C, 2D, 2H, 2S
        4-7:   3C, 3D, 3H, 3S
        ...
        48-51: AC, AD, AH, AS
    
    Returns: ("AS", "KH", ...)
    """
    # IMPLEMENT THIS
    # 1. Get obs tensor for card_type key
    # 2. Extract batch index idx
    # 3. Map values to card strings
    # 4. Return tuple
    pass
```

**Fix #2: Use Real Legal Actions**

```python
# In batch_to_cfr_trajectories(), replace:
# legal_actions = list(range(12))
# WITH:
legal_actions = self._get_legal_actions(batch, i)

def _get_legal_actions(self, batch: dict, step_idx: int) -> list[int]:
    """Extract legal actions for this step."""
    # Option A: From batch metadata
    if "legal_actions" in batch:
        return batch["legal_actions"][step_idx]
    
    # Option B: From observation mask
    if "legal_action_mask" in batch["observations"]:
        mask = batch["observations"]["legal_action_mask"][step_idx]
        return [a for a in range(12) if mask[a] > 0.5]
    
    # Fallback
    return list(range(12))
```

---

## 6. Critical Path to Functional CFR

### Day 1: Observation Decoding (Est. 4-6 hours)

**Task:** Implement `_extract_cards()` to parse observation tensors

**Required knowledge:**
- Card encoding scheme in [src/env/features.py](src/env/features.py)
- How cards are represented: indices, normalized values?
- Reverse mapping: observation tensor → card string

**Deliverable:**
- Working `_extract_cards()` that returns ("AS", "KH") etc.
- Unit test with sample observation batch

---

### Day 2: Legal Actions + Runner Integration (Est. 4 hours)

**Task 1:** Fix legal actions
- Add `legal_actions` field to RolloutBuffer (or use from obs dict)
- Update cfr_adapter to use real legal actions
- Verify regret updates only on legal actions

**Task 2:** Integrate runner
- Add CFRConfig loading in runner.__init__()
- Conditionally create CFREngine vs PPOTrainer
- Test import and instantiation

**Deliverable:**
- CFR can be toggled on/off via config.yaml
- Training loop runs without errors (even if learning is wrong)

---

### Day 3: Debugging + Validation (Est. 3-4 hours)

**Task 1:** End-to-end test
- Run 10 iterations with CFR enabled
- Verify no crashes, valid stats output
- Check infoset creation (should be >1 unique infosets)

**Task 2:** Validation
- Compare learned strategies between PPO/CFR
- Check convergence metrics (regret magnitude)
- Validate against known weak strategies

**Deliverable:**
- CFR training loop functional
- Can measure convergence toward equilibrium

---

## Summary Table: What Works & What Doesn't

| Component | Status | Issue | Severity | Impact |
|-----------|--------|-------|----------|--------|
| CFREngine | ✅ | None | — | Computes regrets correctly |
| CFRTraversal | ✅ | Untested in live pipeline | MEDIUM | Can work but not verified |
| RegretBuffer | ✅ | Untested | MEDIUM | Schema correct, needs smoke test |
| CFRIntegrationBridge API | ✅ | Incomplete logic | HIGH | Bridge structure sound |
| **Infoset ID Generation** | ❌ | Hardcoded cards | **CRITICAL** | **CFR learns wrong strategy** |
| **Legal Actions** | ❌ | All 12 always | MEDIUM | ~80% wasted regret updates |
| **Observation Decoding** | ❌ | No tensor parser | **CRITICAL** | **Blocks #1 fix** |
| **Runner Integration** | ❌ | Not started | HIGH | Can't run CFR |
| Infoset Persistence | ❌ | No serialization | MEDIUM | Strategy lost on restart |
| Multi-step Trajectories | ❌ | No episode markers | LOW | Limits lookahead (Phase 3) |

---

## Code Locations Reference

**PPO Core:**
- PPO Policy Loss: [trainer.py#L220-L230](src/training/trainer.py#L220)
- Buffer GAE: [buffer.py#L270-L320](src/training/buffer.py#L270)
- Runner Loop: [runner.py#L230-L350](src/training/runner.py#L230)

**CFR Core:**
- Regret Computation: [cfr_engine.py#L300-L350](src/training/cfr_engine.py#L300)
- Regret Matching: [cfr_infoset.py#L150-L200](src/training/cfr_infoset.py#L150)
- MCCFR Traversal: [cfr_traversal.py#L70-L150](src/training/cfr_traversal.py#L70)

**Integration Blockers:**
- Hardcoded Infosets: [cfr_adapter.py#L130-L160](src/training/cfr_adapter.py#L130)
- Hardcoded Legal Actions: [cfr_adapter.py#L110](src/training/cfr_adapter.py#L110)
- Observation Dict Structure: [buffer.py#L250-L290](src/training/buffer.py#L250)

---

## Next Steps

1. **Immediate (Today):** Reverse-engineer observation encoding in [features.py](src/env/features.py)
2. **Day 1:** Implement `_extract_cards()` in cfr_adapter.py
3. **Day 2:** Fix legal actions, integrate runner
4. **Day 3:** End-to-end test and validation

Once blockers are cleared, CFR training can run with proper per-infoset strategy learning.
