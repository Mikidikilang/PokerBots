# Deep CFR Refactoring Plan

**Status**: Phase 2.5 Foundation → Phase 3 Implementation  
**Priority**: CRITICAL (blocks all CFR training)  
**Estimate**: 3-4 days, 4 major components  

---

## 🎯 Objective

Replace PPO training loop with Deep CFR architecture:
```
PPO Loop (current):   collect → GAE → train_ppo() → orchestrator → checkpoint
Deep CFR Loop (new):  collect → cfr_adapter → traversal → regret_update → strategy_update → checkpoint
```

Key insight: **Reuse PPO's collector/buffer infrastructure** through adapter middleware. No need to rewrite rollout collection.

---

## 💥 Critical Blockers (Must Fix First)

### Blocker #1: Card Encoding Decoding (cfr_adapter.py:160-185)
**File**: `src/training/cfr_adapter.py`  
**Issue**: `_extract_cards()` is stubbed — cannot decode card tensors→card strings  
**Impact**: Infoset IDs all identical → no strategy learning

**Card Encoding** (from `features.py:_encode_cards`):
```
52-dim multi-hot vector
Index = rank_idx * 4 + suit_idx   where rank_idx ∈ [0,12], suit_idx ∈ [0,3]

RANK_MAP = {"2":0, "3":1, ..., "T":8, "J":9, "Q":10, "K":11, "A":12}
SUIT_MAP = {"S":0, "H":1, "D":2, "C":3}
```

**Decoding Algorithm**:
```python
def decode_cards(card_tensor: Tensor[52]) -> tuple[str, ...]:
    """Reverse card encoding: 52-dim vector → tuple of card strings"""
    indices = torch.nonzero(card_tensor == 1.0).squeeze()
    cards = []
    for idx in indices:
        rank_idx = (idx // 4) % 13
        suit_idx = idx % 4
        rank = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"][rank_idx]
        suit = ["S","H","D","C"][suit_idx]
        cards.append(f"{suit}{rank}")
    return tuple(cards)
```

**Fix (30min)**:
1. Implement `_decode_cards()` helper
2. Parse obs_dict["hole_cards"] and obs_dict["community_cards"]
3. Handle edge cases (no cards → empty tuple)

---

### Blocker #2: Legal Actions Hardcoding (cfr_adapter.py:110)
**File**: `src/training/cfr_adapter.py`  
**Issue**: `legal_actions = list(range(12))` hardcoded → all actions marked legal always  
**Impact**: CFR wastes regrets on impossible moves (~20% efficiency loss)

**Current State**:
- RolloutBuffer collects observations but DISCARDS legal_actions info
- Collector produces observations but action_mask info lost after env.step()

**Fix (1 day)**:
1. Add `legal_actions: list[int]` field to RolloutBuffer
2. Collector stores `action_mask` from env at each timestep
3. cfr_adapter reads from buffer instead of hardcoding

**Before**:
```python
buffer.add(obs, action, reward, lp, val, done)
```

**After**:
```python
buffer.add(obs, action, reward, lp, val, done, legal_actions)
```

---

### Blocker #3: Game Tree State (cfr_adapter.py — doesn't exist)
**File**: `src/training/cfr_adapter.py`  
**Issue**: Cannot reconstruct game tree history from flat buffer  
**Impact**: No action_history → cannot assign correct infosets

**Missing**:
- `action_history` (which actions led to this state?)
- `player` (whose decision point is this?)
- `street` (preflop/flop/turn/river?)

**Partial Solution** (Phase 2.5):
Use flat representation: infoset = hash(hole_cards + board_cards + turn_number)
- Loses some strategic distinctions (but acceptable for Phase 2.5)
- CFR still learns correct strategies
- RTA bridge solution: full history recovered during traversal

**Full Solution** (Phase 3):
Track full game tree during collection:
```python
class CFRTrajectoryCollector:
    def add_step(self, obs, action, player, street, action_history):
        # Reconstructs full game tree path
```

---

## 🔧 Implementation Plan (Phases)

### Phase 2.5A: Fix cfr_adapter.py (1-2 days)

#### Step 1: Card Decoding (30 min)
```python
# src/training/cfr_adapter.py

def _decode_cards(card_tensor: Tensor[52]) -> tuple[str, ...]:
    """Decode 52-dim multi-hot card tensor to card strings."""
    indices = torch.nonzero(card_tensor.flatten() == 1.0).squeeze(-1)
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)
    
    RANK_NAMES = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
    SUIT_NAMES = ["S","H","D","C"]
    
    cards = []
    for idx in indices.tolist():
        rank_idx = idx // 4
        suit_idx = idx % 4
        cards.append(f"{SUIT_NAMES[suit_idx]}{RANK_NAMES[rank_idx]}")
    return tuple(cards)

def _extract_cards(self, obs_dicts, idx, card_type):
    """Extract cards from observation dict."""
    if card_type == "hero":
        tensor = obs_dicts["hole_cards"][idx]
    elif card_type == "board":
        tensor = obs_dicts["community_cards"][idx]
    else:
        raise ValueError(f"Unknown card_type: {card_type}")
    
    return self._decode_cards(tensor)
```

**Test**: 
```python
pytest tests/test_cfr_adapter.py::test_card_decoding -v
```

---

#### Step 2: Legal Actions Storage (4 hours)
**Files to modify**:
- `src/training/buffer.py` — add legal_actions field
- `src/training/collector.py` — capture legal_actions
- `src/training/cfr_adapter.py` — read from buffer

**Changes**:
1. **RolloutBuffer**:
```python
class RolloutBuffer:
    def add(self, obs, action, reward, log_prob, value, done, legal_actions=None):
        # ... existing code ...
        if legal_actions is not None:
            self.legal_actions.append(legal_actions)
```

2. **RolloutCollector**:
```python
def collect_rollout(self, ...):
    obs_dict = self.obs_builder.build(raw_state)
    action_mask = raw_state.get("legal_actions", list(range(9)))
    # ... existing stepping code ...
    self.buffer.add(..., legal_actions=action_mask)
```

3. **CFRTrajectoryAdapter**:
```python
legal_actions = batch.get("legal_actions", None)
if legal_actions is not None:
    trajectory["legal_actions_per_node"] = [legal_actions[i].tolist()]
```

---

#### Step 3: Simplified Infoset IDs (Phase 2.5)
```python
def _generate_infoset_id(self, obs_dicts, idx):
    """Simplified: hash(player, hero_cards, board_cards, turn_number)"""
    hero_cards = self._extract_cards(obs_dicts, idx, "hero")
    board_cards = self._extract_cards(obs_dicts, idx, "board")
    
    # Get turn number from betting history length (proxy for decision depth)
    betting_history = obs_dicts.get("betting_history", None)
    if betting_history is not None:
        action_count = betting_history[idx].nonzero().shape[0] // 11  # 11 dims per action
    else:
        action_count = 0
    
    from src.training.cfr_infoset import hash_infoset
    return hash_infoset(
        player=0,  # Always hero in our setup
        hole_cards=hero_cards,
        board_cards=board_cards,
        action_history=tuple(range(action_count)),  # Placeholder
    )
```

---

### Phase 2.5B: Refactor runner.py (1-2 days)

**Current**: `runner.py` → `TrainingRunner._run_single_iteration()` → `PPOTrainer.train_on_buffer()`

**New**: `runner.py` → `CFRRunner._run_single_iteration()` → `CFREngine.run_deep_cfr_training_loop()`

#### Changes:
1. **Create CFRConfig** (config.yaml):
```yaml
cfr:
  learning_rate: 1e-3
  regret_discount: 1.0  # Pure DCFR
  traversals_per_iteration: 500
  regret_network_updates: 4
  strategy_network_updates: 4
  batch_size: 32
```

2. **Refactor runner._run_single_iteration()** (60 lines → 40 lines):
```python
def _run_single_iteration(self):
    # 1. Collect rollout (same as PPO)
    rollout = self.collector.collect_rollout(...)
    
    # 2. Adapt to CFR trajectories
    cfr_trajs = self.cfr_adapter.batch_to_cfr_trajectories(rollout)
    
    # 3. Run Deep CFR training
    self.cfr_engine.train_on_trajectories(cfr_trajs)
    
    # 4. Same checkpoint/callback logic
    if self.iteration % self.checkpoint_interval == 0:
        self.checkpoint_manager.save(...)
```

3. **Deprecate PPOTrainer** → make optional via config flag:
```python
if config.get("training_algorithm") == "ppo":
    trainer = PPOTrainer(...)
elif config.get("training_algorithm") == "cfr":
    trainer = CFREngine(...)
```

---

### Phase 2.5C: Implement Three-Component CFR Loop (1 day)

**Current CFREngine**: Stub with regret aggregation  
**New CFREngine**: Full pipeline

#### Components:

**1. Traversal Workers** (CPU):
```python
class CFRTraversalWorker:
    def run_traversal(self, sampled_hands: dict) -> RegretSample:
        """
        Args:
            sampled_hands: {"hero_hand": card_tuple, "villain_hand": card_tuple}
        
        Returns:
            {
                "infoset_id": str,
                "regrets": {0: 0.5, 1: -0.2, 2: 0.1},  # per-action regrets
                "observation": tensor,  # for strategy network
            }
        """
        # Initialize game tree from saved point
        # Sample opponent strategy
        # Recurse: compute counterfactual value
        # Backprop regrets to hero
        return sample
```

**2. Regret Network Training** (GPU):
```python
def train_regret_network(self, regret_batch: list[RegretSample]):
    """Predict counterfactual regrets via MSE loss."""
    for sample in regret_batch:
        obs = sample["observation"]
        true_regrets = sample["regrets"]
        
        pred_regrets = self.regret_network(obs)
        
        loss = 0.5 * (pred_regrets - true_regrets) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

**3. Strategy Update** (GPU):
```python
def update_strategy_network(self, strategy_batch: list[StrategyData]):
    """Train strategy network via behavioral cloning (cross-entropy)."""
    for data in strategy_batch:
        obs = data["observation"]
        avg_strategy = data["avg_strategy"]  # from infoset storage
        
        logits = self.strategy_network(obs)
        loss = F.cross_entropy(logits, avg_strategy)
        
        self.strategy_optimizer.zero_grad()
        loss.backward()
        self.strategy_optimizer.step()
```

---

## 📋 File-by-File Changes

| File | Changes | Complexity | Lines |
|------|---------|-----------|-------|
| `src/training/cfr_adapter.py` | Implement card decoding, legal actions, infoset generation | HIGH | +80 |
| `src/training/buffer.py` | Add legal_actions field | LOW | +10 |
| `src/training/collector.py` | Capture legal_actions | LOW | +5 |
| `src/training/cfr_engine.py` | Implement three-component loop, complete stubs | HIGH | +200 |
| `src/training/runner.py` | Switch PPOTrainer→CFREngine | MEDIUM | ±20 |
| `config.yaml` | Add cfr section, training_algorithm flag | LOW | +20 |
| `src/training/cfr_traversal.py` | [Already exists] Verify traversal implementation | LOW | 0 |

---

## ✅ Verification & Testing

### Unit Tests:
```bash
# Card decoding
pytest tests/test_cfr_adapter.py::test_card_decoding -v

# Legal actions
pytest tests/test_cfr_adapter.py::test_legal_actions_extraction -v

# Infoset generation
pytest tests/test_cfr_adapter.py::test_infoset_id_generation -v

# CFR loop
pytest tests/test_training/test_cfr_engine.py -v
```

### Integration Test:
```bash
# Run 10 CFR iterations, check:
# - Multiple infosets created
# - Regrets accumulate
# - Strategy updates
python -c "from src.training.runner import TrainingRunner; r = TrainingRunner(...); r.run(max_iterations=10)"
```

### Convergence Validation:
```bash
# Run 1000 iterations on Kuhn poker
# Check: exploitability < 5% of GTO bound
python tests/test_training/test_cfr_convergence.py --game kuhn --iterations 1000
```

---

## 📊 Timeline

| Phase | Task | Effort | Days | Critical |
|-------|------|--------|------|----------|
| 2.5A.1 | Card decoding impl + tests | 4h | 0.5 | ✅ YES |
| 2.5A.2 | Legal actions storage | 8h | 1.0 | ✅ YES |
| 2.5A.3 | Infoset ID generation | 4h | 0.5 | ✅ YES |
| 2.5B | Runner refactoring | 8h | 1.0 | ⚠️ MEDIUM |
| 2.5C | Three-component loop | 12h | 1.5 | ✅ YES |
| Review | Testing + debugging | 8h | 1.0 | ✅ YES |
| **Total** | | **44h** | **5.5** | |

*Realistic: 3-4 days with 8h/day coding*

---

## 🎯 Success Criteria

After refactoring, must verify:

1. ✓ Multiple infosets created (not all hash to same ID)
2. ✓ Regrets accumulate over iterations
3. ✓ Strategy network converges (loss decreases)
4. ✓ Kuhn poker reaches <10% exploitability in 500 iters
5. ✓ All 159 tests still pass
6. ✓ No numerical instabilities (NaN/Inf)

---

## 🚨 Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking PPO paths | Keep PPO optional via config flag; add feature flag for CFR |
| Card encoding bugs | Write exhaustive tests for all 52 cards + edge cases |
| Regret explosion | Monitor max regret per action; implement clipping if needed |
| Slow convergence | Log infoset counts, avg regret magnitude, strategy entropy |

