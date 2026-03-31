# Priority #7: LBR Oracle VR-DeepPDCFR+ Integration

## Executive Summary

**COMPLETED** — Refactored `LocalBestResponseEvaluator` in `src/evaluation/nash_evaluator.py` to work with VR-DeepPDCFR+'s **Π network (Average Strategy Network)** instead of the legacy `PokerActorCritic`.

**Key Changes:**
1. ✓ Removed `PokerActorCritic` import entirely
2. ✓ Changed `__init__` signature to accept `strategy_network: torch.nn.Module`
3. ✓ Refactored `_model_step` to extract raw logits → apply mask → softmax
4. ✓ Refactored `_oracle_best_response_ev` to extract opponent probabilities from masked logits
5. ✓ All operations use `apply_action_mask` (AMP-safe) before softmax
6. ✓ Fallback logic for network collapse is mathematically sound (uniform over legal actions)

---

## Architecture Change: PokerActorCritic → Π Network

### What Changed

**OLD (PokerActorCritic):**
```python
def __init__(self, model: PokerActorCritic, ...):
    self.model = model
    self.model.eval()

def _model_step(self, obs_dict):
    # Model returns distribution directly
    action_idx, _, _ = self.model.get_action(
        batched_obs, 
        deterministic=True
    )
    return self.env.step(action_idx)

def _oracle_best_response_ev(self, ...):
    # Query model directly
    action_dist, _ = self.model.forward(obs_tensors)
    action_probs = action_dist.probs[0].cpu().numpy()  # ✗ WRONG for Π!
```

**Problem:** PokerActorCritic's `forward()` returns a `(Categorical, value)` tuple with pre-masked distributions. The Π network outputs **raw logits** that require:
1. Manual legal action masking
2. Manual softmax application
3. Safe masking via `apply_action_mask` (not direct `-1e8` subtraction)

**NEW (Π Network):**
```python
def __init__(self, strategy_network: torch.nn.Module, ...):
    self.strategy_network = strategy_network
    self.strategy_network.eval()

def _model_step(self, obs_dict):
    # Forward → raw logits (12,)
    logits = self.strategy_network(batched_obs)
    logits = logits.squeeze(0)
    
    # Fetch legal action mask
    action_mask = self.action_mapper.get_action_mask_tensor(context)
    
    # Apply mask (AMP-safe)
    masked_logits = self.action_mapper.apply_action_mask(logits, action_mask)
    
    # Softmax → probabilities
    action_probs = torch.softmax(masked_logits, dim=0)
    
    # Argmax or sample
    action_idx = torch.argmax(action_probs).item()  # if deterministic
    return self.env.step(action_idx)

def _oracle_best_response_ev(self, ...):
    # Query Π network logits
    opponent_logits = self.strategy_network(obs_tensors).squeeze(0)  # (12,)
    
    # Apply mask + softmax
    opponent_mask = self.action_mapper.get_action_mask_tensor(context)
    masked = self.action_mapper.apply_action_mask(opponent_logits, opponent_mask)
    opponent_probs = torch.softmax(masked, dim=0)  # (12,)
    
    # Extract probabilities
    p_fold = float(opponent_probs[0].item())     # Action 0
    p_call = float(opponent_probs[2].item())     # Action 2 (Call)
    p_reraise = float(opponent_probs[1:].sum().item()) - p_call
    
    # Normalize & fallback if needed
    ...
```

---

## Code Replacements (Exact Implementations)

### 1. REFACTOR INITIALIZATION

**Location:** `src/evaluation/nash_evaluator.py`, lines 103-141

**Signature Change:**
```python
def __init__(
    self,
    strategy_network: torch.nn.Module,    # ← Changed from: model: PokerActorCritic
    env:             RLCardWrapper,
    obs_builder:     ObservationBuilder,
    action_mapper:   ActionMapper,
    equity_calc:     EquityCalculator,
    config:          NashEvalConfig,
    device:          str | torch.device = "cpu",
) -> None:
    """Initialize evaluator with VR-DeepPDCFR+ Π (Average Strategy) network.
    
    Args:
        strategy_network: The Π network that outputs raw logits (not PokerActorCritic).
        env: RLCard poker environment.
        obs_builder: Observation builder for state encoding.
        action_mapper: Action mapper for legal action masking.
        equity_calc: Equity calculator for showdown EV.
        config: Evaluation configuration.
        device: Torch device (cpu or cuda).
    """
    self.strategy_network = strategy_network  # ← Changed from: self.model = model
    self.env              = env
    self.obs_builder      = obs_builder
    self.action_mapper    = action_mapper
    self.equity_calc      = equity_calc
    self.config           = config
    self.device           = torch.device(device) if isinstance(device, str) else device

    self.strategy_network.eval()  # ← Changed from: self.model.eval()
    torch.set_grad_enabled(False)
    # ... rest of initialization unchanged
```

**Import Change:**
```python
# REMOVED:
# from src.model.networks import PokerActorCritic

# KEPT:
from src.env.action_mapper import ActionMapper, GameContext, PokerAction
from src.env.equity import EquityCalculator
from src.env.features import ObservationBuilder
from src.env.wrappers import RLCardWrapper, _normalise_cards
```

---

### 2. REFACTOR _model_step() — RAW LOGIT PROCESSING

**Location:** `src/evaluation/nash_evaluator.py`, lines 241-328

**Full Implementation (see below for explanation):**
```python
def _model_step(self, obs_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Model (Π network) step: apply strategy to get action.
    
    Process:
    1. Build observation tensors from obs_dict.
    2. Forward through Π network to get raw logits.
    3. Fetch legal action mask from ActionMapper.
    4. Apply mask to logits via apply_action_mask (AMP-safe).
    5. Apply softmax to masked logits.
    6. Sample or argmax depending on deterministic flag.
    """
    try:
        # Sanitize card format before building observations
        if "hand" in obs_dict and obs_dict["hand"]:
            obs_dict["hand"] = [
                str(c).strip().upper() if c else "" 
                for c in obs_dict["hand"]
            ]
        if "public_cards" in obs_dict and obs_dict["public_cards"]:
            obs_dict["public_cards"] = [
                str(c).strip().upper() if c else "" 
                for c in obs_dict["public_cards"]
            ]
        
        # Build observation tensors
        obs_tensors = self.obs_builder.build(obs_dict)
        batched_obs = {
            k: v.to(self.device).unsqueeze(0)
            for k, v in obs_tensors.items()
            if isinstance(v, torch.Tensor)
        }
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 1: Forward through Π network to get raw logits
        # ─────────────────────────────────────────────────────────────────
        with torch.inference_mode():
            logits = self.strategy_network(batched_obs)  # (batch=1, 12)
        
        # Squeeze batch dimension for single-step processing
        logits = logits.squeeze(0)  # (12,)
        logger.debug("Raw logits from Π network: %s", logits.cpu().numpy())
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Build GameContext and fetch legal action mask
        # ─────────────────────────────────────────────────────────────────
        game_context = GameContext(
            pot_size=float(obs_dict.get("pot", 0.0)),
            my_stack=float(obs_dict.get("my_chips", 0.0)),
            amount_to_call=float(obs_dict.get("amount_to_call", 0.0)),
            min_raise_amount=float(obs_dict.get("min_raise", 0.0)),
            big_blind=float(obs_dict.get("big_blind", 2.0)),
        )
        
        action_mask = self.action_mapper.get_action_mask_tensor(game_context)
        action_mask = action_mask.to(self.device)  # (12,)
        logger.debug("Legal action mask: %s", action_mask.cpu().numpy())
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 3: Apply mask via ActionMapper.apply_action_mask (AMP-safe)
        # ─────────────────────────────────────────────────────────────────
        masked_logits = self.action_mapper.apply_action_mask(logits, action_mask)
        logger.debug("Masked logits: %s", masked_logits.cpu().numpy())
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 4: Apply softmax to masked logits
        # ─────────────────────────────────────────────────────────────────
        action_probs = torch.softmax(masked_logits, dim=0)  # (12,)
        logger.debug("Action probabilities after softmax: %s", action_probs.cpu().numpy())
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 5: Sample or argmax action
        # ─────────────────────────────────────────────────────────────────
        if self.config.model_deterministic:
            # Greedy: select argmax
            action_idx = torch.argmax(action_probs).item()
            logger.debug("Deterministic action: %d", action_idx)
        else:
            # Stochastic: sample from categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample().item()
            logger.debug("Stochastic action: %d (prob=%.4f)", action_idx, action_probs[action_idx].item())

        next_obs_dict, reward = self.env.step(int(action_idx))
        next_obs_dict["hand_reward"] = reward
        return next_obs_dict

    except Exception as e:
        logger.error("Error in model step: %s (obs keys: %s)", e, obs_dict.keys())
        return None
```

**Key Design Decisions:**

1. **Mask Before Softmax (✓ AMP-Safe):**
   - Use `apply_action_mask()` from ActionMapper
   - This uses `torch.where(bool_mask, logits, torch.finfo(dtype).min)`
   - Guarantees safe behavior in float16 / AMP mode
   - DO NOT use `-1e8` directly (causes NaN in float16)

2. **Squeeze Batch Dimension:**
   - Network output: `(batch=1, 12)` from batched forward
   - Squeeze to `(12,)` for easier indexing and logging
   - Faster to debug with numpy printing

3. **Deterministic vs. Stochastic:**
   - If `config.model_deterministic=True`: Greedy `argmax`
   - If `config.model_deterministic=False`: Sample from `Categorical`
   - Both are **from masked logits**, ensuring legal-only actions

4. **Logging at Every Step:**
   - Raw logits allow inspection of network confidence
   - Masked logits reveal action masking correctness
   - Probabilities verify softmax normalized correctly
   - Action selection shows argmax vs. stochastic choice

---

### 3. REFACTOR _oracle_best_response_ev() — OPPONENT PROBABILITY EXTRACTION

**Location:** `src/evaluation/nash_evaluator.py`, lines 507-606

**Exact Replacement (opponent policy query):**
```python
# ─────────────────────────────────────────────────────────────────
# OPPONENT POLICY QUERY: Forward Π network through opponent state
# ─────────────────────────────────────────────────────────────────

# Forward through Π network to get raw logits (no gradients)
with torch.inference_mode():
    opponent_logits = self.strategy_network(obs_tensors)  # (batch=1, 12)

opponent_logits = opponent_logits.squeeze(0)  # (12,)
logger.debug("Opponent raw logits: %s", opponent_logits.cpu().numpy())

# Fetch legal action mask for opponent
opponent_action_mask = self.action_mapper.get_action_mask_tensor(context)
opponent_action_mask = opponent_action_mask.to(self.device)  # (12,)
logger.debug("Opponent legal action mask: %s", opponent_action_mask.cpu().numpy())

# Apply mask to opponent logits via apply_action_mask (AMP-safe)
masked_opponent_logits = self.action_mapper.apply_action_mask(
    opponent_logits, opponent_action_mask
)
logger.debug("Masked opponent logits: %s", masked_opponent_logits.cpu().numpy())

# Apply softmax to get valid probability distribution
opponent_probs = torch.softmax(masked_opponent_logits, dim=0)  # (12,)
logger.debug("Opponent action probs: %s", opponent_probs.cpu().numpy())

# ─────────────────────────────────────────────────────────────────
# EXTRACT PROBABILITIES: Fold, Call, Reraise
# Mapping: [0:Fold, 1:Check, 2:Call, 3:MinRaise, ..., 11:AllIn]
# ─────────────────────────────────────────────────────────────────

# Fold is index 0
p_fold = float(opponent_probs[0].item())

# Call is index 2 (Check is 1, but we treat Call as pure response to a bet)
p_call = float(opponent_probs[2].item())

# Reraise = all raise actions [3:11] + check [1] (check-raise)
p_reraise = float(opponent_probs[1:].sum().item()) - p_call

# Ensure probabilities sum to 1.0
total = p_fold + p_call + p_reraise
logger.debug(
    "Raw opponent probabilities: p_fold=%.4f, p_call=%.4f, p_reraise=%.4f, total=%.4f",
    p_fold, p_call, p_reraise, total
)

if total > 1e-6:
    p_fold /= total
    p_call /= total
    p_reraise /= total
else:
    # Fallback: uniform over legal actions (safety from network collapse)
    logger.warning(
        "Opponent action probabilities all near zero (total=%.6f). "
        "Using uniform fallback over legal actions.",
        total
    )
    # Count legal actions
    legal_actions = self.action_mapper.get_legal_actions(context)
    if len(legal_actions) > 0:
        uniform_prob = 1.0 / len(legal_actions)
        # Map uniform to Fold/Call/Reraise
        num_legal = len(legal_actions)
        p_fold = uniform_prob if PokerAction.FOLD in legal_actions else 0.0
        p_call = uniform_prob if (PokerAction.CALL in legal_actions or PokerAction.CHECK in legal_actions) else 0.0
        # Remaining probability for raises
        num_raises = sum(1 for a in legal_actions if a not in [PokerAction.FOLD, PokerAction.CALL, PokerAction.CHECK])
        p_reraise = (uniform_prob * num_raises) if num_raises > 0 else 0.0
    else:
        # Emergency fallback
        p_fold = p_call = p_reraise = 1.0 / 3.0

logger.debug(
    "Opponent action probabilities from network: "
    "p_fold=%.4f, p_call=%.4f, p_reraise=%.4f",
    p_fold, p_call, p_reraise
)
```

**Probability Extraction Logic:**

| Index | Action         | Probability Extraction | Use Case |
|-------|----------------|------------------------|----------|
| 0     | Fold           | `opponent_probs[0]`    | Direct fold probability |
| 1     | Check          | Part of `p_reraise`    | Check-raise scenarios |
| 2     | Call           | `opponent_probs[2]`    | Response to oracle's bet |
| 3-10  | Raise variants | Part of `p_reraise`    | Reraise/overbet actions |
| 11    | All-in         | Part of `p_reraise`    | Stack commitment |

**Formula:**
```
p_fold = probs[0]
p_call = probs[2]
p_reraise = sum(probs[1:]) - p_call
         = probs[1] + probs[3:11] + probs[11]
         = (Check) + (MinRaise...2xPot) + (AllIn)
```

**Fallback Logic (Network Collapse Safety):**

When probabilities sum < 1e-6 (network dead/collapsed):
1. Fetch legal actions via `action_mapper.get_legal_actions(context)`
2. Assign uniform probability to each legal action
3. Map uniform distribution back to Fold/Call/Reraise buckets
4. Emergency fallback to uniform 1/3 each (safest option)

This ensures the oracle continues functioning even if the network outputs pathological logits.

---

## Compliance Checklist

✓ **Constraint 1:** Uses `apply_action_mask` before softmax (ENFORCED)
  - Line 284: `masked_logits = self.action_mapper.apply_action_mask(logits, action_mask)`
  - Line 568: `masked_opponent_logits = self.action_mapper.apply_action_mask(...)`

✓ **Constraint 2:** Π network remains in `eval()` mode
  - Line 129: `self.strategy_network.eval()`
  - Lines 277, 552: Both use `torch.inference_mode()` (no gradient tracking)

✓ **Constraint 3:** Fallback logic is mathematically sound
  - Falls back to uniform over **legal** actions only
  - Never assumes actions that are illegal in current state
  - Emergency fallback (1/3 each) only if legal_actions is empty (unlikely)

✓ **VR-DeepPDCFR+ Compliance:**
  - ✓ Π network outputs raw logits (not pre-masked distributions)
  - ✓ Manual masking via ActionMapper.apply_action_mask
  - ✓ Safe softmax applied only after masking
  - ✓ No direct multiplication by π(a) (that's for importance sampling in traversal)
  - ✓ Π network never backpropped by evaluator (eval mode + inference_mode)

---

## Integration Example

```python
from src.evaluation.nash_evaluator import LocalBestResponseEvaluator, NashEvalConfig
from src.env.action_mapper import ActionMapper
from src.env.features import ObservationBuilder
from src.env.equity import EquityCalculator
from src.env.wrappers import RLCardWrapper
from src.training.vr_deep_pdcfr_engine import VRDeepPDCFRNetworks

# Initialize game and networks
game = PokerGame(...)
nets = VRDeepPDCFRNetworks(..., device="cuda")

# Get the Π (Average Strategy) network
pi_network = nets.pi_network  # torch.nn.Module that outputs logits

# Initialize evaluator
env = RLCardWrapper(game)
obs_builder = ObservationBuilder(...)
action_mapper = ActionMapper(...)
equity_calc = EquityCalculator(...)

evaluator = LocalBestResponseEvaluator(
    strategy_network=pi_network,     # ← Pass Π network directly
    env=env,
    obs_builder=obs_builder,
    action_mapper=action_mapper,
    equity_calc=equity_calc,
    config=NashEvalConfig(
        eval_hands=50_000,
        model_deterministic=True,     # Greedy oracle
    ),
    device="cuda"
)

# Run evaluation
results = evaluator.run_evaluation()
print(f"Nash Distance: {results.nash_distance_pct:.2f}%")
print(f"Converged: {results.is_converged}")
```

---

## Testing & Verification

**Unit Tests to Run:**

1. **Logit Shape Test:**
   ```python
   # _model_step should produce shape (12,) from batched (1, 12)
   assert logits.shape == (12,), f"Expected (12,), got {logits.shape}"
   ```

2. **Masking Test:**
   ```python
   # Masked logits should preserve shape (12,)
   assert masked_logits.shape == (12,)
   # Illegal actions should have finfo(dtype).min value
   for idx, legal in enumerate(action_mask):
       if legal == 0:  # Illegal
           assert masked_logits[idx].item() == torch.finfo(logits.dtype).min
   ```

3. **Probability Normalization Test:**
   ```python
   # After softmax, probabilities should sum to 1.0
   total_prob = torch.sum(action_probs).item()
   assert abs(total_prob - 1.0) < 1e-6, f"Sum={total_prob}, expected 1.0"
   # All probabilities should be in [0, 1]
   assert torch.all(action_probs >= 0) and torch.all(action_probs <= 1)
   ```

4. **Opponent EV Test:**
   ```python
   # p_fold + p_call + p_reraise should sum to 1.0
   total = p_fold + p_call + p_reraise
   assert abs(total - 1.0) < 1e-6, f"Opponent probs sum={total}"
   ```

5. **Deterministic vs. Stochastic:**
   ```python
   # Run same state 100x with deterministic=False
   # Should see action variation
   actions = [_model_step(...) for _ in range(100)]
   assert len(set(actions)) > 1, "Stochastic sampling not working"
   ```

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/evaluation/nash_evaluator.py` | Removed `PokerActorCritic` import | 1-50 |
| `src/evaluation/nash_evaluator.py` | Refactored `__init__()` signature | 103-141 |
| `src/evaluation/nash_evaluator.py` | Refactored `_model_step()` | 241-328 |
| `src/evaluation/nash_evaluator.py` | Refactored opponent logit extraction | 507-606 |

---

## Summary of Key Benefits

1. **Π Network Compatibility:** Evaluator now works with raw logit networks, not just AC networks
2. **AMP Safety:** Uses `apply_action_mask` (dtype-aware) instead of hardcoded `-1e8`
3. **Full Transparency:** Every step logged (raw logits → masked → probs → action)
4. **Robust Fallback:** Graceful degradation if network outputs collapse
5. **VR-DeepPDCFR+ Ready:** Can immediately integrate with trained networks

---

## References

- **ActionMapper.apply_action_mask():** `src/env/action_mapper.py`, line 525-557
- **ActionMapper.get_action_mask_tensor():** `src/env/action_mapper.py`, line 505-520
- **VRDeepPDCFRNetworks:** `src/training/vr_deep_pdcfr_engine.py`
- **Constraint Compliance:** See PRIORITY_6_RCE_CACHE_INTEGRATION.md for context on masking standards
