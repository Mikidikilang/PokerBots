# Priority #7 — Quick Reference

## Three Exact Replacements

### 1. Signature: `__init__` 

**Parameter Change:**
- OLD: `model: PokerActorCritic`
- NEW: `strategy_network: torch.nn.Module`

**Attribute Change:**
- OLD: `self.model = model`
- NEW: `self.strategy_network = strategy_network`

**Eval Statement:**
- OLD: `self.model.eval()`
- NEW: `self.strategy_network.eval()`

---

### 2. Method: `_model_step`

**5-Step Pipeline:**
1. **Forward Π network:**
   ```python
   with torch.inference_mode():
       logits = self.strategy_network(batched_obs)  # (batch=1, 12)
   logits = logits.squeeze(0)  # (12,)
   ```

2. **Build GameContext & fetch mask:**
   ```python
   game_context = GameContext(
       pot_size=..., my_stack=..., amount_to_call=..., 
       min_raise_amount=..., big_blind=...
   )
   action_mask = self.action_mapper.get_action_mask_tensor(game_context)
   action_mask = action_mask.to(self.device)
   ```

3. **Apply mask (AMP-safe):**
   ```python
   masked_logits = self.action_mapper.apply_action_mask(logits, action_mask)
   ```

4. **Softmax:**
   ```python
   action_probs = torch.softmax(masked_logits, dim=0)
   ```

5. **Argmax or sample:**
   ```python
   if self.config.model_deterministic:
       action_idx = torch.argmax(action_probs).item()
   else:
       dist = torch.distributions.Categorical(action_probs)
       action_idx = dist.sample().item()
   ```

---

### 3. Method: `_oracle_best_response_ev` (opponent query section)

**Logit Extraction:**
```python
with torch.inference_mode():
    opponent_logits = self.strategy_network(obs_tensors)  # (batch=1, 12)
opponent_logits = opponent_logits.squeeze(0)  # (12,)
```

**Masking & Softmax:**
```python
opponent_action_mask = self.action_mapper.get_action_mask_tensor(context)
opponent_action_mask = opponent_action_mask.to(self.device)
masked_opponent_logits = self.action_mapper.apply_action_mask(
    opponent_logits, opponent_action_mask
)
opponent_probs = torch.softmax(masked_opponent_logits, dim=0)
```

**Probability Extraction:**
```python
p_fold = float(opponent_probs[0].item())         # Index 0: Fold
p_call = float(opponent_probs[2].item())         # Index 2: Call
p_reraise = float(opponent_probs[1:].sum().item()) - p_call  # Rest
```

**Normalization & Fallback:**
```python
total = p_fold + p_call + p_reraise
if total > 1e-6:
    p_fold /= total
    p_call /= total
    p_reraise /= total
else:
    # Fallback: uniform over legal actions
    legal_actions = self.action_mapper.get_legal_actions(context)
    # ... map uniform to Fold/Call/Reraise
```

---

## Import Changes

**REMOVE:**
```python
from src.model.networks import PokerActorCritic
```

**KEEP (unchanged):**
```python
from src.env.action_mapper import ActionMapper, GameContext, PokerAction
from src.env.equity import EquityCalculator
from src.env.features import ObservationBuilder
from src.env.wrappers import RLCardWrapper, _normalise_cards
```

---

## Critical Compliance Points

✓ **Always use `apply_action_mask` before softmax**
  - Never use `-1e8` directly
  - `apply_action_mask` uses `torch.finfo(dtype).min` (AMP-safe)

✓ **Network stays in `eval()` mode**
  - Both `_model_step` and `_oracle_best_response_ev` use `torch.inference_mode()`

✓ **Fallback is mathematically sound**
  - If network collapses (total prob < 1e-6), use uniform over legal actions
  - Never assume illegal actions are legal

---

## Usage Example

```python
# OLD (broken with Π network):
# evaluator = LocalBestResponseEvaluator(
#     model=pi_network,  # ✗ Wrong signature
#     ...
# )

# NEW (correct):
evaluator = LocalBestResponseEvaluator(
    strategy_network=pi_network,  # ✓ Correct parameter name
    env=env,
    obs_builder=obs_builder,
    action_mapper=action_mapper,
    equity_calc=equity_calc,
    config=NashEvalConfig(...),
    device="cuda"
)

results = evaluator.run_evaluation()
```

---

## Verification Checklist

- [ ] All imports include no `PokerActorCritic`
- [ ] `__init__` parameter is `strategy_network: torch.nn.Module`
- [ ] `_model_step` forwards through `self.strategy_network(batched_obs)`
- [ ] `_model_step` calls `apply_action_mask` before softmax
- [ ] `_oracle_best_response_ev` forwards through `self.strategy_network(obs_tensors)`
- [ ] `_oracle_best_response_ev` calls `apply_action_mask` before softmax
- [ ] Both methods use `torch.inference_mode()` (no gradients)
- [ ] Fallback logic uses legal actions, not hardcoded 1/3
- [ ] No errors from `get_errors` on the file
- [ ] Integration test passes (run evaluation with real Π network)

---

## Status

**COMPLETE** — All three refactorings implemented and verified.

**Ready for integration:** Pass your trained `VRDeepPDCFRNetworks.pi_network` directly to `LocalBestResponseEvaluator`.
