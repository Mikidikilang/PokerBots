# Priority #8: Quick Reference

## Three Code Modifications

### 1. Transition Dataclass — Add Legal Mask Field

**File:** `src/training/buffers.py`, lines 93-133

```python
@dataclass(frozen=True)
class Transition:
    infoset_features: np.ndarray
    action_probs: np.ndarray
    legal_mask: np.ndarray              # ← NEW: Binary mask of legal actions
    advantages: Optional[np.ndarray] = None
    iteration: int = 1
    reach_prob: float = 1.0
```

**Validation (in `__post_init__`):**
- legal_mask must be 1D array
- legal_mask length must equal action_probs length
- legal_mask must contain only 0.0 or 1.0

---

### 2. Buffer Manager — Accept and Pass Legal Mask

**File:** `src/training/buffers.py`, lines 528-544

```python
def add_transition(
    self,
    infoset_features: np.ndarray,
    action_probs: np.ndarray,
    legal_mask: np.ndarray,              # ← NEW PARAMETER
    advantages: np.ndarray,
    reach_prob: float = 1.0,
) -> None:
    transition = Transition(
        infoset_features=infoset_features,
        action_probs=action_probs,
        legal_mask=legal_mask,            # ← Pass to Transition
        advantages=advantages,
        iteration=self.current_iteration,
        reach_prob=reach_prob,
    )
    # ... rest unchanged
```

---

### 3. Minibatch Sampling — Return Legal Masks

**File:** `src/training/buffers.py`, lines 378-436

```python
def sample_minibatch(
    self,
    batch_size: int,
    current_iteration: int,
    replace: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # ← 4-tuple return
    """..."""
    
    # ... existing sampling logic ...
    
    # Stack legal masks alongside features and action probs
    legal_masks = np.stack([t.legal_mask for t in sampled])  # ← NEW
    
    # Return 4-tuple instead of 3-tuple
    return features, action_probs, legal_masks, sampled_weights
    #      ←      ←            ←              ← NEW       ← existing
```

---

### 4. Traverse Method — Pass Legal Mask to Buffer

**File:** `src/training/vr_deep_pdcfr_engine.py`, lines 413-423

```python
# Store transition with legal mask
self.buffer_managers[acting_player].add_transition(
    infoset_features=infoset_features,
    action_probs=target_strategy,
    legal_mask=legal_actions.astype(np.float32),  # ← NEW: Convert bool→float
    advantages=instantaneous_advantages,
    reach_prob=player_reach_probs[acting_player],
)
```

---

### 5. Strategy Loss — Apply AMP-Safe Masked Softmax

**File:** `src/training/vr_deep_pdcfr_engine.py`, lines 638-695

```python
def _compute_pi_loss(self, network_bundle, buffer_manager, batch_size, iteration_t):
    """..."""
    
    # Unpack legal masks from minibatch (← NEW)
    features, target_probs, legal_masks, time_decay_weights = \
        buffer_manager.strategy_buffer.sample_minibatch(batch_size, iteration_t)
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).to(self.device)
    target_tensor = torch.FloatTensor(target_probs).to(self.device)
    legal_masks_tensor = torch.FloatTensor(legal_masks).to(self.device)  # ← NEW
    weights_tensor = torch.FloatTensor(time_decay_weights).to(self.device)
    
    # Get raw logits
    logits = network_bundle.strategy(features_tensor)
    
    # ═══════════════════════════════════════════════════════════
    # AMP-SAFE MASKED SOFTMAX (← NEW CRITICAL SECTION)
    # ═══════════════════════════════════════════════════════════
    mask_value = torch.finfo(logits.dtype).min  # Dtype-aware (safe for float16!)
    masked_logits = torch.where(
        legal_masks_tensor.bool(),
        logits,
        torch.full_like(logits, mask_value, dtype=logits.dtype)
    )
    
    # Apply log_softmax AFTER masking (critical!)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    
    # Cross-entropy loss (unchanged)
    entropy_loss = F.kl_div(log_probs, target_tensor, reduction='none')
    entropy_loss = entropy_loss.sum(dim=-1)
    weighted_loss = (entropy_loss * weights_tensor).mean()
    
    return weighted_loss
```

---

## Critical Implementation Details

### Why dtype-aware minimum matters:

```python
# WRONG (can cause NaN in float16):
masked_logits = torch.where(mask, logits, torch.tensor(-1e8))

# CORRECT (safe for all dtypes):
mask_value = torch.finfo(logits.dtype).min  # Handles float16, float32, bfloat16
masked_logits = torch.where(mask, logits, torch.full_like(logits, mask_value, dtype=logits.dtype))
```

### Mask flow through pipeline:

```
traverse() gets legal_actions (bool array)
    ↓
Cast to float32: legal_actions.astype(np.float32)
    ↓
Pass to add_transition(legal_mask=...)
    ↓
Store in Transition.legal_mask (np.ndarray)
    ↓
sample_minibatch() returns (features, probs, masks, weights)
    ↓
Convert to torch.Tensor in _compute_pi_loss
    ↓
Apply AMP-safe masking: torch.where(mask.bool(), logits, finfo.min)
    ↓
Apply log_softmax to masked_logits
    ↓
Compute loss only over legal actions
```

---

## Verification Checklist

- [x] Transition dataclass includes legal_mask
- [x] add_transition() accepts legal_mask parameter
- [x] sample_minibatch() returns 4-tuple (features, probs, masks, weights)
- [x] traverse() passes legal_actions.astype(np.float32) to add_transition()
- [x] _compute_pi_loss() unpacks legal_masks from minibatch
- [x] Masking uses torch.finfo(dtype).min (not -1e8)
- [x] Masking applied before log_softmax (critical!)
- [x] No syntax errors in either file
- [x] Full validation in Transition.__post_init__()

---

## Status

**COMPLETE** — All behavioral cloning masked softmax fixes implemented and verified.

**Ready for integration:** Π network now prevents probability leakage to illegal actions.
