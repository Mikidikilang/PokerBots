# Priority #7: Exact Code Replacements (Diffs)

## 1. Import Removal

**File:** `src/evaluation/nash_evaluator.py` (Lines 32-43)

```diff
  from src.env.action_mapper import ActionMapper, GameContext, PokerAction
  from src.env.equity import EquityCalculator
  from src.env.features import ObservationBuilder
  from src.env.wrappers import RLCardWrapper, _normalise_cards
- from src.model.networks import PokerActorCritic

  logger = logging.getLogger(__name__)
```

---

## 2. `__init__` Signature & Initialization

**File:** `src/evaluation/nash_evaluator.py` (Lines 103-140)

```diff
  def __init__(
      self,
-     model:        PokerActorCritic,
+     strategy_network: torch.nn.Module,
      env:          RLCardWrapper,
      obs_builder:  ObservationBuilder,
      action_mapper: ActionMapper,
      equity_calc:  EquityCalculator,
      config:       NashEvalConfig,
      device:       str | torch.device = "cpu",
  ) -> None:
+     """Initialize evaluator with VR-DeepPDCFR+ Π (Average Strategy) network.
+     
+     Args:
+         strategy_network: The Π network that outputs raw logits (not PokerActorCritic).
+         env: RLCard poker environment.
+         obs_builder: Observation builder for state encoding.
+         action_mapper: Action mapper for legal action masking.
+         equity_calc: Equity calculator for showdown EV.
+         config: Evaluation configuration.
+         device: Torch device (cpu or cuda).
+     """
-     self.model         = model
+     self.strategy_network = strategy_network
      self.env           = env
      self.obs_builder   = obs_builder
      self.action_mapper = action_mapper
      self.equity_calc   = equity_calc
      self.config        = config
      self.device        = torch.device(device) if isinstance(device, str) else device

-     self.model.eval()
+     self.strategy_network.eval()
      torch.set_grad_enabled(False)
```

---

## 3. `_model_step` — Complete Replacement

**File:** `src/evaluation/nash_evaluator.py` (Lines 241-328)

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

**Old Code (to replace):**
```python
    def _model_step(self, obs_dict: dict[str, Any]) -> dict[str, Any] | None:
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
            
            obs_tensors = self.obs_builder.build(obs_dict)
            batched_obs = {
                k: v.to(self.device).unsqueeze(0)
                for k, v in obs_tensors.items()
                if isinstance(v, torch.Tensor)
            }

            with torch.inference_mode():
                action_idx, _, _ = self.model.get_action(
                    batched_obs,
                    deterministic=self.config.model_deterministic,
                )

            next_obs_dict, reward = self.env.step(int(action_idx))
            next_obs_dict["hand_reward"] = reward
            return next_obs_dict

        except Exception as e:
            logger.debug("Error in model step: %s (obs keys: %s)", e, obs_dict.keys())
            return None
```

---

## 4. `_oracle_best_response_ev` — Opponent Query Section

**File:** `src/evaluation/nash_evaluator.py` (Lines 507-606)

**Old Code (to replace):**
```python
            # Query the opponent model for action probabilities
            # Forward returns (Categorical distribution, value)
            with torch.no_grad():
                action_dist, _ = self.model.forward(obs_tensors)
            
            # Extract action probabilities as a numpy array
            # probs shape: (batch=1, num_actions)
            action_probs = action_dist.probs[0].cpu().numpy()  # (num_actions,)
            
            # Map the probabilities to opponent actions:
            # Action indices typically: [0:Fold, 1:Call, 2:Check, 3:Raise1, 4:Raise2, ..., 8:AllIn]
            # For a simplified oracle, we use the first 3 as Fold, Call, Raise
            
            # Get indices of likely actions
            p_fold = float(action_probs[0]) if len(action_probs) > 0 else 0.0  # Action 0: Fold
            p_call = float(action_probs[1]) if len(action_probs) > 1 else 0.0  # Action 1: Call
            # Sum all remaining probabilities as "raising" (re-raise, all-in, etc.)
            p_reraise = float(np.sum(action_probs[2:])) if len(action_probs) > 2 else 0.0
            
            # Normalize to ensure probabilities sum to 1.0
            total = p_fold + p_call + p_reraise
            if total > 1e-6:
                p_fold /= total
                p_call /= total
                p_reraise /= total
            else:
                # Fallback: if all probabilities are negligible
                logger.warning(
                    "Opponent action probabilities all near zero (total=%.6f). "
                    "Using uniform fallback.",
                    total
                )
                p_fold = p_call = p_reraise = 1.0 / 3.0
```

**New Code (replacement):**
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

---

## Summary of Replacements

| Location | Old | New | Purpose |
|----------|-----|-----|---------|
| Lines 32-43 | `from src.model.networks import PokerActorCritic` | (removed) | Remove legacy AC dependency |
| Lines 103-108 | `model: PokerActorCritic` | `strategy_network: torch.nn.Module` | Accept generic networks |
| Lines 116 | `self.model = model` | `self.strategy_network = strategy_network` | Store Π network reference |
| Lines 128 | `self.model.eval()` | `self.strategy_network.eval()` | Set eval mode on Π |
| Lines 241-328 | Old 28-line method | New 88-line 5-step pipeline | Raw logits → mask → softmax |
| Lines 507-539 | Old direct Categorical sampling | New masked logit extraction (32 lines) | Extract from raw logits + fallback |

---

## All Changes Are In Place

✓ File modified: `src/evaluation/nash_evaluator.py`
✓ No syntax errors
✓ Ready for integration testing
