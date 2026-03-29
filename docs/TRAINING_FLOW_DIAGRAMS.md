# Training Architecture: Detailed Code Flow & Dependencies

## 1. PPO Training Loop - Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  TrainingRunner.run()  [runner.py:100-130]                     │
│  - Initializes environment, network, buffer, collector, trainer │
│  - Enters main iteration loop                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Iteration N: START  │
                    │  (max 10,000+)      │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┴────────────────────────┐
        │                                               │
        ▼                                               │
┌───────────────────────────────────────────────────────┐
│ Step 1: RolloutCollector.collect_rollout()           │
│  [collector.py]                                      │
│                                                     │
│ FOR step = 0 to 2047:                              │
│   obs_dict = env.current_obs()                     │
│   action_logits, value = network(obs_dict)         │
│   action ~ Categorical(softmax(logits))            │
│   log_prob = distribution.log_prob(action)         │
│   obs', reward, done = env.step(action)            │
│   buffer.add(obs, action, reward, log_prob, value) │
│                                                     │
│ [FIX C1] At rollout END:                          │
│   bootstrap_value = network(obs')[1]  # value head │
│   buffer.set_last_bootstrap_value(bootstrap_value)  │
│                                                     │
│ Returns: CollectStats(...)                         │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 2: RolloutBuffer.compute_gae()                  │
│  [buffer.py:270-320]                                 │
│                                                      │
│ [FIX C1] last_value = buffer.get_last_bootstrap_value()
│          (guaranteed to be V(s_T) from truncation)  │
│                                                      │
│ FOR t = 2047 down to 0:                            │
│   next_value = V(s_{t+1})  [or last_value at end]  │
│   delta_t = reward_t + γ * next_value * (1-done_t) │
│           - value_t                                 │
│   advantage_t = delta_t                             │
│               + (γλ) next_advantage                │
│   returns_t = advantage_t + value_t                │
│                                                      │
│ Normalize advantages:                               │
│   advantage = (advantage - mean) / (std + ε)      │
│                                                      │
│ Allocate consolidated tensors (O(1) indexing):     │
│   _obs_tensors, _actions_tensor, _log_probs_tensor │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 3a: PPOTrainer.train_on_buffer()                │
│  [trainer.py:135-190]                                │
│                                                      │
│ FOR epoch = 0 to num_epochs-1:  [default: 4]       │
│                                                      │
│   FOR batch in buffer.get_mini_batches():           │
│     # batch_size = 2048 / 4 = 512                   │
│     {obs_dict, actions, old_log_probs,              │
│      advantages, returns, old_values}               │
│                                                      │
│     ─────────────────────────────────────────      │
│     Step 3b: _compute_and_step(batch)              │
│     [trainer.py:220-310]                            │
│     ─────────────────────────────────────────      │
│                                                      │
│     # Forward pass                                  │
│     action_dist, new_values = network(obs_dict)    │
│     new_log_probs = action_dist.log_prob(actions) │
│     entropy = action_dist.entropy()                 │
│                                                      │
│     # Compute losses                                │
│     ──────────────────                             │
│     ratio = exp(new_log_probs - old_log_probs)    │
│     surr1 = ratio * advantages                     │
│     surr2 = clip(ratio, 1±eps) * advantages       │
│     L_policy = -mean(min(surr1, surr2))           │
│                                                      │
│     if clip_range_vf:                              │
│       v_clip = old_values + clip(..., ±range)     │
│       L_value = 0.5 * max(                         │
│         (new_values - returns)²,                   │
│         (v_clip - returns)²                        │
│       )                                             │
│     else:                                           │
│       L_value = 0.5 * (new_values - returns)²     │
│                                                      │
│     L_entropy = entropy.mean()                     │
│     L_total = L_policy                             │
│             + 0.5 * L_value                        │
│             - 0.01 * L_entropy                     │
│                                                      │
│     # [FIX C-4] Cross-rank NaN guard               │
│     if dist.initialized():                         │
│       nan_flags = (NaN in L_total, L_policy,...)  │
│       dist.all_reduce(max(nan_flags))              │
│       if any_nan: raise FloatingPointError ALL_RANKS
│                                                      │
│     # Backward pass                                 │
│     optimizer.zero_grad()                          │
│     L_total.backward()                             │
│     clip_grad_norm_(network, max_norm=0.5)        │
│     optimizer.step()                               │
│                                                      │
│     if scheduler: scheduler.step()                 │
│     ─────────────────────────────────────────      │
│                                                      │
│   Early stopping if KL divergence too large       │
│     approx_kl = ((ratio-1) - log_ratio).mean()   │
│     if approx_kl > 1.5 * target_kl: BREAK        │
│                                                      │
│ Returns: {policy_loss, value_loss, total_loss, ...}│
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 4: Orchestrator Callback                         │
│  orchestrator.on_iteration_end(iteration, stats)     │
│  [Optional: curriculum learning, reward shaping]     │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 5: DDP Synchronization                           │
│  on_ddp_sync(iteration)  [Multi-GPU distributed]    │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 6: Buffer Reset                                  │
│  buffer.reset()                                       │
│  - Clears _observations, _actions, _rewards, ...      │
│  - Resets pos to 0, full to False                    │
└───────────────┬───────────────────────────────────────┘
                │
                ├─ Checkpoint (if iteration % save_interval == 0)
                │  └─ Save network weights to disk
                │
                └─ LOOP to Iteration N+1 (unless max_iter reached)
```

---

## 2. CFR Integration Path (Blocked)

```
Alternative Flow: CFRIntegrationBridge.train_on_buffer()
                  [cfr_adapter.py:240-270]

┌──────────────────────────────────────────────────────┐
│  CFRIntegrationBridge.train_on_buffer(buffer)        │
│  [Replaces PPOTrainer.train_on_buffer() in Step 3]   │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 1. Compute GAE (same as PPO)                         │
│    buffer.compute_gae(...)                           │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 2. FOR each mini-batch in buffer.get_mini_batches()  │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 3. Convert to CFR format                             │
│    CFRTrajectoryAdapter.batch_to_cfr_trajectories()  │
│    [cfr_adapter.py:70-130]                           │
│                                                      │
│    FOR each timestep i in batch:                    │
│      obs_single = batch["observations"][i]         │
│      action_int = batch["actions"][i]              │
│      reward = batch["returns"][i]                  │
│                                                      │
│      ❌ BLOCKED: _generate_infoset_id()             │
│         └─ Hardcoded hero_cards=("A","K")          │
│         └─ Hardcoded board_cards=()                │
│         └─ ALL observations → same infoset         │
│         └─ CFR learns ONE strategy (WRONG!)        │
│                                                      │
│      ❌ BLOCKED: _extract_cards()                   │
│         └─ Returns dummy values                    │
│         └─ No tensor parsing implemented           │
│                                                      │
│      ❌ ISSUE: legal_actions = list(range(12))     │
│         └─ All 12 actions always legal            │
│         └─ ~80% regret wasted on illegal moves    │
│                                                      │
│      trajectory = {                                 │
│        "states": [obs_tensor],                     │
│        "actions": [action_int],                    │
│        "infoset_ids": [infoset_hash],  ← WRONG!   │
│        "legal_actions_per_node": [all_12],  ← SUBOPTIMAL
│        "reward": reward,                           │
│      }                                             │
│                                                      │
│      Accumulate trajectories list                  │
│                                                      │
│ Returns: list of trajectories                       │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 4. Train CFR on batch                                │
│    CFREngine.train_on_rollouts(trajectories)        │
│    [cfr_engine.py:210-280]                           │
│                                                      │
│    FOR each trajectory:                             │
│      1. compute_counterfactual_regret()            │
│         [cfr_engine.py:280-350]                     │
│         └─ Calls compute_counterfactual_values()   │
│         └─ Returns {infoset: {action: regret}}     │
│                                                      │
│      2. update_strategy_from_regrets()             │
│         [cfr_engine.py:350-380]                     │
│         └─ infoset_storage.add_regret(...)        │
│         └─ Accumulates cumulative regrets          │
│         └─ Computes regret-matched strategy        │
│                                                      │
│      3. update_network_values()  (optional)        │
│         └─ Train value network on counterfactual   │
│            values as targets                       │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 5. Aggregate stats across batches                    │
│    Normalize by batch count                         │
│    Returns: {cfr_loss, avg_regret, num_infosets}   │
└──────────────────────────────────────────────────────┘
```

---

## 3. Data Structure: Observation Dictionary

```
What RolloutCollector sees (env.current_obs()):
┌─ observation = dict[str, torch.Tensor]
│  ├─ "hero_cards": tensor(?) → Card indices (0-51)?
│  ├─ "board_cards": tensor(?) → Community card indices
│  ├─ "opponent_cards": tensor(?) → Visible opponent cards (if any)
│  ├─ "pot_odds": tensor([float]) → Pot odds
│  ├─ "stacks": tensor([[hero_stack], [opp_stack]]) → Remaining chips
│  ├─ "position": tensor([int]) → Position flag (BTN/BB/SB)
│  ├─ "action_history": tensor(?) → Previous actions (encoded?)
│  └─ ... [other env metrics]

What RolloutBuffer stores:
┌─ _observations: list[observation_dict]
│  └─ Index by step: _observations[12] = obs at step 12
│
What buffer.get_mini_batches() yields:
┌─ batch["observations"]: dict[str, tensor]
│  ├─ "hero_cards": tensor[batch_size, ...] ← batch stacked
│  ├─ "board_cards": tensor[batch_size, ...]
│  ├─ "pot_odds": tensor[batch_size, 1]
│  └─ ... [stacked tensors]
│
What cfr_adapter needs to do:
┌─ For each timestep i in batch:
│  ├─ Extract observation: obs[i] from batch["observations"][i]
│  ├─ Parse "hero_cards" tensor → ("AS", "KH")  ❌ MISSING
│  ├─ Parse "board_cards" tensor → ("QC", "JH", "TS")  ❌ MISSING
│  └─ Create infoset_hash from cards  ❌ BLOCKED
```

**Key Unknown:** Card encoding scheme
- Are indices 0-51 stored as float normalized to [0,1]?
- How to map index back to card string?
- What is exact tensor shape for cards?

---

## 4. Data Flow: PPO Buffer → CFR Engine

```
Timeline of Data Structures

[Rollout Phase - 2048 steps]
        │
        ├─ RolloutCollector._current_obs: dict (current state)
        ├─ RolloutCollector._done: bool (episode terminal)
        │
        └─ RolloutBuffer (accumulating):
           ├─ _observations: [obs_dict, obs_dict, ..., obs_dict]  [2048 items]
           ├─ _actions: [action, action, ..., action]              [2048 items]
           ├─ _rewards: [reward, ..., reward]                      [2048 items]
           ├─ _log_probs: [log_prob, ..., log_prob]               [2048 items]
           ├─ _values: [value, ..., value]                        [2048 items]
           ├─ _dones: [done, done, ..., done]                     [2048 items]
           │
           └─ [FIX C1] _last_bootstrap_value: float = V(s_T)
              (Set by collector.collect_rollout() before returning)

[GAE Phase]
        │
        ├─ buffer.compute_gae(last_value=buffer.get_last_bootstrap_value())
        │
        └─ RolloutBuffer (after compute_gae):
           ├─ _advantages: tensor[2048]
           ├─ _returns: tensor[2048]
           ├─ Consolidated tensors:
           │  ├─ _obs_tensors: {key: tensor[2048, ...]}
           │  ├─ _actions_tensor: tensor[2048]
           │  ├─ _log_probs_tensor: tensor[2048]
           │  └─ _values_tensor: tensor[2048]

[Mini-batch Phase]
        │
        ├─ buffer.get_mini_batches()
        │  └─ Shuffles indices, yields 4 mini-batches
        │
        └─ Mini-batch (batch_size=512):
           {
             "observations": {
               key: tensor[512, ...]
             },
             "actions": tensor[512],
             "old_log_probs": tensor[512],
             "advantages": tensor[512],
             "returns": tensor[512],
             "old_values": tensor[512],
           }

[PPO Training Phase]
        │
        ├─ trainer.train_on_buffer(buffer)
        │
        └─ For each batch:
           ├─ _compute_and_step(batch)
           │  ├─ forward: network(batch["observations"])
           │  ├─ compute PPO losses
           │  ├─ backward + optimizer.step()
           │
           └─ Returns: {policy_loss, value_loss, ...}

[CFR Training Phase - Alternative]
        │
        ├─ cfr_bridge.train_on_buffer(buffer)
        │
        ├─ buffer.compute_gae()  [same as PPO]
        │
        └─ For each mini-batch:
           ├─ CFRTrajectoryAdapter.batch_to_cfr_trajectories(batch)
           │  ├─ For each timestep i:
           │  │  ├─ ❌ _generate_infoset_id(obs_dicts, i)
           │  │  │   └─ Should: extract ("AS","KH") from obs_dicts["hero_cards"][i]
           │  │  │   └─ Actually: returns dummy ("A","K")
           │  │  │
           │  │  └─ ❌ _extract_cards(obs_dicts, i, "hero")
           │  │      └─ Should: parse tensor → card string
           │  │      └─ Actually: returns ("A","K")
           │  │
           │  └─ Returns: [...trajectories...]
           │
           ├─ CFREngine.train_on_rollouts(trajectories)
           │  ├─ compute_counterfactual_regret()
           │  ├─ update_strategy_from_regrets()
           │  ├─ update_network_values()
           │
           └─ Returns: {cfr_loss, avg_regret, ...}
```

---

## 5. Class Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│ TrainingRunner                                              │
│ ├─ network: nn.Module (actor-critic network)               │
│ ├─ env: PokerEnvironment                                   │
│ ├─ obs_builder: ObservationBuilder                         │
│ ├─ buffer: RolloutBuffer                                  │
│ │  ├─ config: RolloutBufferConfig                         │
│ │  ├─ [stores] obs, actions, rewards, log_probs, values  │
│ │  └─ [generates] advantages, returns per GAE             │
│ ├─ trainer: PPOTrainer  ← OR ─→ CFRIntegrationBridge      │
│ │  PPOTrainer:                                            │
│ │  ├─ optimizer: torch.optim.Adam                         │
│ │  └─ [method] train_on_buffer(buffer) → dict stats      │
│ │                                                          │
│ │  CFRIntegrationBridge: ← REPLACEMENT (NOT YET INTEGRATED)
│ │  ├─ cfr_engine: CFREngine                              │
│ │  ├─ adapter: CFRTrajectoryAdapter                      │
│ │  └─ [method] train_on_buffer(buffer) → dict stats      │
│ │
│ ├─ collector: RolloutCollector                           │
│ │  ├─ network: nn.Module                                │
│ │  ├─ env: PokerEnvironment                             │
│ │  ├─ buffer: RolloutBuffer                             │
│ │  └─ [method] collect_rollout(n_steps)                │
│ │     └─ [FIX C1] Calls buffer.set_last_bootstrap_value()│
│ │
│ └─ orchestrator: Orchestrator (optional)                │
│    └─ on_iteration_end(iteration, stats)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CFREngine (NEW - Not Yet Integrated)                       │
│ ├─ config: CFRConfig                                       │
│ ├─ network: CounterfactualValueNetwork                     │
│ │  ├─ network: nn.Module (wrapped network)               │
│ │  └─ [method] forward(obs) → (logits, value)           │
│ ├─ optimizer: torch.optim.Adam                            │
│ ├─ infoset_storage: InformationSetStorage               │
│ │  ├─ infosets: dict[str, InformationSet]               │
│ │  └─ [methods] get_or_create, add_regret, get_strategy │
│ ├─ regret_buffer: RegretBuffer                           │
│ │  ├─ samples: list[RegretSample]                       │
│ │  └─ [method] add_sample(...) via reservoir sampling   │
│ ├─ regret_trainer: RegretNetworkTrainer                 │
│ ├─ strategy_buffer: StrategyBuffer                       │
│ │  └─ samples: list[StrategySample]                    │
│ ├─ strategy_trainer: StrategyNetworkTrainer             │
│ └─ mccfr: MCCFRTraversal (optional)                     │
│    ├─ env: PokerEnvironment                             │
│    ├─ network: nn.Module                                │
│    ├─ infoset_storage: InformationSetStorage            │
│    └─ [method] external_sampling_traversal(...)        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CFRIntegrationBridge (NEW - Not Yet Integrated)           │
│ ├─ cfr_engine: CFREngine                                 │
│ ├─ infoset_storage: InformationSetStorage               │
│ ├─ adapter: CFRTrajectoryAdapter                        │
│ └─ [method] train_on_buffer() ← PPOTrainer replacement  │
│    └─ Calls CFREngine.train_on_rollouts() per batch    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CFRTrajectoryAdapter                                       │
│ ├─ infoset_storage: InformationSetStorage               │
│ ├─ obs_keys_seen: set[str]                              │
│ │                                                        │
│ └─ [method] batch_to_cfr_trajectories(batch)           │
│    ├─ _flatten_obs_dict()                              │
│    ├─ _generate_infoset_id()  ❌ HARDCODED            │
│    ├─ _extract_cards()  ❌ RETURNS DUMMY              │
│    └─ _obs_dict_to_tensor()                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Method Call Graph: Single Training Iteration

```
TrainingRunner._run_single_iteration()
│
├─ 1. collector.collect_rollout(2048 steps)
│     │
│     └─ RolloutCollector.collect_rollout()
│        │
│        ├─ For step = 0..2047:
│        │  ├─ obs_dict = env.current_obs()
│        │  ├─ action_logits, value = network(obs_dict)
│        │  ├─ action ~ Categorical(softmax(logits))
│        │  ├─ obs', reward, done = env.step(action)
│        │  ├─ buffer.add(obs, action, reward, log_prob, value, done)
│        │  │
│        │  ├─ [FIX C1] if done or step==2047:
│        │  │           bootstrap_value = network(obs')[1]
│        │  │           buffer.set_last_bootstrap_value(bootstrap_value)
│        │  │
│        │  └─ [if final step] break
│        │
│        └─ Returns: CollectStats(steps, returns_mean, episodes_count)
│
├─ 2. buffer.compute_gae(last_value=buffer.get_last_bootstrap_value())
│     │
│     └─ RolloutBuffer.compute_gae()
│        │
│        ├─ last_value = buffer.get_last_bootstrap_value()  [FIX C1]
│        ├─ For t = 2047..0:
│        │  ├─ delta_t = r_t + γ * next_value * (1-done) - V(s_t)
│        │  ├─ advantage_t += delta_t * (γλ)^0 + next_advantage * (γλ)^1 + ...
│        │
│        ├─ Normalize advantages
│        └─ _consolidate_tensors()  [allocate batch tensors]
│
├─ 3. trainer.train_on_buffer(buffer)
│     │
│     ├─ PPO Path:
│     │  └─ PPOTrainer.train_on_buffer()
│     │     │
│     │     ├─ For epoch = 0..3:
│     │     │  └─ For batch in buffer.get_mini_batches():
│     │     │     └─ _compute_and_step(batch)
│     │     │        ├─ Forward: action_dist, values = network(obs)
│     │     │        ├─ Compute losses (policy, value, entropy)
│     │     │        ├─ [FIX C-4] Cross-rank NaN guard
│     │     │        ├─ Backward: L.backward()
│     │     │        ├─ Clip grads, optimizer.step()
│     │     │        └─ Returns: {policy_loss, value_loss, ...}
│     │     │
│     │     └─ Returns: {policy_loss_avg, value_loss_avg, ...}
│     │
│     └─ CFR Path (NOT YET INTEGRATED):
│        └─ CFRIntegrationBridge.train_on_buffer()
│           │
│           ├─ buffer.compute_gae()  [same]
│           │
│           ├─ For batch in buffer.get_mini_batches():
│           │  │
│           │  ├─ CFRTrajectoryAdapter.batch_to_cfr_trajectories(batch)
│           │  │  │
│           │  │  ├─ For each timestep i:
│           │  │  │  ├─ obs_single = extract observation[i]
│           │  │  │  ├─ ❌ infoset_id = _generate_infoset_id()  [BLOCKED]
│           │  │  │  ├─ ❌ legal_actions hardcoded to [0..11]  [ISSUE]
│           │  │  │  └─ trajectory = new entry
│           │  │  │
│           │  │  └─ Returns: list of trajectories
│           │  │
│           │  └─ CFREngine.train_on_rollouts(trajectories)
│           │     │
│           │     ├─ For trajectory:
│           │     │  ├─ regrets = compute_counterfactual_regret()
│           │     │  │  └─ regrets = compute_counterfactual_values()
│           │     │  │
│           │     │  └─ update_strategy_from_regrets()
│           │     │     └─ infoset_storage.add_regret(infoset, action, regret)
│           │     │
│           │     └─ Returns: {cfr_loss, avg_regret, ...}
│           │
│           └─ Returns: {cfr_loss_avg, avg_regret_avg, ...}
│
├─ 4. on_iteration_end(iteration, iter_stats)  [Orchestrator callback]
│     
├─ 5. on_ddp_sync(iteration)  [Multi-GPU sync]
│
└─ 6. buffer.reset()  [Clear for next iteration]
```

---

## 7. Critical Blocking Points

```
BLOCKER #1: Observation Decoding
┌─────────────────────────────────────────────┐
│  cfr_adapter.py:130-160  [_generate_infoset_id]
│
│  CURRENT:
│    hero_cards = ("A", "K")         ← HARDCODED
│    board_cards = ()                ← HARDCODED
│    action_history = ()             ← HARDCODED
│
│  REQUIRED:
│    hero_cards = _extract_cards(obs_dicts, idx, "hero")
│    board_cards = _extract_cards(obs_dicts, idx, "board")
│    action_history = _extract_action_history(obs_dicts, idx)
│
│  DEPENDENCY:
│    Must understand card encoding in features.py
│    ├─ Card representation type (int, float, tensor)
│    ├─ Range (0-51 after encoding?)
│    ├─ Decoding function: index → "AS"/"KH"/etc
│    └─ Example: 0→"2C", 1→"2D", ..., 51→"AS"
└─────────────────────────────────────────────┘

BLOCKER #2: Legal Actions Hardcoded
┌─────────────────────────────────────────────┐
│  cfr_adapter.py:110  [batch_to_cfr_trajectories]
│
│  CURRENT:
│    legal_actions = list(range(12))  ← ALL ACTIONS ALWAYS LEGAL
│
│  REQUIRED:
│    legal_actions = _get_legal_actions(batch, step_idx)
│
│  OPTIONS:
│    A. env.get_legal_actions()
│       └─ Need environment reference in adapter
│    B. buffer.legal_actions[step_idx]
│       └─ Need to store in RolloutBuffer
│    C. obs_dicts["legal_action_mask"][idx]
│       └─ Need to add to observation dict
│
│  PREFERRED: Option B (cleanest integration)
└─────────────────────────────────────────────┘

ISSUE #3: Runner Integration Not Started
┌─────────────────────────────────────────────┐
│  runner.py:75-80  [__init__]
│
│  CURRENT:
│    self.trainer = PPOTrainer(...)
│
│  REQUIRED:
│    if config.get("use_cfr"):
│      cfr_engine = CFREngine(...)
│      self.trainer = CFRIntegrationBridge(cfr_engine)
│    else:
│      self.trainer = PPOTrainer(...)
│
│  DEPENDENCY:
│    - CFRConfig.from_dict() must parse config.yaml
│    - cfr_engine must be instantiated with network, device, env
└─────────────────────────────────────────────┘
```

---

## 8. Test Points for Validation

```
Unit Tests:
┌─ Test 1: Card Extraction
│  ├─ Input: obs_dicts["hero_cards"] = tensor([...])
│  ├─ Process: _extract_cards(..., "hero")
│  └─ Assert: Returns 2-tuple of card strings ("AS", "KH")
│
├─ Test 2: Infoset Hashing
│  ├─ Input: ("AS", "KH"), ("QC", "JH"), []
│  ├─ Process: hash_infoset(...)
│  └─ Assert: Returns deterministic SHA1 hash string
│
├─ Test 3: Legal Actions
│  ├─ Input: batch with mixed legal/illegal actions
│  ├─ Process: _get_legal_actions(batch, step_idx)
│  └─ Assert: Returns non-full list for each state
│
├─ Test 4: Trajectory Conversion
│  ├─ Input: PPO mini-batch (512 samples)
│  ├─ Process: batch_to_cfr_trajectories(batch)
│  └─ Assert: Returns 512 trajectories with valid infoset IDs
│
├─ Test 5: CFREngine Integration
│  ├─ Input: list of 512 trajectories
│  ├─ Process: cfr_engine.train_on_rollouts(trajectories)
│  └─ Assert: Returns valid stats dict with cfr_loss, avg_regret
│
└─ Test 6: End-to-End Runner Integration
   ├─ Input: config.yaml with use_cfr: true
   ├─ Process: TrainingRunner.run() for 1 iteration
   └─ Assert: No crashes, CFR stats logged

Integration Tests:
└─ Test: 10 iterations with CFR vs PPO
   ├─ Compare learned strategies
   ├─ Verify CFR regret decreasing over time
   └─ Validate infoset count > 1
```
