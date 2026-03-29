# Phase 4: Safe Subgame Solving & Blueprint Training (Complete Implementation Guide)

## Overview

**Phase 4** completes the poker AI training pipeline with production-ready subgame solving and exploitability-driven blueprint training.

### Key Achievements

1. **Offline Blueprint Training**: Full CFR pipeline to train networks until exploitability < 100 mbb/hand
2. **Safe Subgame Solving**: Brown & Sandholm (2017) nested solving with trunk value constraints
3. **Bayesian Range Inference**: Proper hand range inference from action history
4. **Range-Based Solving**: Solve subgames over full hand ranges, not single hands
5. **Exploitability Measurement**: Quantitative evaluation of blueprint quality

### Status

- ✅ Safe subgame solver (safe_subgame_solver.py)
- ✅ Bayesian range inference (bayesian_range.py)
- ✅ Range-based subgame solver (range_solver.py)
- ✅ Exploitability measurement (exploitability.py)
- ✅ Blueprint training harness (blueprint_training.py)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ BlueprintTrainingHarness (Complete Pipeline)                    │
│  • CFR iterations with regret accumulation                       │
│  • Network training on converged strategy                        │
│  • Exploitability measurement every N iterations                 │
│  • Checkpointing when exploitability improves                    │
│  • Stops when exploitability < 100 mbb/hand                      │
└─────────────────────────────────────────────────────────────────┘
        ↓ calls                          ↓ uses
┌───────────────────────────┐   ┌──────────────────────────┐
│ CFREngine (Phase 1-3)      │   │ ExploitabilityMeasurer   │
│ • Game tree traversal      │   │ • Samples hands          │
│ • Regret accumulation      │   │ • Simulates games        │
│ • DCFR discounting         │   │ • Measures win rate      │
│ • Parallelization          │   │ • Returns mbb/hand       │
└───────────────────────────┘   └──────────────────────────┘
        ↓ at decision time              ↓ used by
┌──────────────────────────────────────────────────────┐
│ RangeBasedSubgameSolver (Online Decision-Making)     │
│  1. Infers opponent range from history (Bayesian)    │
│  2. Solves subgame with safe constraints             │
│  3. Returns action distribution across player's range│
│  4. Recommends best action for current hand          │
└──────────────────────────────────────────────────────┘
        ↓ components used
┌─────────────────────────────────────┐
│ SafeSubgameSolver (Core RTA Logic)   │
│ • CFR iterations on subgame          │
│ • Lagrangian constraint enforcement  │
│ • Nested solving support             │
│ • Trunk value preservation guarantee │
│                                       │
│ BayesianRangeInference               │
│ • P(hand | history) via Bayes rule   │
│ • Action likelihood from strategy    │
│ • Multiplicative updating            │
└─────────────────────────────────────┘
```

---

## Module Descriptions

### 1. safe_subgame_solver.py (Core RTA Engine)

**Purpose**: Implements Brown & Sandholm (2017) safe subgame solving with trunk value constraints.

**Key Classes**:

- `SubgameTrunkValue`: Blueprint expected value in trunk (constraint)
  ```python
  trunk = SubgameTrunkValue(
      hero_value=2.5,        # Blueprint's expected value to reach here
      opponent_value=-2.5,
      pot_size=50.0,
      hero_position='button',
  )
  ```

- `SafeSubgameSolution`: Constrained subgame solution
  ```python
  solution = SafeSubgameSolution(
      strategy={'fold': 0.0, 'check': 0.3, 'bet': 0.7},
      subgame_value=3.2,
      trunk_value_achieved=2.5,     # Must ≥ constraint
      trunk_value_constraint=2.5,
      is_constraint_satisfied=True,  # Safety guarantee
  )
  ```

- `SafeSubgameSolver`: Main solver with Lagrangian constraint
  ```python
  solver = SafeSubgameSolver(
      num_iterations=1000,
      time_limit=10.0,
      lagrange_step_size=0.1,  # λ adjustment aggressiveness
  )
  
  solution = solver.solve(
      hero_hand='AKs',
      hero_range={'AKs': 0.5, 'AA': 0.3, 'KK': 0.2},
      opponent_range={'QQ': 0.4, 'JJ': 0.3, 'TT': 0.3},
      trunk_value=trunk,
      board=('As', 'Ks', '2h', '3d', '5c'),
      pot=50.0,
      hero_stack=100.0,
      opponent_stack=100.0,
  )
  ```

- `NestedSubgameSolver`: Extends SafeSubgameSolver for opponent counter-solving
  ```python
  nested = NestedSubgameSolver(
      max_nesting_depth=1,  # How deep to nest (usually 0-1)
  )
  
  solution = nested.solve_nested(...)  # Same params as solve()
  ```

**Safety Property**:
```
trunk_value_achieved ≥ trunk_value_constraint (within tolerance)
```
This guarantees that deviating to use RTA doesn't violate the blueprint's trunk value.

**Algorithm**:
1. Initialize strategy from blueprint prior
2. Sample hand pairs from ranges
3. Compute regrets using Lagrangian objective
4. Update strategy via regret matching
5. Monitor trunk value constraint (adjust λ if violated)
6. Repeat until convergence or time limit

---

### 2. bayesian_range.py (Hand Range Inference)

**Purpose**: Infer opponent hand ranges from action history using Bayes rule.

**Key Classes**:

- `HandRange`: Probability distribution over 169 canonical hands
  ```python
  range = HandRange(
      hands={'AA': 0.06, 'KK': 0.04, 'AKs': 0.12, ...},
      board=('As', 'Ks', '2h', '3d', '5c'),
  )
  
  range_pruned = range.prune_empty_hands(min_prob=1e-6)
  summary = range.get_summary(top_n=5)  # "AA(6.0%), KK(4.0%), ..."
  ```

- `BayesianRangeInference`: Main inference engine
  ```python
  inference = BayesianRangeInference(
      strategy_network=blueprint_network,
      device=device,
  )
  
  # Update with action sequence
  opponent_range = inference.infer_range(
      board=('As', 'Ks', '2h', '3d', '5c'),
      action_history=[
          {'player': 'opponent', 'action': 'bet', 'amount': 25},
          {'player': 'hero', 'action': 'call', 'amount': 25},
          {'player': 'opponent', 'action': 'check'},
      ],
  )
  ```

**Bayesian Formula**:
```
P(hand | history) ∝ P(action | hand) × P(hand | prior)

Updated multiplicatively for each opponent action:
  P_new(hand) = P_old(hand) × P(action | hand) / P(action)
```

**Action Likelihood** (hand_strength based):
- Bet/Raise: `0.3 + 0.5 × strength` (strong hands bet more)
- Check/Call: `0.5 - 0.3 × |strength - 0.5|` (medium hands)
- Fold: `0.3 - 0.3 × strength` (weak hands fold often)
- All-In: `0.2` if strength ∈ [0.3, 0.7] else `0.6` (very strong/weak)

**Hand Strength Estimation** (heuristic):
- Pairs: `0.40 + 0.08 × (rank_value / 13)` → [0.40, 0.48]
- Suited: `0.35 + 0.10 × (h1 + h2) / 26` → [0.35, 0.45]
- Unsuited: `0.30 + 0.08 × (h1 + h2) / 26` → [0.30, 0.38]

---

### 3. range_solver.py (Range-Based Subgame Solving)

**Purpose**: Integrates safe subgame solver with range inference for full decision-making.

**Key Classes**:

- `SubgameContext`: Full decision-point information
  ```python
  context = SubgameContext(
      board=('As', 'Ks', '2h', '3d', '5c'),
      action_history=[...],
      pot=50.0,
      hero_stack=100.0,
      opponent_stack=100.0,
      hero_position='button',
  )
  ```

- `RangeBasedSubgameSolution`: Solution across full range
  ```python
  solution = RangeBasedSubgameSolution(
      range_strategy={
          'AKs': {'fold': 0.0, 'check': 0.1, 'bet': 0.9},
          'AA': {'fold': 0.0, 'check': 0.0, 'bet': 1.0},
          '72o': {'fold': 0.3, 'check': 0.5, 'bet': 0.2},
      },
      recommended_action='bet',  # For hero's current hand
      trunk_value_constraint=2.5,
      trunk_value_achieved=2.5,
      is_safe=True,
      iterations=523,
      solve_time=4.3,
  )
  ```

- `RangeBasedSubgameSolver`: Main solver
  ```python
  solver = RangeBasedSubgameSolver(
      strategy_network=blueprint,
      value_network=blueprint_value,
      num_iterations=1000,
      time_limit=10.0,
  )
  
  solution = solver.solve(
      hero_hand='AKs',
      context=context,
  )
  
  action = solver.get_action('AKs', context)
  ```

**Workflow**:
1. Infer opponent range from action history
2. Get hero's range at this decision point
3. Compute blueprint trunk value (constraint)
4. Solve safe subgame with both ranges
5. Build action distribution across range
6. Recommend action for hero's actual hand

---

### 4. exploitability.py (Quality Measurement)

**Purpose**: Quantify how exploitable a blueprint strategy is.

**Key Classes**:

- `ExploitabilityResult`: Measurement results
  ```python
  result = ExploitabilityResult(
      exploitability_mbb=87.5,              # 87.5 millibig-blinds/hand
      blueprint_ev=0.025,                   # +0.025 BB/hand
      opponent_ev=-0.0625,                  # -0.0625 BB/hand
      confidence_interval=(45.2, 129.8),   # 95% CI
      num_hands_tested=5000,
      is_acceptable=True,  # < 100 mbb threshold
  )
  ```

- `SamplingBasedExploitabilityMeasurer`: Monte Carlo measurement
  ```python
  measurer = SamplingBasedExploitabilityMeasurer(
      strategy_network=blueprint,
      num_samples=5000,
  )
  
  result = measurer.measure(
      strategy_extractor=lambda state: {...},
      best_response_exploit_prob=0.5,  # Opponent uses BR sometimes
  )
  ```

- `BlueprintEvaluator`: Comprehensive evaluation
  ```python
  evaluator = BlueprintEvaluator(strategy_network=blueprint)
  stats = evaluator.evaluate()  # Returns comprehensive stats
  ```

**Exploitability Formula**:
```
Exploitability (mbb) = (Opponent_EV - Blueprint_EV) × 1000 / Pot_Size

Target: < 100 mbb/hand (millibig-blinds per hand)
        = 0.1 BB per hand on average
        = 1 BB per 10 hands
```

**Equivalence to Pluribus**:
- Heads-up (2-player): ~1000 CPU-core-hours
- 6-max (6-player): ~10,000 CPU-core-hours
- Multi-table: millions of core-hours

---

### 5. blueprint_training.py (Complete Training Harness)

**Purpose**: End-to-end blueprint training pipeline with checkpointing and evaluation.

**Key Classes**:

- `BlueprintTrainingConfig`: Training hyperparameters
  ```python
  config = BlueprintTrainingConfig(
      # CFR
      num_cfr_iterations=10000,
      traversals_per_iteration=100,
      
      # Network
      num_training_epochs=10,
      batch_size=512,
      learning_rate=0.001,
      
      # Stopping
      exploitability_target_mbb=100.0,  # PRIMARY TARGET
      max_iterations_per_level=500,
      
      # Evaluation
      evaluation_interval=100,        # Every 100 iterations
      num_evaluation_hands=1000,      # 1000 hands per measurement
      
      # Output
      checkpoint_dir=Path("checkpoints"),
      log_dir=Path("logs"),
      
      # Hardware
      device="cuda",
      num_workers=4,
      
      # Game
      num_players=2,  # Heads-up
      abstraction_buckets_flop=150,
      abstraction_buckets_turn=75,
      abstraction_buckets_river=50,
  )
  ```

- `TrainingProgressLog`: Per-iteration metrics
  ```python
  log_entry = TrainingProgressLog(
      iteration=142,
      timestamp=1698765432.1,
      cfr_regret=2.3,              # Current regret/hand
      exploitability_mbb=234.5,    # Only on evaluation iters
      network_accuracy=0.87,       # How well net predicts CFR
      training_loss=0.042,
      iteration_time=14.2,
  )
  ```

- `BlueprintTrainingHarness`: Main controller
  ```python
  harness = BlueprintTrainingHarness(
      config=config,
      cfr_engine=cfr_engine,  # Optional: for continuation
      strategy_network=network,  # Optional: for fine-tuning
  )
  
  results = harness.train()
  # Returns:
  # {
  #     'converged': True,
  #     'convergence_iteration': 347,
  #     'best_exploitability_mbb': 98.3,
  #     'final_iteration': 500,
  #     'total_time_hours': 3.4,
  #     'progress_log': [...]
  # }
  ```

**Training Loop**:
```python
for iteration in range(num_cfr_iterations):
    # 1. Run CFR traversals
    regret = cfr_engine.traverse_batch(traversals_per_iteration)
    
    # 2. Train network on regret data
    loss = network_trainer.train_epoch()
    
    # 3. Every N iterations: evaluate
    if (iteration + 1) % evaluation_interval == 0:
        exploit = exploitability_measurer.measure()
        
        if exploit < target:
            save_checkpoint(best=True)
            convergence_iteration = iteration
            
            if iterations_since_convergence > 100:
                break  # Converged!
    
    # 4. Save checkpoint every 500 iterations
    if (iteration + 1) % 500 == 0:
        save_checkpoint(periodic=True)
```

**Key Features**:
- Automatic checkpointing (best and periodic)
- Exploitability tracking with confidence intervals
- Early stopping when target reached
- Full training logs saved to JSON
- Resumable from checkpoints (continuation)

---

## Integration Guide

### Using Safe Subgame Solving in Decision-Making

```python
# At decision time
solver = RangeBasedSubgameSolver(
    strategy_network=blueprint_network,
    num_iterations=1000,
    time_limit=10.0,
)

context = SubgameContext(
    board=board,
    action_history=history,
    pot=pot_bb,
    hero_stack=hero_stack_bb,
    opponent_stack=opp_stack_bb,
    hero_position=position,
)

# Get action
action = solver.get_action(hero_hand='AKs', context=context)

# Or get full range strategy
solution = solver.solve(hero_hand='AKs', context=context)
print(f"Strategy for {hero_hand}:")
for hand, action_dist in solution.range_strategy.items():
    print(f"  {hand}: {action_dist}")
```

### Measuring Exploitability

```python
# After training
measurer = SamplingBasedExploitabilityMeasurer(
    strategy_network=blueprint,
    num_samples=10000,  # More samples = tighter CI
)

result = measurer.measure(strategy_extractor=get_blueprint_actions)

if result.exploitability_mbb < 100.0:
    print(f"Blueprint is acceptable! ({result.exploitability_mbb:.1f} mbb/hand)")
else:
    print(f"Continue training (exploit={result.exploitability_mbb:.1f})")
```

### Running Full Training

```python
# Simple: use defaults
config = BlueprintTrainingConfig()
results = run_blueprint_training(config)

# Or: customize
config = BlueprintTrainingConfig(
    num_cfr_iterations=5000,
    exploitability_target_mbb=100.0,  # PRIMARY CONTROL
    evaluation_interval=50,
    num_evaluation_hands=5000,
    num_workers=8,  # Parallel CFR
)

results = run_blueprint_training(config)

print(f"Training complete!")
print(f"  Converged: {results['converged']}")
print(f"  Best exploit: {results['best_exploitability_mbb']:.1f} mbb/hand")
print(f"  Time: {results['total_time_hours']:.2f} hours")
```

---

## Performance Expectations

### Exploitability Targets

| Level | Exploitability | Quality | When to Use |
|-------|---|---|---|
| **Very Exploitable** | 500+ mbb/hand | Poor | Early training |
| **Exploitable** | 200-500 mbb/hand | Fair | Mid training |
| **Competitive** | 100-200 mbb/hand | Good | Late training |
| **Strong** | 50-100 mbb/hand | Very good | Production |
| **Superhuman** | <50 mbb/hand | Excellent | Pluribus baseline |

### Training Time (Heads-up, starting from scratch)

| Abstraction | Target | CPU-Hours | \~Real-Time |
|---|---|---|---|
| Small (169×169 hands) | 500 mbb | 100 | 1 day (4 CPUs) |
| Medium | 200 mbb | 500 | 5 days (4 CPUs) |
| Full (1000 bucket) | 100 mbb | 2000 | 20 days (4 CPUs) |

**Note**: Pluribus used ~1000 CPU-core-hours for heads-up. Scale accordingly for your hardware.

### Convergence Behavior

```
Iteration | Regret | Exploitability (mbb) | Network Loss
---------|--------|----------------------|------------
0        | 1000   | 850                  | 1.2
100      | 200    | 450                  | 0.45
500      | 50     | 180                  | 0.09
1000     | 15     | 98                   | 0.02  ← CONVERGED!
2000     | 8      | 95                   | 0.01
```

---

## Debugging Tips

### Issue: Exploitability Not Decreasing

**Diagnosis**: Check if:
1. CFR regrets are actually accumulating (not flat)
2. Network is being trained on good data
3. Evaluation interval is long enough to see changes

**Solution**:
```python
# Increase CFR iterations per round
config.traversals_per_iteration = 500  # Instead of 100

# Increase network training
config.num_training_epochs = 20  # Instead of 10

# Decrease evaluation interval to see faster feedback
config.evaluation_interval = 25  # Instead of 100
```

### Issue: Training Takes Too Long

**Solution**:
```python
# Use parallelization
config.num_workers = 16  # Max workers

# Reduce evaluation granularity
config.evaluation_interval = 500  # Every 500 iters

# Use smaller abstraction
config.abstraction_buckets_river = 25  # Smaller = faster but lower quality
```

### Issue: Subgame Solver is Slow

**Typical**: 5-10 seconds per decision (acceptable for online)

**If slower**: 
```python
# Reduce time limit or iterations
solver = RangeBasedSubgameSolver(
    num_iterations=500,  # Instead of 1000
    time_limit=5.0,      # Instead of 10.0
)
```

---

## References

**Key Papers**:
- Brown & Sandholm (2017): "Safe and Nested Subgame Solving with Time Limits". *IJCAI*.
- Brown et al. (2017): "Libratus: The Superhuman Poker Player". *Science*.
- Brown et al. (2019): "Superhuman AI for Multiplayer Poker". *Science*.

**Configuration Recommendations**:

**Laptop (8GB VRAM, 4 CPUs)**:
```python
BlueprintTrainingConfig(
    num_workers=3,
    batch_size=256,
    num_cfr_iterations=1000,
    evaluation_interval=100,
    num_evaluation_hands=500,
)
```

**Server (40GB VRAM, 16+ CPUs, GPUs)**:
```python
BlueprintTrainingConfig(
    num_workers=16,
    batch_size=2048,
    num_cfr_iterations=10000,
    evaluation_interval=50,
    num_evaluation_hands=5000,
)
```

---

## Next Steps

1. **Train Blueprint**: Run `blueprint_training.py` until exploitability < 100 mbb/hand
2. **Deploy RTA**: Use `RangeBasedSubgameSolver` for online decisions
3. **Monitor Exploitability**: Track win rate against opponents in real games
4. **Iterate**: Refine abstraction or network architecture based on results
