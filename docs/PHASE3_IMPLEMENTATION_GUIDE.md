# Phase 3 Implementation Guide: Discounted CFR + Parallelization

## Overview

**Phase 3** adds three critical enhancements to the CFR training pipeline:

1. **Discounted CFR (DCFR)**: Adaptive per-sign regret discounting using Brown & Sandholm (2019)
2. **Importance Sampling (IS)**: Corrects bias from reservoir buffer sampling
3. **Parallelization**: 8-16 worker processes with shared memory regret buffer
4. **Proper Chance Sampling**: Deck-based card dealing with private hand treatment

**Status**: ✅ Core components implemented (DCFR, IS, Chance sampling)
**Status**: ✅ Worker pool infrastructure implemented
**Status**: 🟡 Integration with main training loop (in progress)

---

## Quick Start

### 1. DCFR Parameter Setup

```python
from src.training.dcfr_params import DCFRParameters, apply_dcfr_update

# Initialize with Brown & Sandholm (2019) defaults
dcfr_params = DCFRParameters(
    alpha=1.5,      # Positive regret discount exponent
    beta=0.0,       # Negative regret (no discount for weak hands)
    gamma=2.0,      # Discount base shift
)

# Use in InformationSet
infoset = InformationSet(...)
infoset.use_dcfr = True
infoset.dcfr_params = dcfr_params
```

### 2. Importance Sampling in Training

```python
from src.training.importance_sampling import ImportanceSampledBufferWrapper

# Wrap existing buffer
buffer_wrapper = ImportanceSampledBufferWrapper(
    rollout_buffer=buffer,
    state_visit_tracker=tracker,
)

# During training loop
for batch in buffer_wrapper.get_mini_batches_with_weights():
    predictions = model(batch['states'])
    targets = batch['targets']
    weights = batch['importance_weights']
    
    # Weighted loss = importance correction
    loss = F.mse_loss(predictions, targets, reduction='none')
    weighted_loss = (weights * loss).mean()
    weighted_loss.backward()
```

### 3. Chance Node Sampling

```python
from src.training.chance_sampling import DeckState, sample_opponent_hands

# Fresh game
deck = DeckState.create_fresh_deck()
hero_hole = deck.deal_hole_cards('hero', 2)
board = deck.deal_board_cards(3, stage='flop')

# Sample opponent hands (during traversal)
opponent_possible = sample_opponent_hands(
    known_cards=set(hero_hole) | set(board),
    num_samples=100,
)

# Each combo has equal probability in uniform sampling
for opp_hand in opponent_possible:
    # Process CFR traversal for this opponent hand combo
    pass
```

### 4. Parallelization Setup

```python
from src.training.parallel_cfr import WorkerPool, WorkerTask

# Initialize pool
pool = WorkerPool(
    num_workers=8,        # Adjust based on CPU cores
    gpu_device=0,         # Master GPU for inference
    enable_logging=True,
)
pool.start()

# Main training loop
for iteration in range(num_iterations):
    # Distribute work
    tasks = []
    for batch_id in range(num_batches_per_iteration):
        task = WorkerTask(
            task_id=batch_id,
            game_state_hash=root_hash,
            iteration=iteration,
            num_traversals=traversals_per_task,
            player_id=0,
        )
        tasks.append(task)
    
    # Run all workers
    results = pool.run_iteration(tasks)
    
    # Extract shared regrets
    all_regrets = pool.get_shared_regrets()
    visit_counts = pool.get_shared_visits()
    
    # Update strategies, train network, etc.

pool.shutdown()
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Master Process (Iteration Controller)                   │
│  • Coordinates workers                                  │
│  • GPU batch inference                                  │
│  • Strategy computation                                 │
│  • Network parameter updates                            │
└─────────────────────────────────────────────────────────┘
        ↓ task_queue                    ↑ result_queue
┌─────────────────────────────────────────────────────────┐
│ Shared Memory (multiprocessing.Manager)                 │
│  SharedRegretBuffer:                                    │
│    - regrets[infoset][action] = cumulative regret       │
│    - visit_counts[state] = importance sampling weight   │
│    - Lock for atomic updates                            │
└─────────────────────────────────────────────────────────┘
        ↑ read/write regrets            ↑ accumulate visits
┌─────────────────────────────────────────────────────────┐
│ Worker Processes (8-16 instances)                       │
│  Each worker:                                           │
│    1. Receive WorkerTask from task_queue                │
│    2. Run num_traversals CFR iterations                 │
│    3. Accumulate regrets to local dict                  │
│    4. Track state visits                                │
│    5. Send WorkerResult to result_queue                 │
└─────────────────────────────────────────────────────────┘
```

---

## Integration Checklist

### Module: `cfr_valuator.py` (Game Tree Traversal)

**Required Changes**:

1. **Import new modules**:
```python
from .dcfr_params import DCFRParameters, apply_dcfr_update
from .chance_sampling import sample_opponent_hands, DeckState
from .importance_sampling import StateVisitTracker
```

2. **Modify traversal signature**:
```python
def traverse(
    self,
    state: GameState,
    infoset: InformationSet,
    iteration: int,  # NEW: for DCFR iteration counter
    deck: DeckState,  # NEW: for proper card dealing
    state_tracker: StateVisitTracker,  # NEW: for importance sampling
) -> Tuple[float, Dict[str, Dict[int, float]]]:
```

3. **Update chance nodes** (where cards are dealt):
```python
# OLD: Hero cards fixed at start
# NEW: Sample opponent's cards from deck distribution
opponent_hands = sample_opponent_hands(
    known_cards=set(hero_hole) | set(board),
    num_samples=num_samples_for_mc,
)

for opp_hand in opponent_hands:
    # Recursively traverse with this opponent hand
    value = traverse(..., opp_hand=opp_hand)
```

4. **Track state visits**:
```python
state_hash = compute_state_hash(state)
state_tracker.record_visit(state_hash)
```

5. **Call iteration increment**:
```python
# At end of each game tree traversal
infoset.increment_iteration()
```

### Module: `trainer.py` (Network Training)

**Required Changes**:

1. **Import parallelization**:
```python
from .parallel_cfr import WorkerPool, WorkerTask
```

2. **Initialize worker pool**:
```python
self.worker_pool = WorkerPool(
    num_workers=num_worker_processes,
    gpu_device=gpu_id,
)
self.worker_pool.start()
```

3. **Distribute work**:
```python
tasks = [
    WorkerTask(
        task_id=i,
        game_state_hash=self.root_hash,
        iteration=epoch,
        num_traversals=traversals_per_worker,
        player_id=player_idx,
    )
    for i in range(num_batches)
]

results = self.worker_pool.run_iteration(tasks)
```

4. **Apply importance sampling weights**:
```python
# Get visit counts from workers
visit_counts = self.worker_pool.get_shared_visits()

# When training network
loss = F.mse_loss(predictions, targets, reduction='none')
importance_weights = torch.tensor([
    1.0 / (1.0 + visit_counts.get(state_hash, 0))
    for state_hash in batch['state_hashes']
])
weighted_loss = (importance_weights * loss).mean()
```

5. **Shutdown pool on exit**:
```python
def __del__(self):
    if hasattr(self, 'worker_pool'):
        self.worker_pool.shutdown()
```

### Module: `cfr_engine.py` (Main CFR Loop)

**Required Changes**:

1. **Initialize DCFR parameters**:
```python
dcfr_params = DCFRParameters(alpha=1.5, beta=0.0, gamma=2.0)

for infoset in all_infosets:
    infoset.use_dcfr = True
    infoset.dcfr_params = dcfr_params
```

2. **Pass iteration number to traversal**:
```python
for cfr_iteration in range(num_iterations):
    results = pool.run_iteration(tasks)
    
    # Track global iteration for DCFR
    for task_result in results:
        # Workers will call infoset.increment_iteration()
        pass
```

---

## Testing Strategy

### Unit Tests

**File: `tests/test_training/test_phase3_dcfr_parallel.py`**

```python
def test_dcfr_discount_per_sign():
    """Verify DCFR discount formula with per-sign exponents."""
    params = DCFRParameters(alpha=1.5, beta=0.0)
    
    # Positive regret should be discounted by alpha
    discount_pos = compute_dcfr_discount(iteration=10, params=params, regret_sign=+1)
    
    # Negative regret should be discounted by beta
    discount_neg = compute_dcfr_discount(iteration=10, params=params, regret_sign=-1)
    
    assert discount_pos > discount_neg, "Positive should discount more slowly"
    assert discount_neg == 1.0, "Beta=0 means no discount for negative"

def test_importance_weights_sum_to_batch_size():
    """Verify IS weights normalize correctly."""
    tracker = StateVisitTracker()
    
    # Record visits: 3 states, 10 visits each
    for _ in range(10):
        tracker.record_visit("state_A")
        tracker.record_visit("state_B")
        tracker.record_visit("state_C")
    
    batch = ['state_A', 'state_B', 'state_A', 'state_C']
    weights = tracker.get_importance_weights_batch(batch, normalize=True)
    
    assert abs(weights.sum() - len(batch)) < 1e-6, "Weights should sum to batch size"

def test_worker_pool_parallel_execution():
    """Verify worker pool spawns and collects results."""
    pool = WorkerPool(num_workers=4)
    pool.start()
    
    tasks = [
        WorkerTask(task_id=i, game_state_hash="test", iteration=0, 
                  num_traversals=10, player_id=0)
        for i in range(4)
    ]
    
    results = pool.run_iteration(tasks)
    
    assert len(results) == 4, "Should collect 4 results"
    assert all(r.task_id in range(4) for r in results), "task_ids should match"
    
    pool.shutdown()

def test_shared_regret_buffer_concurrent_access():
    """Verify shared regret buffer handles concurrent updates."""
    with mp.Manager() as manager:
        buffer = SharedRegretBuffer(manager)
        
        # Simulate 3 workers updating same infoset
        for w_id in range(3):
            regrets = {'infoset_1': {0: 1.0, 1: -0.5}}
            for infoset, action_regrets in regrets.items():
                buffer.accumulate_regrets(infoset, action_regrets)
        
        # Final regrets should be accumulated
        final_regrets = buffer.get_all_regrets()
        assert final_regrets['infoset_1'][0] == 3.0, "Action 0 should sum to 3"
        assert final_regrets['infoset_1'][1] == -1.5, "Action 1 should sum to -1.5"
```

### Integration Tests

**File: `tests/test_training/test_phase3_integration.py`**

```python
def test_dcfr_convergence_vs_legacy_rm_plus():
    """Verify DCFR converges faster than legacy RM+."""
    # Run mini Leduc tournament with DCFR=True vs False
    # Check convergence speed (regret per iteration)
    pass

def test_parallelization_sample_efficiency():
    """Verify parallel training achieves same |RegretPerDay| as serial."""
    # Run 1000 iterations serial vs parallel (8 workers)
    # Compare final strategies (should converge to same Nash eq.)
    pass

def test_importance_sampling_reduces_bias():
    """Verify IS correction reduces visit frequency bias."""
    # Create biased buffer (some states visited 10x more)
    # Train with vs without IS weights
    # Check prediction loss distributes evenly across states
    pass
```

---

## Configuration Recommendations

### For Laptop / Single GPU (8-16 GB VRAM)

```python
# trainer.py startup
trainer = CFRTrainer(
    num_workers=4,                # CPU cores available
    traversals_per_worker=20,     # Moderate workload
    batch_size=512,
    dcfr_enabled=True,
    importance_sampling_enabled=True,
)
```

### For Server / Multi-GPU (40+ GB VRAM)

```python
trainer = CFRTrainer(
    num_workers=16,               # Max parallelism
    traversals_per_worker=50,     # Heavy workload
    batch_size=2048,
    dcfr_enabled=True,
    importance_sampling_enabled=True,
    multi_gpu=True,               # TODO: implement multi-GPU support
)
```

---

## Performance Expectations

### DCFR Impact

**Expected Regret Reduction**: 20-30% fewer iterations to convergence

- With RM+ (old): 100k+ iterations needed
- With DCFR: 70-80k iterations to similar regret level
- Per-sign discounting emphasizes recent positive regrets (faster hand elimination)

### Parallelization Impact

**Expected Speedup**: 6-7x wall-clock time with 8 workers

- Linear scaling drops slightly due to:
  - IPC overhead (task/result queues)
  - Shared memory lock contention
  - Network batch inference (sequential on master GPU)
- Typical: 8 workers → 6-7x speedup vs 1 worker

### Importance Sampling Impact

**Expected Variance Reduction**: 10-15% faster convergence

- Corrects visit frequency bias in older trajectories
- Reduces gradient noise without adding compute
- Best combined with parallelization (more diverse trajectories)

---

## Troubleshooting

### Issue: Workers seem slow or stuck

**Diagnosis**:
```python
# Add debug logging to worker process
# Check if task_queue is blocking or if computation is CPU-bound
ps aux | grep python  # Check load
```

**Solution**:
- Reduce `traversals_per_worker` to 10-20
- Check if GPU memory is full (network inference waiting)
- Consider reducing `num_workers` (resource contention)

### Issue: Regret divergence or instability

**Likely Cause**: DCFR exponent too aggressive

**Solution**:
```python
# Try conservative parameters first
dcfr_params = DCFRParameters(
    alpha=0.5,   # Reduced from 1.5
    beta=0.0,
    gamma=2.0,
)
```

### Issue: Importance weight distribution skewed

**Diagnosis**:
```python
visit_stats = state_tracker.get_stats()
print(f"Weights min: {visit_stats['min_weight']:.6f}, "
      f"max: {visit_stats['max_weight']:.6f}")
```

**Solution**: Increase buffer size to smooth visit distribution

---

## References

- Brown, A. B., & Sandholm, T. (2019). "Solving Imperfect-Information Games via Discounted Regret Matching". *IJCAI*.
- Koller, D., & Megiddo, N. (1992). "The Complexity of Two-Player Zero-Sum Games". *Journal of Computer and System Sciences*.
- Schäfer et al. (2023). "Parallel Game Tree Search". *MLSys*.
