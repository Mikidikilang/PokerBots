% Phase 3 Implementation Plan: Full MCCFR with Proper Game Tree

## Architecture Overview

Phase 3 upgrades the CFR training from Phase 2.5's simplified approach to a full production-ready system with:
1. Proper game tree traversal via External Sampling MCCFR
2. Accurate counterfactual computation  
3. Full regret accumulation and strategy convergence

## Components to Build

### 1. MCCFRTraversalWorker (src/training/cfr_traversal.py)
**Status**: Skeleton exists, needs completion

Key methods to implement:
- `external_sampling_traversal()` - Core recursive algorithm
  - Terminal state detection
  - Current player extraction from game state
  - Legal action enumeration
  - Regret computation per action
  - Opponent action sampling with importance weighting

- `_compute_counterfactual_reach_probs()` - Importance sampling
  - Track P(reach | strategies)
  - Scale regrets by opponent reach probability
  - Handle card abstraction bucket weights

- `_get_infoset_id()` - Hash game state
  - Extract player, hole cards, board cards
  - Encode action history
  - Return canonical infoset ID

- `_game_state_to_features()` - Observation encoding
  - Convert game state → neural network features
  - Handle card abstraction bucketing
  - Normalize for network input

### 2. MCCFREngine Updates (src/training/cfr_engine.py)
**Status**: Phase 2.5 simplified version exists

Required changes:
- Integrate MCCFRTraversal.external_sampling_traversal()
- Replace Phase 2.5 regret computation with full game tree traversal
- Batch multiple traversals per training iteration
- Accumulate regrets across traversals

### 3. CardAbstraction Integration (src/env/card_abstraction.py)
**Status**: Already exists (per MASTER_NOTE.md)

Usage:
- Map flop/turn/river cards to buckets
- Weight samples by P(concrete_cards | bucket)
- Seed traversal with abstract game states

### 4. Test Suite (Phase 3D)
Test coverage:
- Traversal on tiny games (2-card hand)
- Infoset hashing consistency
- Regret accumulation over 10 iterations
- Strategy convergence metrics
- Kuhn poker benchmark (GTO comparison)

## Phase 3 Phases

### Phase 3.1: Core Game Tree Traversal
**Time**: 3-4 hours
**Deliverable**: MCCFRTraversal fully functional

Steps:
1. Implement external_sampling_traversal() recursion
2. Terminal state detection
3. Current player extraction from game state  
4. Infoset ID generation with action history
5. Unit tests on mock game tree

### Phase 3.2: Proper Counterfactual Computation
**Time**: 2-3 hours
**Deliverable**: Reach probability tracking

Steps:
1. Implement reach probability chains
2. Opponent reach prob scaling for counterfactual values
3. Card abstraction bucket weighting
4. Regret importance scaling

### Phase 3.3: Integration & Optimization
**Time**: 2-3 hours
**Deliverable**: Full pipeline with regret accumulation

Steps:
1. Integrate traversal into CFREngine
2. Batch traversals per iteration
3. Regret network updates from accumulated regrets
4. Strategy network updates from regret matching

### Phase 3.4: Validation & Benchmarking
**Time**: 2-3 hours
**Deliverable**: Test suite + Kuhn poker convergence

Steps:
1. Traversal unit tests + integration tests
2. Regret accumulation validation
3. Kuhn poker convergence test (→ GTO)
4. Exploitability measurement

## Technical Details

### External Sampling Formula
```
For player p at infoset h:

1. Get current strategy π(a | h) from regret matching
2. For each legal action a:
   h' = h + a
   v(a) = external_sampling_traversal(h', p)
3. Compute counterfactual regrets:
   μ_{-p}(h) = product of opponent action probs reaching h
   regret(a) = [v(a) - v_avg] * μ_{-p}(h)
4. Store: infoset_storage.add_regret(infoset_id, a, regret)
```

### Reach Probability Chain
```
reach_probs = {
    player_0: ∏ P(a_t | infoset_t) for player 0's actions
    player_1: ∏ P(a_t | infoset_t) for player 1's actions
}

For counterfactual value:
  μ_{-p}(h) = reach_probs[1-p]
```

### Card Abstraction Weights
```
For each concrete card combination in bucket:
    weight(concrete_cards | bucket) = P(concrete | bucket)
    weighted_regret = regret * weight
```

## Success Criteria

Phase 3 is complete when:
- ✓ MCCFRTraversal.external_sampling_traversal() produces non-zero regrets
- ✓ Regret accumulation increases over iterations
- ✓ Strategy converges (network loss decreases)
- ✓ Kuhn poker benchmark: <10% exploitability vs GTO after 1000 iterations
- ✓ All Phase 3D tests pass
- ✓ No regressions in Phase 2.5D tests

## File Dependencies

```
cfr_engine.py
├─ calls → cfr_traversal.py
├─ uses → cfr_infoset.py (InformationSetStorage)
└─ updates → regret_buffer.py, strategy_buffer.py

cfr_traversal.py  
├─ uses → cfr_infoset.py
├─ calls → env (PokerEnvironment)
├─ calls → network (forward pass)
└─ uses → card_abstraction.py (bucket weights)

cfr_adapter.py [MODIFIED]
├─ removed in Phase 3 (replaced with direct traversal)
└─ OR kept for Phase 2.5 backward compatibility
```

## Backward Compatibility

Phase 3 maintains dual paths:
- **Phase 2.5 path** (simplified): `config["cfr"]["full_mccfr"] = false`
  - Uses Phase 2.5 regret approximation
  - Faster, proof-of-concept
  
- **Phase 3 path** (production): `config["cfr"]["full_mccfr"] = true`
  - Uses full game tree traversal
  - Converges to Nash equilibrium
  - Proper exploitability measurement

Both paths use same regret/strategy networks, same training loop.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Game tree explosion (1000s of nodes) | External sampling → O(1) samples per iteration |
| Infinite recursion | depth limit, terminal detection |
| NaN regrets | numerical stability checks, clipping |
| Slow traversal | Vectorize with batch operations, cache strategies |
| Infoset hash collisions | Use (player, cards, history) tuple, cryptographic hash |

---

## Next: Phase 3.1 Implementation

Starting with MCCFRTraversal core algorithm.
