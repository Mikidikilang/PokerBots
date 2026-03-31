# VR-DeepPDCFR+ Training Scripts

Complete training orchestration for VR-DeepPDCFR+ on 6-Max No-Limit Hold'em.

## Overview

This directory contains three main scripts:

1. **`train_6max_vr_deep.py`** - Master training script (main entry point)
2. **`evaluate_strategy.py`** - Strategy evaluation and analysis
3. **`setup.sh`** - Environment setup and installation

## Quick Start

### 1. Setup Environment

```bash
bash scripts/setup.sh
```

This will:
- Create virtual environment
- Install dependencies (PyTorch, RLCard, etc.)
- Create necessary directories (logs, checkpoints, data)
- Run tests

### 2. Configure Training

Edit `config.yaml` with your settings:

```yaml
# CFR parameters
cfr:
  num_iterations: 50000        # Total training iterations
  max_tree_depth: 60           # Max game tree depth
  exploitability_update_freq: 250  # Eval every N iterations

# Network configuration
networks:
  shared_architecture:
    hidden_dims: [512, 512]    # Hidden layer dimensions
  
  cumulative_advantage:
    learning_rate: 1e-3
  instantaneous_advantage:
    learning_rate: 1e-3
  value_baseline:
    learning_rate: 1e-3
  average_strategy:
    learning_rate: 1e-3

# Buffer configuration
buffers:
  advantage_buffer_size: 4000000   # Advantage buffer capacity
  strategy_buffer_size: 4000000    # Strategy buffer capacity
```

### 3. Start Training

```bash
python scripts/train_6max_vr_deep.py --config config.yaml
```

## Script Details

### `train_6max_vr_deep.py`

**Main Training Master Script**

Orchestrates the complete VR-DeepPDCFR+ pipeline:

```
Initialization:
├─ Load configuration
├─ Initialize device (GPU/CPU)
├─ Setup monitoring (WandB)
├─ Initialize game environment
├─ Create per-player networks, buffers, optimizers
├─ Initialize evaluation (LBR oracle)
└─ Ready for training

Training Loop (per iteration):
├─ Start iteration
├─ Reset environment
├─ For each updating player:
│  └─ External Sampling MCCFR traversal
├─ Train all networks
├─ End iteration
├─ Log to WandB
├─ Checkpoint (every N iterations)
└─ Evaluate (every M iterations)
```

**Features:**
- ✅ Per-player network and buffer management
- ✅ External Sampling MCCFR traversal
- ✅ Gradient-based learning (Adam optimizer)
- ✅ WandB integration for monitoring
- ✅ Checkpoint management (saves to `checkpoints/`)
- ✅ Periodic LBR exploitability evaluation
- ✅ Graceful shutdown with emergency checkpointing
- ✅ Signal handling (SIGINT, SIGTERM)

**Usage:**

```bash
# Standard training
python scripts/train_6max_vr_deep.py

# Custom config
python scripts/train_6max_vr_deep.py --config my_config.yaml

# GPU training
python scripts/train_6max_vr_deep.py --device cuda

# CPU-only training
python scripts/train_6max_vr_deep.py --device cpu

# Offline WandB (no cloud sync)
python scripts/train_6max_vr_deep.py --wandb-offline
```

**Output:**
- Logs: `logs/train_YYYYMMDD_HHMMSS.log`
- Checkpoints: `checkpoints/vr_deep_pdcfr_iteration_XXXXXX.pt`
- WandB dashboard: Track in real-time at https://wandb.ai

**Key Components:**

1. **Environment** (`RLCardWrapper`)
   - 6-Max NLHE with 200BB starting stacks
   - Legal action enforcement
   - Automatic pot tracking

2. **Networks** (per player, 6 total)
   - Cumulative advantage network: $A_S^t(s) = \sum_{t'=1}^t A^{t'}(s)$
   - Instantaneous advantage network: $A^t(s)$ (VR baseline)
   - Value network: $V(s)$ (advantage baseline)
   - Average strategy network: $\sigma(s)$ (convergence target)

3. **External Sampling MCCFR**
   - One updating player per traversal
   - Uniform sampling of opponent actions
   - Counterfactual value computation
   - Regret accumulation and averaging

4. **Training**
   - Advantage regression (squared loss)
   - Strategy extraction (KL divergence or cross-entropy)
   - Value approximation
   - Gradient descent on experience batches

5. **Monitoring**
   - WandB integration with automatic logging
   - Iteration-wise loss tracking
   - Checkpoint version control
   - Emergency recovery

### `evaluate_strategy.py`

**Post-Training Analysis Tool**

Comprehensive evaluation of trained strategies:

```
Evaluation Pipeline:
├─ Load checkpoint
├─ Initialize networks & environment
├─ [1] Exploitability evaluation
│  ├─ Local Best Response (LBR) oracle
│  └─ Nash Distance (%)
├─ [2] Game play evaluation
│  ├─ Play 10K games
│  ├─ Collect payoff statistics
│  └─ Mean, std, median payoffs per player
└─ [3] Network statistics
   ├─ Parameter counts
   └─ Layer weight statistics
```

**Features:**
- ✅ LBR exploitability (Nash distance %)
- ✅ Game play statistics (mean payoffs, variance)
- ✅ Network architecture analysis
- ✅ JSON output for downstream analysis

**Usage:**

```bash
# Evaluate checkpoint
python scripts/evaluate_strategy.py --checkpoint checkpoints/vr_deep_pdcfr_iteration_010000.pt

# Custom output
python scripts/evaluate_strategy.py \
  --checkpoint checkpoints/vr_deep_pdcfr_iteration_010000.pt \
  --output my_results.json

# With custom game count
python scripts/evaluate_strategy.py \
  --checkpoint checkpoints/vr_deep_pdcfr_iteration_010000.pt \
  --num-games 50000

# CPU evaluation
python scripts/evaluate_strategy.py \
  --checkpoint checkpoints/vr_deep_pdcfr_iteration_010000.pt \
  --device cpu
```

**Output Format:**

```json
{
  "checkpoint": "checkpoints/vr_deep_pdcfr_iteration_010000.pt",
  "iteration": 10000,
  "timestamp": "1711881600.0",
  "exploitability": {
    "nash_distance_pct": 15.3,
    "oracle_mbb_hand": -0.00453
  },
  "game_play": {
    "player_0_mean_payoff": 0.0234,
    "player_0_std_payoff": 2.156,
    "player_0_median_payoff": 0.01234,
    ...
  },
  "network_stats": {
    "player_0_total_params": 1048576,
    "player_0_strategy_weight_mean": 0.0012,
    ...
  }
}
```

### `setup.sh`

**Environment Setup Script**

Automated installation and configuration:

```bash
bash scripts/setup.sh
```

**Tasks:**
1. ✅ Check Python 3 availability
2. ✅ Create virtual environment
3. ✅ Install PyTorch (CUDA 11.8)
4. ✅ Install RLCard and poker libraries
5. ✅ Install ML dependencies
6. ✅ Install training utilities (WandB, PyYAML)
7. ✅ Install dev tools (pytest, black, flake8)
8. ✅ Create directories (logs, checkpoints, data)
9. ✅ Run test suite

**Output:**
- Virtual environment: `venv/`
- Directories: `logs/`, `checkpoints/`, `data/`
- Test results: stdout

## Training Architecture

### Network Components (per player)

```
Input Observation (843 dims)
  ↓
Shared Hidden Layers (512 → 512 → ReLU)
  ├───→ Cumulative Advantage Head → A_S^t(s)
  ├───→ Instantaneous Advantage Head → A^t(s)
  ├───→ Value Head → V(s)
  └───→ Strategy Head (Logits) → σ(s)
```

### Training Loop Flow

```
Iteration t:
  1. Reset environment
  2. For each updating player p:
     a. Traverse with ES-MCCFR
     b. Accumulate advantage estimates
     c. Collect strategy samples
  3. Train all networks:
     a. Advantage networks (MSE loss)
     b. Value network (MSE loss)
     c. Strategy network (KL divergence loss)
  4. Empty buffers
  5. Log metrics
  6. Checkpoint and evaluate
```

### External Sampling MCCFR

```
traverse(state, reach_probs, updating_player):
  if terminal:
    return payoffs - baseline
  
  if not updating_player's turn:
    sample opponent action uniformly
  else:
    add to advantage buffer
    for each action:
      recursively traverse with updated reach probs
    
  aggregate counterfactual values
  return regret estimates
```

## Monitoring & Evaluation

### WandB Integration

Tracks in real-time:
- Loss curves (advantage, value, strategy)
- Iteration progress
- Checkpoint metadata
- Exploitability trends
- Game play statistics

Dashboard: https://wandb.ai/

### Checkpointing

Automatic saves every N iterations:
- Model weights (all 6 players)
- Optimizer states
- Iteration metadata
- Emergency recovery on SIGINT

Restore:
```python
checkpoint = torch.load("checkpoints/vr_deep_pdcfr_iteration_010000.pt")
iteration = checkpoint["iteration"]
```

### LBR Evaluation

Periodic oracle evaluation:
- Local Best Response computation
- Nash distance measurement
- Exploitability bounds
- Per-player win rates vs. oracle

## Troubleshooting

### Out of Memory

```python
# config.yaml
buffers:
  advantage_buffer_size: 2000000  # Reduce from 4M
  strategy_buffer_size: 2000000

networks:
  shared_architecture:
    hidden_dims: [256, 256]       # Reduce from [512, 512]
```

### Slow Training

```bash
# Use GPU
python scripts/train_6max_vr_deep.py --device cuda

# Reduce evaluation frequency
# config.yaml
cfr:
  exploitability_update_freq: 1000  # Less frequent
```

### Checkpoint Corruption

```bash
# Load emergency checkpoint
python scripts/evaluate_strategy.py \
  --checkpoint checkpoints/vr_deep_pdcfr_emergency_iter_005000.pt
```

## Performance Metrics

Expected timeline:

| Iteration | Time | Nash Distance | Oracle mbb/hand |
|-----------|------|---------------|-----------------|
| 1K        | ~10m | ~45%          | -2.5 mbb/hand   |
| 10K       | ~2h  | ~25%          | -0.8 mbb/hand   |
| 50K       | ~10h | ~5%           | -0.1 mbb/hand   |

*Times on NVIDIA A100 GPU*

## Advanced Usage

### Custom Training Config

```bash
# Create custom_config.yaml
python scripts/train_6max_vr_deep.py --config custom_config.yaml
```

### Distributed Training (Future)

```bash
# Multi-GPU training
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  scripts/train_6max_vr_deep.py
```

### Hyperparameter Sweep

```bash
# Use WandB Sweep
wandb sweep sweep_config.yaml
wandb agent <SWEEP_ID>
```

## References

- VR-DeepPDCFR+: Variance-Reduced Counterfactual Regret Minimization
- External Sampling MCCFR: Lanctot et al., 2009
- Deep CFR: Steinhauser et al., 2019
- RLCard: Zhu et al., 2020

## Support

For issues or questions:
1. Check logs (`logs/train_*.log`)
2. Review config against documentation
3. Open issue with checkpoint name and version info

---

**Last Updated:** March 31, 2026  
**Version:** 1.0
