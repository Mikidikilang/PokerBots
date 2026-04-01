#!/usr/bin/env bash
# Phase A: Heads-Up Nash Convergence Smoke Test - Quick Start

set -e  # Exit on any error

echo "=========================================================================="
echo "PHASE A: HEADS-UP NASH CONVERGENCE SMOKE TEST"
echo "=========================================================================="
echo ""
echo "This script runs a 2-player heads-up test for 10,000 iterations to validate"
echo "that the bug fixes result in proper convergence to Nash equilibrium."
echo ""
echo "Prerequisites:"
echo "  - Python 3.9 or higher"
echo "  - CUDA 11.8+ (optional, CPU fallback available)"
echo "  - ~8-12 GB GPU memory or ~16 GB RAM (CPU)"
echo "  - Dependencies: torch, yaml, rlcard"
echo ""
echo "=========================================================================="
echo ""

# Verify Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python version: $PYTHON_VERSION"

# Check if config file exists
if [ ! -f "config_heads_up_smoke.yaml" ]; then
    echo "❌ ERROR: config_heads_up_smoke.yaml not found in current directory."
    echo "Make sure you're in the poker_ai_v6 root directory."
    exit 1
fi
echo "✓ Config file found: config_heads_up_smoke.yaml"

# Check if runner script exists
if [ ! -f "scripts/run_heads_up_phase_a.py" ]; then
    echo "❌ ERROR: scripts/run_heads_up_phase_a.py not found."
    exit 1
fi
echo "✓ Runner script found: scripts/run_heads_up_phase_a.py"

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import torch; print(f'✓ torch version: {torch.__version__}')" || {
    echo "❌ Missing torch. Install with: pip install torch"
    exit 1
}
python3 -c "import yaml; print('✓ yaml available')" || {
    echo "❌ Missing pyyaml. Install with: pip install pyyaml"
    exit 1
}

# Create necessary directories
mkdir -p checkpoints/heads_up_phase_a
mkdir -p logs
echo "✓ Created checkpoint and log directories"

# Check GPU availability
echo ""
echo "Checking device availability..."
GPU_AVAILABLE=$(python3 -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')")
if [ "$GPU_AVAILABLE" == "YES" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✓ GPU detected: $GPU_NAME"
    DEVICE="cuda"
else
    echo "⚠ GPU not detected. Will use CPU (slower, but available)"
    DEVICE="cpu"
fi

# Print run summary
echo ""
echo "=========================================================================="
echo "SMOKE TEST CONFIGURATION"
echo "=========================================================================="
echo "Players:                 2 (Heads-Up)"
echo "Iterations:              10,000"
echo "Traversals/iteration:    100"
echo "Total game states:       1,000,000"
echo "Batch size:              4096"
echo "Network epochs:          4"
echo "DCFR parameters:         α=1.5, β=0, γ=2.0"
echo "Evaluation interval:     Every 500 iterations"
echo "Expected runtime:        12-24 hours (GPU) / 2-3 days (CPU)"
echo "Device:                  $DEVICE"
echo "=========================================================================="
echo ""

# Confirmation
echo "Starting Phase A Heads-Up Nash Convergence Test..."
echo "(Press Ctrl+C to cancel, or any key to continue)"
read -t 5 || true
echo ""

# Run the smoke test
echo "Running: python3 scripts/run_heads_up_phase_a.py --config config_heads_up_smoke.yaml"
echo ""

python3 scripts/run_heads_up_phase_a.py --config config_heads_up_smoke.yaml

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SMOKE TEST COMPLETED SUCCESSFULLY"
    echo "=========================================================================="
    echo ""
    echo "Check the following files for results:"
    echo "  - logs/heads_up_phase_a_*.log    (detailed training logs)"
    echo "  - checkpoints/heads_up_phase_a/  (trained network checkpoints)"
    echo ""
    echo "Expected outcome:"
    echo "  - Exploitability < 1.0 mBB/hand by iteration 5,000"
    echo "  - Exploitability ≈ 0.3-0.5 mBB/hand by iteration 10,000"
    echo "  - All 8 bug fixes validated"
    echo ""
    echo "Next steps:"
    echo "  1. Review PHASE_A_SMOKE_TEST.md for detailed analysis"
    echo "  2. Run Phase B (6-player smoke test) if successful"
    echo "  3. Deploy to GPU cluster for production 6-Max training"
else
    echo "❌ SMOKE TEST FAILED (Exit code: $EXIT_CODE)"
    echo "=========================================================================="
    echo ""
    echo "Check logs for errors:"
    echo "  - tail -f logs/heads_up_phase_a_*.log"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: Reduce batch_size or network hidden_dims in config"
    echo "  - Module not found: Ensure all dependencies are installed"
    echo "  - CUDA errors: Check torch/CUDA compatibility"
fi

exit $EXIT_CODE
