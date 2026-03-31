#!/bin/bash
# ==============================================================================
# VR-DeepPDCFR+ Production Launch Script
# ==============================================================================
# This script launches the production training run on a GPU cluster with:
# - nohup isolation for SSH disconnection resilience
# - Graceful shutdown signal handling
# - Comprehensive logging to logs/production_master.log
# - Background PID output for monitoring
#
# Usage:
#   bash scripts/launch_production.sh
#
# To monitor the training:
#   tail -f logs/production_master.log
#   ps aux | grep train_6max_vr_deep
#
# To gracefully shut down:
#   kill -SIGTERM <PID>

# Set strict error handling
set -euo pipefail

# Define project root (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
PRODUCTION_LOG="$LOG_DIR/production_master.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# Activate Python virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating Python virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating Python virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "WARNING: No virtual environment found. Using system Python."
fi

# Set working directory to project root
cd "$PROJECT_ROOT"

# Export environment variables for production
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="0"  # Adjust to your GPU setup

# ==============================================================================
# VALIDATION CHECKS
# ==============================================================================

echo "VR-DeepPDCFR+ Production Launch — Validation Phase"
echo "=================================================="

# Check that config_production.yaml exists
if [ ! -f "$PROJECT_ROOT/config_production.yaml" ]; then
    echo "ERROR: config_production.yaml not found in $PROJECT_ROOT"
    echo "Please ensure config_production.yaml exists before launching."
    exit 1
fi
echo "✓ config_production.yaml found"

# Check that training script exists
if [ ! -f "$PROJECT_ROOT/scripts/train_6max_vr_deep.py" ]; then
    echo "ERROR: scripts/train_6max_vr_deep.py not found"
    exit 1
fi
echo "✓ Training script found"

# Check that equity_cache exists (from Priority #6)
if [ ! -d "$PROJECT_ROOT/equity_cache" ]; then
    echo "WARNING: equity_cache directory not found. RCE lookups may be slow."
    echo "Ensure Priority #6 (equity precomputation) has been completed."
else
    CACHE_SIZE=$(du -sh "$PROJECT_ROOT/equity_cache" | cut -f1)
    echo "✓ equity_cache found (size: $CACHE_SIZE)"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
else
    echo "WARNING: NVIDIA GPU not detected. Training will use CPU (very slow)."
fi

# ==============================================================================
# LAUNCH PRODUCTION TRAINING
# ==============================================================================

echo ""
echo "Starting production training run..."
echo "=================================================="
echo "Configuration:  config_production.yaml"
echo "Log file:       $PRODUCTION_LOG"
echo "PID file:       (see below)"
echo "Time:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""

# Launch with nohup for SSH resilience
nohup python -m scripts.train_6max_vr_deep \
    --config config_production.yaml \
    > "$PRODUCTION_LOG" 2>&1 &

TRAINING_PID=$!

# Wait a moment for the process to start and ensure it didn't immediately crash
sleep 2

if ! kill -0 $TRAINING_PID 2>/dev/null; then
    echo "ERROR: Training process failed to start (PID $TRAINING_PID)"
    echo "Check logs: $PRODUCTION_LOG"
    exit 1
fi

# ==============================================================================
# POST-LAUNCH STATUS
# ==============================================================================

echo "✓ Training process started successfully!"
echo ""
echo "PRODUCTION TRAINING LAUNCHED"
echo "=================================================="
echo "PID:                    $TRAINING_PID"
echo "Log file:               $PRODUCTION_LOG"
echo "Configuration:          config_production.yaml"
echo "Start time:             $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""
echo "MONITORING COMMANDS:"
echo "  # View live logs:"
echo "    tail -f $PRODUCTION_LOG"
echo ""
echo "  # Check process status:"
echo "    ps aux | grep -E 'PID|$TRAINING_PID' | grep -v grep"
echo ""
echo "  # View GPU stats:"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "SHUTDOWN COMMANDS:"
echo "  # Graceful shutdown (SIGTERM):"
echo "    kill -TERM $TRAINING_PID"
echo ""
echo "  # Force shutdown (SIGKILL):"
echo "    kill -KILL $TRAINING_PID"
echo ""

# Write PID to file for easy reference
echo "$TRAINING_PID" > "$LOG_DIR/production.pid"
echo "PID saved to: $LOG_DIR/production.pid"

exit 0
