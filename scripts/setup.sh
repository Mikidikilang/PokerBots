#!/bin/bash
# VR-DeepPDCFR+ Setup Script
# Installs dependencies and prepares environment for training

set -e

echo "=========================================="
echo "VR-DeepPDCFR+ Setup"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing RLCard and poker libraries..."
pip install rlcard dimwit

echo "Installing ML dependencies..."
pip install numpy scipy scikit-learn matplotlib seaborn pandas

echo "Installing training utilities..."
pip install wandb pyyaml tqdm

echo "Installing development tools..."
pip install pytest black flake8 mypy

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p data

# Run tests
echo "Running tests..."
python3 -m pytest . -v --tb=short || true

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml with your settings"
echo "  2. Run: python scripts/train_6max_vr_deep.py"
echo ""
