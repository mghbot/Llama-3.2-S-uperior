#!/bin/bash
# Quick start script for Llama 3.2 1B Expert Augmentation

set -e

echo "=================================================="
echo "Llama 3.2 1B Expert Augmentation - Quick Start"
echo "=================================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install -r requirements.txt

# Test imports
echo ""
echo "[3/5] Testing imports..."
python3 test_imports.py

# Check HuggingFace authentication
echo ""
echo "[4/5] Checking HuggingFace authentication..."
if huggingface-cli whoami &> /dev/null; then
    echo "✓ Already logged in to HuggingFace"
    huggingface-cli whoami
else
    echo "✗ Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    echo "Or provide token via --auth-token flag"
fi

# Show next steps
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Train a model:"
echo "     python main.py train --dataset wikitext --dataset-config wikitext-2-raw-v1"
echo ""
echo "  2. Benchmark a model:"
echo "     python main.py benchmark --model-path ./llama-3.2-1b-expert"
echo ""
echo "  3. Run inference:"
echo "     python main.py inference --model-path ./llama-3.2-1b-expert"
echo ""
echo "For more information, see README.md"
echo "=================================================="
