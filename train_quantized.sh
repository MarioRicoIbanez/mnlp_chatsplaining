#!/bin/bash
set -e

echo "Creating and activating Conda environment: mnlp_m2"

# Ensure conda command is available
source ~/.bashrc  # Or ~/.zshrc depending on your shell

# Create environment if it doesn't already exist
if ! conda info --envs | grep -q "^mnlp_m2 "; then
    conda create -y -n mnlp_m2 python=3.12.8
fi

conda activate mnlp_m2

echo "Installing dependencies from requirements.txt..."

pip install --upgrade pip
pip install -r requirements.txt

echo "Running quantization script..."

python train_quantized/quantize_llmcompressor.py

echo "Quantization complete."
