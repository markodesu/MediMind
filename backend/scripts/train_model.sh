#!/bin/bash
# Simple training script for MediMind
# This script checks prerequisites and runs training

set -e  # Exit on error

echo "=========================================="
echo "MediMind Model Training Script"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$BACKEND_DIR"

# Check dataset exists
DATASET_PATH="app/data/dataset.jsonl"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
    echo "   Please run: python3 -m app.data.prepare_dataset"
    exit 1
fi

echo "✅ Dataset found: $DATASET_PATH"
ENTRY_COUNT=$(wc -l < "$DATASET_PATH" | xargs)
echo "   Entries: $ENTRY_COUNT"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import peft" 2>/dev/null || {
    echo "❌ ERROR: 'peft' not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
}

python3 -c "import bitsandbytes" 2>/dev/null || {
    echo "❌ ERROR: 'bitsandbytes' not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
}

python3 -c "import accelerate" 2>/dev/null || {
    echo "❌ ERROR: 'accelerate' not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
}

echo "✅ All dependencies installed"
echo ""

# Check GPU
echo "Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No GPU detected - training will be VERY slow on CPU")
    print("   Recommended: Use Google Colab for free GPU access")
    response = input("Continue anyway? (y/N): ")
    if response.lower() != 'y':
        exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Training cancelled."
    exit 1
fi

echo ""

# Set default output directory
OUTPUT_DIR="${1:-./models/medimind-phi2-lora}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

python3 -m app.training.train \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update backend/.env:"
echo "   LORA_MODEL_PATH=$OUTPUT_DIR"
echo ""
echo "2. Or update backend/app/config.py:"
echo "   LORA_MODEL_PATH = \"$OUTPUT_DIR\""
echo ""
echo "3. Restart your backend server"
echo ""

