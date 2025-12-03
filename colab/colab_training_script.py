"""
MediMind Training Script for Google Colab
Copy this into Colab cells and run step by step.

Step-by-step instructions:
1. Enable GPU: Runtime → Change runtime type → GPU
2. Run each cell in order
3. Upload your dataset.jsonl when prompted
4. Wait for training (2-4 hours)
5. Download the trained model
"""

# ============================================================================
# CELL 1: Check GPU
# ============================================================================
"""
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ No GPU! Enable GPU in Runtime → Change runtime type")
"""

# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================
"""
!pip install -q transformers peft bitsandbytes accelerate datasets torch
"""

# ============================================================================
# CELL 3: Upload Dataset
# ============================================================================
"""
from google.colab import files
import os

# Create directories
os.makedirs("MediMind/backend/app/data", exist_ok=True)

print("📤 Please upload your dataset.jsonl file...")
uploaded = files.upload()

# Move uploaded file
for filename in uploaded.keys():
    if 'dataset.jsonl' in filename:
        os.rename(filename, "MediMind/backend/app/data/dataset.jsonl")
        print(f"✅ Dataset uploaded!")

# Verify
with open("MediMind/backend/app/data/dataset.jsonl", 'r') as f:
    line_count = sum(1 for _ in f)
print(f"✅ Dataset verified: {line_count} entries")
"""

# ============================================================================
# CELL 4: Create Training Script
# ============================================================================
"""
# This creates the complete training script
# (Full script content is in backend/app/training/train.py)
# We'll use the existing script structure
"""

# ============================================================================
# CELL 5: Clone or Copy Training Code
# ============================================================================
"""
# Option A: If you have the repo on GitHub, clone it:
# !git clone https://github.com/YOUR_USERNAME/MediMind.git

# Option B: Copy the training script directly (we'll do this in the next cells)
"""

# ============================================================================
# CELL 6-10: Main Training Code (Copy from backend/app/training/train.py)
# ============================================================================
"""
# The complete training code will be executed here
# See docs/COLAB_SETUP.md for full instructions
"""

