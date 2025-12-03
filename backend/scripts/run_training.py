#!/usr/bin/env python3
"""
Simple training runner script for MediMind.
Checks prerequisites and runs training with sensible defaults.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("=" * 50)
    print("Checking Prerequisites...")
    print("=" * 50)
    
    errors = []
    
    # Check dataset
    dataset_path = backend_dir / "app" / "data" / "dataset.jsonl"
    if not dataset_path.exists():
        errors.append(f"❌ Dataset not found: {dataset_path}\n   Run: python3 -m app.data.prepare_dataset")
    else:
        with open(dataset_path, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"✅ Dataset found: {dataset_path}")
        print(f"   Entries: {line_count}")
    
    # Check dependencies
    dependencies = ['peft', 'bitsandbytes', 'accelerate', 'transformers', 'datasets', 'torch']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} installed")
        except ImportError:
            errors.append(f"❌ {dep} not installed\n   Run: pip install -r requirements.txt")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU available: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print("⚠️  No GPU detected - training will be VERY slow on CPU")
            print("   Recommended: Use local GPU with 8GB+ VRAM")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(0)
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
    
    if errors:
        print("\n" + "=" * 50)
        print("ERRORS FOUND:")
        print("=" * 50)
        for error in errors:
            print(error)
        sys.exit(1)
    
    print("\n✅ All prerequisites met!")
    print("=" * 50)
    return True


def run_training():
    """Run the training script."""
    import subprocess
    
    dataset_path = backend_dir / "app" / "data" / "dataset.jsonl"
    output_dir = backend_dir / "models" / "medimind-phi2-lora"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: 3")
    print(f"Batch size: 4")
    print(f"Learning rate: 2e-4")
    print("=" * 50 + "\n")
    
    # Run training
    cmd = [
        sys.executable,
        "-m", "app.training.train",
        "--dataset_path", str(dataset_path),
        "--output_dir", str(output_dir),
        "--num_epochs", "3",
        "--batch_size", "4",
        "--learning_rate", "2e-4",
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=str(backend_dir))
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    
    # Success message
    print("\n" + "=" * 50)
    print("Training Completed Successfully!")
    print("=" * 50)
    print(f"\n✅ Model saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Update backend/.env:")
    print(f"   LORA_MODEL_PATH={output_dir}")
    print("\n2. Or update backend/app/config.py:")
    print(f'   LORA_MODEL_PATH = "{output_dir}"')
    print("\n3. Restart your backend server")
    print("=" * 50)


if __name__ == "__main__":
    try:
        if check_prerequisites():
            run_training()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

