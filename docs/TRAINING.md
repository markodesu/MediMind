# MediMind Model Training Documentation

Complete guide for fine-tuning the phi-2 model on medical QA data using LoRA (Low-Rank Adaptation).

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Training Options](#training-options)
4. [Step-by-Step Training Guide](#step-by-step-training-guide)
5. [Configuration](#configuration)
6. [Testing Trained Models](#testing-trained-models)
7. [Integration with Backend](#integration-with-backend)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Overview

MediMind uses **LoRA (Low-Rank Adaptation)** to fine-tune the `microsoft/phi-2` model on medical question-answer pairs. LoRA is a parameter-efficient fine-tuning method that:

- **Reduces memory usage** by 70-90% compared to full fine-tuning
- **Trains faster** with fewer parameters
- **Maintains model quality** while being more efficient
- **Allows easy model switching** without retraining the base model

### Architecture

```
Base Model (phi-2)
    ↓
LoRA Adapter (trained on medical data)
    ↓
Fine-tuned MediMind Model
```

---

## Prerequisites

### 1. Dataset

Ensure your dataset is prepared:

```bash
cd backend
python -m app.data.prepare_dataset
```

This creates `backend/app/data/dataset.jsonl` with format:
```json
{"instruction": "What is a headache?", "response": "A headache is...", "source": "mediqa_qa", "safety_level": "safe"}
```

### 2. Dependencies

Install training dependencies:

```bash
pip install peft bitsandbytes accelerate transformers datasets torch
```

Or install from requirements:

```bash
pip install -r backend/requirements.txt
```

### 3. Hardware Requirements

| Setup | VRAM | Training Time (3 epochs) |
|-------|------|---------------------------|
| **4-bit Quantization** (recommended) | 6-8 GB | ~2-4 hours |
| **8-bit Quantization** | 10-12 GB | ~1.5-3 hours |
| **Full Precision** | 16+ GB | ~1-2 hours |
| **CPU** (not recommended) | N/A | 24+ hours |

**Recommendation:** Use a GPU with at least 8GB VRAM for local training.

---

## Training Options

### Option 1: Local Training (GPU Required)

Best for: Users with local GPU access

**Advantages:**
- Full control over training process
- No time limits
- Can resume from checkpoints

**Disadvantages:**
- Requires GPU hardware
- Higher electricity costs

### Option 2: Local GPU Training (Alternative Setup)

Best for: Users with local GPU (8GB+ VRAM)

**Advantages:**
- Full control over training process
- No session time limits
- More reliable than cloud solutions
- Faster training with direct GPU access

**Requirements:**
- GPU with 8GB+ VRAM (NVIDIA recommended)
- CUDA installed
- Sufficient disk space for model checkpoints

**Note:** Colab is not recommended due to instability and frequent crashes during training.

### Option 3: HuggingFace AutoTrain

Best for: Users who want the easiest setup

**Advantages:**
- No code required
- Web-based interface
- Automatic hyperparameter tuning
- Free tier available

**Disadvantages:**
- Less control over training
- Requires dataset on HuggingFace Hub

---

## Step-by-Step Training Guide

### Option 1: Local Training

#### Step 1: Verify Dataset

```bash
cd backend
head -2 app/data/dataset.jsonl
```

Should show JSONL entries with `instruction` and `response` fields.

#### Step 2: Install Dependencies

```bash
pip install peft bitsandbytes accelerate
```

#### Step 3: Run Training

```bash
python -m app.training.train \
    --dataset_path app/data/dataset.jsonl \
    --output_dir ./models/medimind-phi2-lora \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

#### Step 4: Monitor Training

Training will show:
- Loss values (should decrease)
- Evaluation metrics
- Checkpoint saves every 100 steps

**Expected Output:**
```
MediMind Fine-tuning with LoRA
==================================================
Loading model: microsoft/phi-2
Loading dataset from app/data/dataset.jsonl...
Loaded 1234 examples
Preprocessing dataset...
Train examples: 1110
Validation examples: 124
Starting training...
Epoch 1/3: 100%|████████| 278/278 [15:23<00:00, loss=2.45]
Epoch 2/3: 100%|████████| 278/278 [15:18<00:00, loss=1.89]
Epoch 3/3: 100%|████████| 278/278 [15:21<00:00, loss=1.34]
Training complete!
Model saved to: ./models/medimind-phi2-lora
```

---

### Option 2: Local GPU Training (Alternative Method)

#### Step 1: Verify GPU Access

```bash
nvidia-smi  # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

#### Step 2: Install Dependencies

```bash
pip install transformers peft bitsandbytes accelerate datasets torch
```

#### Step 3: Prepare Training Environment

Ensure you have:
- `backend/app/training/train.py`
- `backend/app/data/dataset.jsonl`
- Sufficient disk space (at least 10GB for checkpoints)

#### Step 4: Run Training

```bash
cd backend
python -m app.training.train \
    --dataset_path app/data/dataset.jsonl \
    --output_dir ./models/medimind-phi2-lora \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

#### Step 5: Verify Trained Model

After training completes, the model will be saved to:
```
./models/medimind-phi2-lora/
```

You can test it by updating your `.env` file:
```env
LORA_MODEL_PATH=./models/medimind-phi2-lora
```

---

### Option 3: HuggingFace AutoTrain

#### Step 1: Verify Dataset on HuggingFace

Your dataset should be at: `markodesu/Medimind`

If not uploaded:
```bash
cd backend/app/data
huggingface-cli upload markodesu/Medimind dataset.jsonl --repo-type dataset
```

#### Step 2: Create AutoTrain Project

1. Go to https://huggingface.co/autotrain
2. Click **"Create Project"**
3. Select **"LLM Fine-tuning"**

#### Step 3: Configure Project

**Model Selection:**
- Base Model: `microsoft/phi-2`
- Task: **Causal Language Modeling**

**Dataset:**
- Dataset: `markodesu/Medimind`
- Text Column: `instruction`
- Target Column: `response`

**Training Settings:**
- Training Type: **LoRA**
- LoRA r: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.1`
- Epochs: `3`
- Batch size: `4`
- Learning rate: `2e-4`
- Max sequence length: `512`

#### Step 4: Start Training

Click **"Start Training"** and wait for completion (typically 2-4 hours).

#### Step 5: Download Model

After training:
1. Go to your AutoTrain project
2. Click on the trained model
3. Download or use directly from HuggingFace Hub

---

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | 3 | Number of training passes |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 2e-4 | Learning rate for optimizer |
| `--max_length` | 512 | Maximum sequence length |
| `--save_steps` | 100 | Save checkpoint every N steps |
| `--no_4bit` | False | Disable 4-bit quantization |

### LoRA Configuration

LoRA settings are in `train.py`:

```python
lora_config = LoraConfig(
    r=8,              # LoRA rank (higher = more parameters)
    lora_alpha=16,    # Scaling factor
    lora_dropout=0.1, # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
)
```

**Adjusting LoRA Rank:**
- `r=4`: Smaller adapter, faster training, less capacity
- `r=8`: Balanced (recommended)
- `r=16`: Larger adapter, more capacity, slower training

---

## Testing Trained Models

### Quick Test

```bash
python -m app.training.load_trained_model \
    --model_path ./models/medimind-phi2-lora \
    --test_prompt "Human: What is a headache?\nAssistant:"
```

### Expected Output

```
Loading base model: microsoft/phi-2
Loading LoRA adapter from: ./models/medimind-phi2-lora

Testing with prompt: Human: What is a headache?
Assistant:
Response: A headache is a pain or discomfort in the head, scalp, or neck...
```

### Interactive Testing

Create a test script:

```python
from app.training.load_trained_model import load_trained_model, test_model

model, tokenizer = load_trained_model(
    "microsoft/phi-2",
    "./models/medimind-phi2-lora"
)

# Test multiple prompts
prompts = [
    "Human: What is a fever?\nAssistant:",
    "Human: How do I treat a cold?\nAssistant:",
    "Human: When should I see a doctor?\nAssistant:",
]

for prompt in prompts:
    test_model(model, tokenizer, prompt)
```

---

## Integration with Backend

### Step 1: Set Model Path

**Option A: Environment Variable (.env)**

Create/update `backend/.env`:

```env
LORA_MODEL_PATH=./models/medimind-phi2-lora
```

**Option B: Config File**

Update `backend/app/config.py`:

```python
LORA_MODEL_PATH: Optional[str] = "./models/medimind-phi2-lora"
```

### Step 2: Verify Model Loading

Start the backend:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Look for:

```
Loading model...
Loading LoRA adapter from: ./models/medimind-phi2-lora
✅ LoRA adapter loaded successfully!
Model loaded on GPU
Model loaded successfully!
```

### Step 3: Test API

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a headache?", "history": []}'
```

### Step 4: Compare Results

**Before Training:**
- Generic responses
- Lower medical accuracy
- May include irrelevant information

**After Training:**
- Medical-specific responses
- Higher accuracy on medical questions
- Better context understanding
- Improved confidence scores

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch_size 2`
2. Reduce max length: `--max_length 256`
3. Enable 4-bit quantization (default)
4. Use gradient accumulation: Already set to 4

### Issue: Training Too Slow

**Symptoms:**
- Training takes 10+ hours
- Very slow per-epoch time

**Solutions:**
1. **Use GPU:** Training on CPU is extremely slow
2. **Reduce dataset size:** Use subset for testing
3. **Reduce max_length:** `--max_length 256`
4. **Use Colab:** Free GPU access

### Issue: Model Not Loading

**Symptoms:**
```
⚠️ Error loading LoRA adapter: ...
Falling back to base model...
```

**Solutions:**
1. Check path: Ensure `LORA_MODEL_PATH` is correct
2. Install PEFT: `pip install peft`
3. Check model files: Ensure adapter files exist
4. Verify format: Model should be LoRA adapter, not merged

### Issue: Poor Training Results

**Symptoms:**
- Loss not decreasing
- Model gives irrelevant responses

**Solutions:**
1. **Check dataset quality:** Ensure dataset is properly formatted
2. **Increase epochs:** `--num_epochs 5`
3. **Adjust learning rate:** Try `1e-4` or `3e-4`
4. **Increase LoRA rank:** Change `r=8` to `r=16`
5. **Check data preprocessing:** Verify format_prompt function

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'peft'
```

**Solutions:**
```bash
pip install peft bitsandbytes accelerate
```

---

## Advanced Topics

### Merging LoRA Weights

For faster inference (no PEFT dependency):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./models/medimind-phi2-lora")

# Merge weights
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./models/medimind-phi2-merged")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.save_pretrained("./models/medimind-phi2-merged")
```

Then update `config.py`:
```python
MODEL_NAME = "./models/medimind-phi2-merged"
LORA_MODEL_PATH = None
```

### Hyperparameter Tuning

Experiment with different settings:

```bash
# Lower learning rate (more stable)
python -m app.training.train --learning_rate 1e-4

# Higher LoRA rank (more capacity)
# Edit train.py: r=16, lora_alpha=32

# More epochs (better convergence)
python -m app.training.train --num_epochs 5
```

### Resume Training

To resume from checkpoint:

```python
# In train.py, add:
training_args = TrainingArguments(
    ...
    resume_from_checkpoint="./models/medimind-phi2-lora/checkpoint-200",
)
```

### Custom Dataset Format

If your dataset has different format, modify `preprocess_function` in `train.py`:

```python
def preprocess_function(examples, tokenizer, max_length: int = 512):
    # Custom formatting
    prompts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(examples["question"], examples["answer"])
    ]
    # ... rest of function
```

---

## Best Practices

1. **Always validate dataset** before training
2. **Start with small epochs** (1-2) to test setup
3. **Monitor loss curves** - should decrease steadily
4. **Save checkpoints** regularly (default: every 100 steps)
5. **Test on validation set** to avoid overfitting
6. **Use GPU** for reasonable training times
7. **Keep base model** separate from LoRA adapter
8. **Document training parameters** for reproducibility

---

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Phi-2 Model Card](https://huggingface.co/microsoft/phi-2)
- [Transformers Training Guide](https://huggingface.co/docs/transformers/training)

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review training logs for errors
3. Verify dataset format matches expected structure
4. Check GPU/CPU compatibility

---

**Last Updated:** 2024-11-30
**Version:** 1.0.0

