# MediMind Model Training

This directory contains scripts for fine-tuning the phi-2 model on the medical QA dataset using LoRA (Low-Rank Adaptation).

> 📚 **For complete documentation, see [docs/TRAINING.md](../../docs/TRAINING.md)**

## Prerequisites

1. **Install training dependencies:**
   ```bash
   pip install peft bitsandbytes accelerate
   ```

2. **GPU Recommended:** Training on CPU is very slow. Use a GPU (CUDA) for reasonable training times.

3. **Dataset:** Ensure `backend/app/data/dataset.jsonl` exists (run `prepare_dataset.py` first).

## Training Options

### Option 1: Local Training (with GPU)

```bash
cd backend
python -m app.training.train \
    --dataset_path app/data/dataset.jsonl \
    --output_dir ./models/medimind-phi2-lora \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### Option 2: Google Colab

1. Upload the training script and dataset to Colab
2. Install dependencies:
   ```python
   !pip install transformers peft bitsandbytes accelerate datasets
   ```
3. Run training with GPU runtime enabled

### Option 3: HuggingFace AutoTrain (Easiest)

1. Upload your dataset to HuggingFace Hub (already done: `markodesu/Medimind`)
2. Go to https://huggingface.co/autotrain
3. Create a new project:
   - **Model:** `microsoft/phi-2`
   - **Task:** LLM Fine-tuning
   - **Dataset:** `markodesu/Medimind`
   - **Training Type:** LoRA
   - **Settings:**
     - LoRA r: 8
     - LoRA alpha: 16
     - Epochs: 3
     - Batch size: 4
     - Learning rate: 2e-4
4. Start training (free tier available)

## Training Parameters

- **LoRA Rank (r):** 8 - Controls adapter size (higher = more parameters)
- **LoRA Alpha:** 16 - Scaling factor for LoRA weights
- **Epochs:** 3 - Number of training passes
- **Batch Size:** 4 - Adjust based on GPU memory
- **Learning Rate:** 2e-4 - Standard for LoRA fine-tuning
- **Max Length:** 512 - Maximum sequence length

## Memory Requirements

- **4-bit Quantization (default):** ~6-8 GB VRAM
- **8-bit Quantization:** ~10-12 GB VRAM
- **Full Precision:** ~16+ GB VRAM (not recommended)

## After Training

1. **Test the model:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel
   
   base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
   model = PeftModel.from_pretrained(base_model, "./models/medimind-phi2-lora")
   tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
   ```

2. **Update backend to use trained model:**
   - Update `app/services/llm.py` to load the LoRA adapter
   - Or merge LoRA weights into base model for faster inference

## Merging LoRA Weights (Optional)

To merge LoRA weights into the base model for faster inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
model = PeftModel.from_pretrained(base_model, "./models/medimind-phi2-lora")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/medimind-phi2-merged")
```

Then update `app/services/llm.py` to load from `./models/medimind-phi2-merged`.

