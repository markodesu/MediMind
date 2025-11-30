# HuggingFace Training Guide for MediMind

## ✅ STEP 1 - Dataset Verification

**Status: COMPLETE ✓**

- **Location:** `backend/app/data/dataset.jsonl`
- **Entries:** 1,337
- **Format:** JSONL with `instruction` and `response` fields
- **File size:** 2.2 MB

**Dataset Format:**
```json
{
  "instruction": "What is Flu?",
  "response": "Flu is a respiratory infection...",
  "source": "mediqa_qa",
  "safety_level": "safe"
}
```

**Verification:**
```bash
cd backend/app
ls -lh data/dataset.jsonl
# Should show: dataset.jsonl (2.2M)
```

---

## ✅ STEP 2 - Upload to HuggingFace

### Create Dataset Repository

1. **Create HuggingFace Account** (if needed)
   - Go to: https://huggingface.co/join

2. **Create New Dataset**
   - Go to: https://huggingface.co/new-dataset
   - **Name:** `Medimind/medical_chatbot_dataset` (or your preferred name)
   - **Visibility:** Private (recommended for medical data)
   - Click "Create dataset"

3. **Upload dataset.jsonl**
   - Go to your dataset page
   - Click "Add file" → "Upload file"
   - Upload: `backend/app/data/dataset.jsonl`
   - Wait for upload to complete

**Note:** AutoTrain can also work with datasets from your local machine, but uploading to HF makes it easier.

---

## ✅ STEP 3 - Use AutoTrain (Free)

### Access AutoTrain

1. Go to: https://huggingface.co/autotrain
2. Click **"Create Project"**
3. Select **"LLM Finetuning"**

### Model Selection

**Recommended FREE models:**
- **Llama-3.2-1B-Instruct** ⭐ (fast, small, perfect for small teams)
- **Qwen2.5-1.5B-instruct** (good alternative)

### Dataset Configuration

1. **Select your dataset:**
   - Choose: `Medimind/medical_chatbot_dataset` (or upload directly)
   - Or: Upload `dataset.jsonl` directly

2. **Dataset format:**
   - AutoTrain should auto-detect: `instruction / response`
   - Verify it shows:
     - **Instruction column:** `instruction`
     - **Response column:** `response`

### Training Settings

**Recommended settings for best free results:**

| Setting | Value |
|---------|-------|
| **LoRA** | Enabled ✓ |
| **r** | 8 |
| **α (alpha)** | 16 |
| **Epochs** | 3 |
| **Batch size** | 4 |
| **Learning rate** | 2e-4 |
| **Max seq length** | 512 |
| **Optimizer** | AdamW |
| **Scheduler** | cosine |

**Note:** Default settings also work fine if you want to start quickly.

### Start Training

1. Review all settings
2. Click **"Start Training"**
3. AutoTrain will:
   - Queue your job (free tier may have wait time)
   - Train the model
   - Show progress logs
   - Notify you when complete

**Training time:** Typically 30 minutes to 2 hours depending on queue and model size.

---

## ✅ STEP 4 - After Training Completes

### Model Repository

You'll receive a model repository like:
```
your-username/medimind-medical-chatbot-1b
```

**Contents:**
- `adapter_config.json` - LoRA adapter configuration
- `adapter_model.safetensors` - LoRA weights
- `README.md` - Model card
- Training logs

### Inference Code

AutoTrain provides inference code. Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Llama-3.2-1B-Instruct"
adapter = "your-username/medimind-medical-chatbot-1b"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter)

# Inference
prompt = "What should I do if I have mild food poisoning?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## ✅ STEP 5 - Integration with Backend

Once you have your trained model, provide me with:

1. **Model repository name** (e.g., `your-username/medimind-medical-chatbot-1b`)
2. **Base model name** (e.g., `meta-llama/Llama-3.2-1B-Instruct`)

I will then create:

- ✅ `backend/app/services/finetuned_llm.py` - Model loader and inference
- ✅ Updated `backend/app/routers/chat.py` - FastAPI route integration
- ✅ Safety layer - Confidence scoring + disclaimers
- ✅ Streaming response support (optional)
- ✅ Configuration updates

---

## Current Dataset Status

**Location:** `backend/app/data/dataset.jsonl`
**Size:** 1,337 entries (2.2 MB)
**Format:** ✓ Correct (instruction/response)
**Ready for:** ✓ HuggingFace AutoTrain

---

## Quick Start Commands

```bash
# Verify dataset
cd backend/app
ls -lh data/dataset.jsonl
wc -l data/dataset.jsonl  # Should show 1337

# View sample entry
head -1 data/dataset.jsonl | python3 -m json.tool

# Ready to upload to HuggingFace!
```

---

## Notes

- Your dataset includes safety disclaimers in all responses
- Only prescription dosages are filtered (as per your requirements)
- Confidence scoring will handle safety regulation
- Model will learn general medical knowledge, not diagnosis

**Next:** Upload to HuggingFace and start AutoTrain training!

