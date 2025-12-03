# 🚀 Quick Start: Train on Google Colab (5 Minutes Setup)

The fastest way to train your MediMind model using Google Colab's free GPU.

---

## 📋 Prerequisites

✅ Your dataset: `backend/app/data/dataset.jsonl` (1,337 entries)  
✅ Google account (for Colab access)  
✅ 2-4 hours for training

---

## 🎯 Step-by-Step (Copy & Paste)

### Step 1: Open Google Colab

1. Go to: **https://colab.research.google.com**
2. Click: **File → New notebook**
3. Name it: "MediMind Training"

### Step 2: Enable GPU ⚠️ IMPORTANT

1. Click **Runtime → Change runtime type**
2. Select **GPU** (T4)
3. Click **Save**
4. **Do this BEFORE running any code!**

### Step 3: Install Dependencies

Copy and run this in a Colab cell:

```python
!pip install -q transformers peft bitsandbytes accelerate datasets torch
```

Wait 2-3 minutes for installation.

### Step 4: Upload Dataset

Copy and run this in a new cell:

```python
from google.colab import files
import os

# Create directories
os.makedirs("MediMind/backend/app/data", exist_ok=True)
os.makedirs("MediMind/backend/app/training", exist_ok=True)
os.makedirs("MediMind/backend/models", exist_ok=True)

print("📤 Please upload your dataset.jsonl file...")
uploaded = files.upload()

# Move uploaded file
for filename in uploaded.keys():
    if 'dataset.jsonl' in filename.lower():
        os.rename(filename, "MediMind/backend/app/data/dataset.jsonl")
        print(f"✅ Dataset uploaded!")

# Verify
with open("MediMind/backend/app/data/dataset.jsonl", 'r') as f:
    line_count = sum(1 for _ in f)
print(f"✅ Dataset verified: {line_count} entries")
```

**Then click "Choose Files"** and upload your `backend/app/data/dataset.jsonl`.

### Step 5: Copy Training Script

**Download your training script** from your repo and upload it to Colab, OR copy the script content directly:

Run this to download from your repo:

```python
# Option 1: If your repo is on GitHub
!git clone https://github.com/YOUR_USERNAME/MediMind.git

# OR Option 2: Upload train.py directly using files.upload()
```

Or **copy the entire training script** into a new cell (see full script below).

### Step 6: Run Training

Copy and run this (this will take 2-4 hours):

```python
import sys
sys.path.insert(0, 'MediMind/backend')

from app.training.train import train

# Run training
train(
    dataset_path="MediMind/backend/app/data/dataset.jsonl",
    output_dir="MediMind/backend/models/medimind-phi2-lora",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
)
```

### Step 7: Download Model

After training completes (2-4 hours), run this:

```python
from google.colab import files
import shutil

# Create zip file
model_dir = "MediMind/backend/models/medimind-phi2-lora"
zip_path = "medimind-phi2-lora.zip"

shutil.make_archive('medimind-phi2-lora', 'zip', model_dir)
print("📥 Downloading model...")
files.download(zip_path)
print("✅ Download complete!")
```

---

## 📦 Complete Training Script

If you need to create the training script in Colab, copy this entire block into a cell:

```python
%%writefile MediMind/backend/app/training/train.py
# (Copy the entire content of backend/app/training/train.py here)
# See your local file at: backend/app/training/train.py
```

Or **download it from your repo** if it's on GitHub.

---

## ✅ After Downloading

1. **Extract the zip**: `unzip medimind-phi2-lora.zip`
2. **Move to your project**: 
   ```bash
   mv medimind-phi2-lora /home/student/projects/MediMind/backend/models/
   ```
3. **Update config**: Add to `backend/.env`:
   ```env
   LORA_MODEL_PATH=./models/medimind-phi2-lora
   ```
4. **Restart backend**: 
   ```bash
   cd backend && python3 -m uvicorn app.main:app --reload
   ```

---

## ⚡ All-in-One Script (Alternative)

For an even easier experience, you can use the prepared notebook:

1. **Upload notebook file** (if you have it):
   - Go to Colab
   - File → Upload notebook
   - Upload: `colab/MediMind_Training.ipynb` (if available)

2. **Or follow the step-by-step above**

---

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU detected | Runtime → Change runtime type → GPU |
| Out of memory | Reduce `batch_size` to 2 |
| Dataset not found | Check file path: `MediMind/backend/app/data/dataset.jsonl` |
| Module not found | Re-run: `!pip install -q transformers peft bitsandbytes accelerate datasets torch` |

---

## 📚 More Help

- **Detailed guide**: `docs/COLAB_SETUP.md` - Complete walkthrough
- **Training docs**: `docs/TRAINING.md` - Full documentation
- **Quick reference**: `docs/QUICK_START_TRAINING.md`

---

**That's it! Your model will be trained in 2-4 hours.** 🚀

