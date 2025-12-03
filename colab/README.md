# Google Colab Training for MediMind

This folder contains resources for training MediMind on Google Colab.

---

## 🚀 Quick Start

**Choose your method:**

### Option 1: Step-by-Step Guide (Recommended)

Follow the **complete guide**: [`../docs/COLAB_QUICK_START.md`](../docs/COLAB_QUICK_START.md)

**What you'll do:**
1. Open Google Colab
2. Enable GPU
3. Copy & paste code cells
4. Upload your dataset
5. Run training (2-4 hours)
6. Download the model

**Time to setup**: 5 minutes

---

### Option 2: Detailed Walkthrough

For more detailed instructions, see: [`../docs/COLAB_SETUP.md`](../docs/COLAB_SETUP.md)

---

## 📋 Files in This Folder

- `README.md` - This file
- `colab_training_script.py` - Reference script (not directly runnable)
- (Future) `MediMind_Training.ipynb` - Pre-made Colab notebook

---

## ⚡ What You Need

✅ Your dataset: `backend/app/data/dataset.jsonl`  
✅ Google account  
✅ 2-4 hours for training  

---

## 🎯 Steps Summary

1. **Open Colab**: https://colab.research.google.com
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Install packages**: `!pip install transformers peft bitsandbytes accelerate datasets torch`
4. **Upload dataset**: Upload `backend/app/data/dataset.jsonl`
5. **Run training**: Copy training code and run
6. **Download model**: Download the zip file when done

---

## 📚 Documentation

- **Quick Start**: [`../docs/COLAB_QUICK_START.md`](../docs/COLAB_QUICK_START.md) ⭐ Start here!
- **Detailed Guide**: [`../docs/COLAB_SETUP.md`](../docs/COLAB_SETUP.md)
- **Full Training Docs**: [`../docs/TRAINING.md`](../docs/TRAINING.md)

---

## 💡 Tips

- ✅ Enable GPU **before** running any code
- ✅ Training takes 2-4 hours - be patient!
- ✅ Colab sessions may timeout after 12 hours of inactivity
- ✅ The model zip file will be ~100-200 MB

---

**Ready to train?** Go to [`../docs/COLAB_QUICK_START.md`](../docs/COLAB_QUICK_START.md) 🚀

