# MediMind Quick Start Guide

Get up and running with MediMind in 5 minutes.

## Prerequisites

- Python 3.8+
- Node.js 16+
- GPU (optional, but recommended for training)

## Installation

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

### 3. Environment Configuration

Create `backend/.env`:

```env
MODEL_NAME=microsoft/phi-2
CONFIDENCE_THRESHOLD=0.3
UCA_MEDICAL_CONTACT_NAME=Dr. Kyal
UCA_MEDICAL_PHONE=+996XXXXXXXXX  # Set your actual phone number here
UCA_MEDICAL_LOCATION=1st floor, Academic Block, near GYM
```

## Running the Application

### Start Backend

```bash
cd backend
uvicorn app.main:app --reload
```

Backend runs on: `http://localhost:8000`

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs on: `http://localhost:5173`

## Testing

### Test API

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a headache?"}'
```

### Test with Python

```bash
cd backend
python -m app.test_chat
```

## Training (Optional)

### Quick Training (Local GPU)

```bash
cd backend
pip install peft bitsandbytes accelerate
python -m app.training.train --output_dir ./models/medimind-phi2-lora
```

### Use Trained Model

Add to `backend/.env`:

```env
LORA_MODEL_PATH=./models/medimind-phi2-lora
```

Restart backend to load the trained model.

## Next Steps

- 📖 Read [API Documentation](API.md)
- 🏗️ Understand [Architecture](ARCHITECTURE.md)
- 🎓 Learn [Training Guide](TRAINING.md)

## Troubleshooting

### Backend won't start

- Check Python version: `python --version` (need 3.8+)
- Verify dependencies: `pip install -r requirements.txt`
- Check port 8000 is available

### Frontend won't connect

- Verify backend is running on port 8000
- Check CORS settings in `backend/app/main.py`
- Verify `VITE_API_URL` in frontend `.env`

### Model loading errors

- Check `MODEL_NAME` in `.env`
- Verify internet connection (model downloads from HuggingFace)
- For LoRA: Check `LORA_MODEL_PATH` exists

## Common Commands

```bash
# Backend
cd backend && uvicorn app.main:app --reload

# Frontend
cd frontend && npm run dev

# Test API
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# Prepare dataset
cd backend && python -m app.data.prepare_dataset

# Train model
cd backend && python -m app.training.train
```

---

**Need help?** Check the full documentation in the `docs/` directory.

