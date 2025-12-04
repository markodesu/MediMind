## MediMind - AI Health Guidance Chatbot for UCA

**University of Central Asia (UCA) - Medical Guidance System**

MediMind is an AI-powered health guidance chatbot designed for UCA students and staff. The system uses a multi-layered safety approach: urgent cases are automatically redirected to UCA's medical services, while general health questions are handled by the AI with confidence scoring and safety checks.

### 🤖 Intelligent Features

MediMind implements **3 intelligent behaviors**:

1. **✅ Language Understanding / Generation** – Natural language processing with the `microsoft/phi-2` model (with optional LoRA fine-tuning), supporting context-aware conversations
2. **✅ Decision-Making** – Multi-layered safety system: urgent case detection, confidence-based routing, and safety flags for low-confidence responses
3. **✅ Prediction / Recommendation** – Medical advice suggestions with multi-factor confidence scoring (response quality, medical relevance, danger detection)

**📖 [Detailed Intelligent Features](docs/INTELLIGENT_FEATURES.md)** – Full explanation with code evidence.

### 📚 Documentation

All documentation is in the `docs/` directory:

- **[Quick Start Guide](docs/QUICK_START.md)** – Get up and running in minutes  
- **[Training Guide](docs/TRAINING.md)** – Full fine-tuning guide  
- **[API Documentation](docs/API.md)** – Backend API endpoints and usage  
- **[Architecture](docs/ARCHITECTURE.md)** – System architecture and design

An AI-powered health guidance chatbot built with **FastAPI** (backend) and **React + Vite + TypeScript** (frontend).

## Project Structure

```text
MediMind/
├── backend/                      # FastAPI backend application
│   ├── app/                      # Main application package
│   │   ├── model/                # Model loading and inference logic
│   │   │   ├── __init__.py
│   │   │   └── inference.py      # Model inference functions
│   │   ├── routers/              # FastAPI route handlers
│   │   │   ├── __init__.py
│   │   │   └── chat.py           # Chat endpoint routes
│   │   ├── schemas/              # Pydantic request/response models
│   │   │   ├── __init__.py
│   │   │   └── chat.py           # Chat-related schemas
│   │   ├── services/             # LLM and business logic services
│   │   │   └── llm.py            # LLM inference / orchestration
│   │   ├── utils/                # Helper utilities
│   │   │   ├── __init__.py
│   │   │   ├── preprocess.py     # Text preprocessing functions
│   │   │   └── confidence.py     # Confidence calculation utilities
│   │   ├── knowledge_base/       # Knowledge base for symptom advice
│   │   │   ├── symptoms.json     # Symptom-to-advice mappings
│   │   │   └── utils.py          # Knowledge base query functions
│   │   ├── data/                 # Dataset preparation scripts and data
│   │   │   ├── dataset.jsonl
│   │   │   └── prepare_dataset.py
│   │   ├── training/             # Training and fine-tuning code
│   │   │   ├── train.py
│   │   │   └── load_trained_model.py
│   │   ├── verify_intelligent_features.py
│   │   ├── config.py             # Configuration (env variables)
│   │   └── main.py               # FastAPI application entry point
│   ├── scripts/                  # Helper scripts (e.g. training runner)
│   │   └── run_training.py
│   ├── tests/                    # Unit and integration tests
│   ├── requirements.txt          # Python dependencies
│   └── venv/                     # Virtual environment (gitignored)
├── frontend/                     # Vite + React + TypeScript UI
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   ├── pages/                # Page-level views (e.g. ChatPage)
│   │   ├── contexts/             # React context providers (e.g. theme)
│   │   ├── lib/                  # API client and shared types
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── index.html
├── models/                       # AI model weights and configs
│   └── (model files – use Git LFS if storing locally)
├── docs/                         # Documentation
│   └── (reports, slides, API spec, architecture diagrams, etc.)
└── README.md                     # This file
```

## Folder Descriptions

### `backend/app/model/`
Model-loading and inference logic. The `inference.py` module loads the transformer model and generates responses to user questions.

### `backend/app/services/`
Higher-level services for working with the LLM and application logic. `llm.py` orchestrates prompts, safety checks, and confidence handling.

### `backend/app/routers/`
FastAPI route handlers, organized by feature. Each router module defines API endpoints for a specific domain (e.g., chat, health checks).

### `backend/app/schemas/`
Pydantic models for request and response validation. Schemas are split into separate files by domain (e.g., `chat.py` for chat-related schemas).

### `backend/app/utils/`
Helper functions for common operations:
- **`preprocess.py`** – Text preprocessing for better model performance  
- **`confidence.py`** – Confidence score calculation for model responses  

### `backend/app/knowledge_base/`
Stores symptom-to-advice mappings in JSON format. `utils.py` provides functions to query and retrieve advice based on symptoms.

### `backend/app/data/`
Dataset preparation scripts (`prepare_dataset.py`) and processed datasets (`dataset.jsonl`) used for training/fine-tuning.

### `backend/app/training/`
Training utilities and scripts. Includes `train.py` for fine-tuning and `load_trained_model.py` for loading trained adapters/checkpoints.

### `backend/tests/`
Directory for unit and integration tests.

### `docs/`
Project documentation including design documents, API specifications, and architecture diagrams.

### `models/`
Storage for AI model weights and configuration files. Use Git LFS for large model files if storing locally.

## Setup

1. **Backend Setup**

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**

   Create a `.env` file in the `backend/` directory:

   ```env
   # Base model
   MODEL_NAME=microsoft/phi-2
   CONFIDENCE_THRESHOLD=0.3  # Lowered to showcase model capabilities while maintaining safety

   # Optional LoRA fine-tuned model
   # LORA_MODEL_PATH=./models/medimind-phi2-lora

   # UCA medical contact details used when routing to human care
   UCA_MEDICAL_CONTACT_NAME=Dr. Kyal
   UCA_MEDICAL_PHONE=+996XXXXXXXXX  # Set your actual phone number here (not committed to Git)
   UCA_MEDICAL_LOCATION=1st floor, Academic Block, near GYM

   # API metadata (optional)
   API_TITLE=MediMind API
   API_VERSION=1.0.0
   ```

3. **Run the Backend**

   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

4. **Frontend Setup and Run**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## API Endpoints

- **`GET /`** – Root endpoint  
- **`GET /api/v1/`** – Health check  
- **`POST /api/v1/chat`** – Chat endpoint for medical questions  

Interactive API docs are available at **`/docs`** when the backend server is running.

## Verification of Intelligent Features

To verify that all intelligent features are working end-to-end:

```bash
cd backend
# Make sure the backend is running first
python -m app.verify_intelligent_features
```

This script verifies:
- ✅ Language Understanding / Generation  
- ✅ Decision-Making (confidence routing and urgent case detection)  
- ✅ Prediction / Recommendation

## Safety Features

MediMind implements a **multi-layered safety approach**:

1. **Pre-filtering**: Urgent keywords (severe, emergency, chest pain, bleeding, etc.) trigger immediate doctor referral before model inference
2. **Model Safety Instructions**: System prompts explicitly instruct the model to avoid diagnoses and prescriptions
3. **Confidence Scoring**: Multi-factor algorithm that penalizes dangerous keywords and rewards medical relevance
4. **Safety Flags**: All responses include a `safe` boolean field based on confidence threshold (0.3)
5. **Post-processing**: Removes explicit dosages and prescription language

The confidence threshold of 0.3 allows the model to showcase its capabilities while maintaining safety through these multiple layers.  
