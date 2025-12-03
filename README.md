# MediMind - AI Health Guidance Chatbot for UCA

**University of Central Asia (UCA) - Medical Guidance System**

MediMind is an AI-powered health guidance chatbot designed specifically for UCA students and staff. When the system has low confidence in its response, it directs users to UCA's medical services.

## 🤖 Intelligent Features

MediMind implements **3 intelligent behaviors**:

1. **✅ Language Understanding / Generation** - Natural language processing with phi-2 model
2. **✅ Decision-Making** - Confidence-based routing to human professionals
3. **✅ Prediction / Recommendation** - Medical advice generation with confidence scoring

**📖 [See Detailed Documentation](docs/INTELLIGENT_FEATURES.md)** - Complete explanation with code evidence

## 📚 Documentation

Complete documentation is available in the `docs/` directory:

- **[Intelligent Features](docs/INTELLIGENT_FEATURES.md)** - Detailed AI behavior documentation ⭐
- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Google Colab Training](docs/COLAB_QUICK_START.md)** - Train on free GPU (Recommended) 🆕
- **[Training Guide](docs/TRAINING.md)** - Complete guide for fine-tuning the model
- **[API Documentation](docs/API.md)** - Backend API endpoints and usage
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture and design

### Quick Links

- 🤖 [Intelligent Features](docs/INTELLIGENT_FEATURES.md) - AI behaviors and code evidence
- 🚀 [Quick Start](docs/QUICK_START.md) - Installation and setup
- ☁️ [Google Colab Training](docs/COLAB_QUICK_START.md) - Free GPU training (2-4 hours) ⭐
- 📚 [Training Guide](docs/TRAINING.md) - Complete fine-tuning documentation
- 📡 [API Reference](docs/API.md) - API endpoints and examples
- 🏗️ [Architecture](docs/ARCHITECTURE.md) - System design and components

An AI-powered health guidance chatbot built with FastAPI and React.

## Project Structure

```
MediMind/
├── backend/                    # FastAPI backend application
│   ├── app/                    # Main application package
│   │   ├── model/              # Model loading and inference logic
│   │   │   ├── __init__.py
│   │   │   └── inference.py   # Model inference functions
│   │   ├── routers/            # FastAPI route handlers
│   │   │   ├── __init__.py
│   │   │   └── chat.py        # Chat endpoint routes
│   │   ├── schemas/            # Pydantic request/response models
│   │   │   ├── __init__.py
│   │   │   └── chat.py        # Chat-related schemas
│   │   ├── utils/              # Helper utilities
│   │   │   ├── __init__.py
│   │   │   ├── preprocess.py  # Text preprocessing functions
│   │   │   └── confidence.py # Confidence calculation utilities
│   │   ├── knowledge_base/     # Knowledge base for symptom advice
│   │   │   ├── symptoms.json  # Symptom-to-advice mappings
│   │   │   └── utils.py       # Knowledge base query functions
│   │   ├── config.py           # Configuration (env variables)
│   │   └── main.py             # FastAPI application entry point
│   ├── tests/                  # Unit and integration tests
│   ├── requirements.txt        # Python dependencies
│   └── venv/                   # Virtual environment (gitignored)
├── frontend/                   # Vite + React UI
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.jsx
│   └── index.html
├── models/                     # AI model weights and configs
│   └── (model files - use Git LFS if storing locally)
├── docs/                       # Documentation
│   ├── (proposal, report, slides, API spec, architecture diagrams)
└── README.md                   # This file
```

## Folder Descriptions

### `backend/app/model/`
Contains all model-loading and inference logic. The `inference.py` module handles loading the transformer model and generating responses to user questions.

### `backend/app/routers/`
FastAPI route handlers organized by feature. Each router module defines API endpoints for a specific domain (e.g., chat, health checks).

### `backend/app/schemas/`
Pydantic models for request and response validation. Schemas are split into separate files by domain (e.g., `chat.py` for chat-related schemas).

### `backend/app/utils/`
Helper functions for common operations:
- **preprocess.py**: Text preprocessing for better model performance
- **confidence.py**: Confidence score calculation for model responses

### `backend/app/knowledge_base/`
Stores symptom-to-advice mappings in JSON format. The `utils.py` module provides functions to query and retrieve advice based on symptoms.

### `backend/app/config.py`
Centralized configuration management using environment variables. Loads settings from `.env` file using `python-dotenv`.

### `backend/tests/`
Directory for unit tests and integration tests (to be populated).

### `docs/`
Project documentation including design documents, API specifications, and architecture diagrams.

### `models/`
Storage for AI model weights and configuration files. Use Git LFS for large model files if storing locally.

## Setup

1. **Backend Setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file in the `backend/` directory:
   ```env
   MODEL_NAME=google/flan-t5-small
   MAX_NEW_TOKENS=60
   CONFIDENCE_THRESHOLD=0.6
   API_TITLE=MediMind API
   API_VERSION=1.0.0
   ```

3. **Run the Backend:**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

4. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/v1/` - Health check
- `POST /api/v1/chat` - Chat endpoint for medical questions

See `/docs` for interactive API documentation when the server is running.

## Verification

To verify all intelligent features are working:

```bash
cd backend
# Make sure backend is running first
python -m app.verify_intelligent_features
```

This will test:
- ✅ Language Understanding/Generation
- ✅ Decision-Making
- ✅ Prediction/Recommendation
