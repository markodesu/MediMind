# MediMind Architecture Documentation

Complete architecture overview of the MediMind system.

## System Overview

MediMind is a medical guidance chatbot system built with:
- **Backend:** FastAPI (Python)
- **Frontend:** React + TypeScript + Vite
- **AI Model:** microsoft/phi-2 (fine-tuned with LoRA)
- **Deployment:** Local development, production-ready

```
┌─────────────┐
│   Frontend  │ (React + TypeScript)
│  (Port 5173) │
└──────┬──────┘
       │ HTTP/REST
       │
┌──────▼──────┐
│   Backend   │ (FastAPI)
│  (Port 8000) │
└──────┬──────┘
       │
┌──────▼──────┐
│  LLM Model │ (phi-2 + LoRA)
│  (GPU/CPU)  │
└─────────────┘
```

---

## Project Structure

```
MediMind/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── config.py           # Configuration settings
│   │   ├── main.py             # FastAPI application entry
│   │   ├── routers/            # API route handlers
│   │   │   └── chat.py         # Chat endpoint
│   │   ├── schemas/            # Pydantic models
│   │   │   └── chat.py         # Request/response schemas
│   │   ├── services/           # Business logic
│   │   │   └── llm.py          # LLM inference service
│   │   ├── training/           # Model training scripts
│   │   │   ├── train.py        # LoRA fine-tuning script
│   │   │   └── load_trained_model.py
│   │   ├── data/               # Dataset preparation
│   │   │   └── prepare_dataset.py
│   │   ├── utils/              # Helper functions
│   │   │   ├── preprocess.py   # Text preprocessing
│   │   │   └── confidence.py  # Confidence calculation
│   │   └── knowledge_base/    # Medical knowledge base
│   │       └── symptoms.json
│   ├── models/                 # Trained model storage
│   │   └── medimind-phi2-lora/ # LoRA adapter
│   ├── tests/                  # Unit tests
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── lib/                # Utilities
│   │   │   ├── api.ts          # API client
│   │   │   └── types.ts        # TypeScript types
│   │   └── pages/              # Page components
│   └── package.json
│
└── docs/                       # Documentation
    ├── TRAINING.md
    ├── API.md
    └── ARCHITECTURE.md
```

---

## Backend Architecture

### 1. Application Layer (`main.py`)

- **FastAPI Application:** Main application instance
- **CORS Middleware:** Cross-origin resource sharing
- **Router Registration:** Includes chat router
- **Startup Events:** Model loading on server start

### 2. Router Layer (`routers/chat.py`)

- **Endpoint:** `POST /api/v1/chat`
- **Request Validation:** Pydantic schema validation
- **Business Logic:** Calls LLM service
- **Response Formatting:** Returns structured response
- **Safety Checks:** Low confidence → UCA medical contact

### 3. Service Layer (`services/llm.py`)

- **Model Loading:** Lazy loading of phi-2 model
- **LoRA Support:** Loads fine-tuned adapter if available
- **Inference:** Text generation with conversation history
- **Confidence Calculation:** Heuristic-based confidence scoring

### 4. Schema Layer (`schemas/chat.py`)

- **ChatRequest:** Validates incoming requests
- **ChatResponse:** Structures API responses
- **MessageHistory:** Conversation history format

### 5. Configuration (`config.py`)

- **Settings Management:** Pydantic settings with .env support
- **Model Configuration:** Model name, LoRA path, tokens
- **UCA Settings:** Medical contact information
- **Thresholds:** Confidence threshold configuration

---

## Frontend Architecture

### 1. Component Structure

```
ChatPage (Main Container)
├── Header
├── ChatWindow
│   ├── MessageList
│   │   └── MessageBubble (per message)
│   └── Composer (input)
└── (State Management)
```

### 2. State Management

- **React Hooks:** useState, useCallback
- **Message History:** Maintained in component state
- **Loading States:** UI feedback during API calls

### 3. API Integration (`lib/api.ts`)

- **HTTP Client:** Fetch API
- **Request Formatting:** Converts to backend format
- **Error Handling:** Network and API errors
- **Type Safety:** TypeScript interfaces

---

## Data Flow

### Chat Request Flow

```
1. User types message in frontend
   ↓
2. Frontend formats request with history
   ↓
3. POST /api/v1/chat
   ↓
4. Router validates request (Pydantic)
   ↓
5. Service formats conversation prompt
   ↓
6. LLM generates response
   ↓
7. Confidence calculation
   ↓
8. Safety check (confidence < threshold?)
   ↓
9. Response formatting
   ↓
10. Return JSON to frontend
   ↓
11. Frontend displays message
```

### Model Loading Flow

```
1. Server startup
   ↓
2. load_model() called
   ↓
3. Load base model (phi-2)
   ↓
4. Check LORA_MODEL_PATH
   ↓
5. If exists: Load LoRA adapter
   ↓
6. Move to GPU (if available)
   ↓
7. Set model.eval()
   ↓
8. Ready for inference
```

---

## Model Architecture

### Base Model: microsoft/phi-2

- **Type:** Causal Language Model (Decoder-only)
- **Parameters:** 2.7B
- **Context Length:** 2048 tokens
- **Architecture:** Transformer-based

### Fine-tuning: LoRA (Low-Rank Adaptation)

- **Method:** Parameter-efficient fine-tuning
- **Rank (r):** 8
- **Alpha:** 16
- **Target Modules:** Attention layers (q_proj, k_proj, v_proj, dense)
- **Memory Reduction:** ~70-90% vs full fine-tuning

### Inference

- **Format:** `Human: {message}\nAssistant:`
- **History:** Previous messages included in prompt
- **Generation:** Greedy decoding (do_sample=False)
- **Max Tokens:** 150 (configurable)

---

## Configuration Management

### Environment Variables

All configuration via `.env` file:

```env
# Model
MODEL_NAME=microsoft/phi-2
LORA_MODEL_PATH=./models/medimind-phi2-lora
MAX_NEW_TOKENS=150

# Safety
CONFIDENCE_THRESHOLD=0.5

# UCA
UCA_MEDICAL_CONTACT_NAME=Dr. Kyal
UCA_MEDICAL_PHONE=+996708136013
UCA_MEDICAL_LOCATION=1st floor, Academic Block, near GYM
```

### Pydantic Settings

- **Type Safety:** Validated at startup
- **Defaults:** Sensible defaults if not set
- **Environment:** Loads from .env automatically

---

## Security Considerations

### 1. Input Validation

- **Pydantic Schemas:** Automatic validation
- **Length Limits:** Max message length enforced
- **Type Checking:** Type-safe request handling

### 2. Safety Mechanisms

- **Confidence Threshold:** Low confidence → redirect to professional
- **Content Filtering:** Dataset filtered for safety
- **Disclaimers:** Responses include safety disclaimers

### 3. CORS

- **Restricted Origins:** Only allowed origins
- **Credentials:** Controlled credential sharing

---

## Performance Optimization

### 1. Model Loading

- **Lazy Loading:** Model loaded once at startup
- **GPU Support:** Automatic GPU detection and usage
- **Quantization:** 4-bit quantization for memory efficiency

### 2. Inference

- **Batch Processing:** Single requests (can be extended)
- **Greedy Decoding:** Faster than sampling
- **Context Management:** Limited history to avoid token limits

### 3. Caching

- **Model Caching:** Model stays in memory
- **Response Caching:** Not implemented (can be added)

---

## Error Handling

### Backend Errors

1. **Model Loading Errors:** Fallback to base model
2. **Generation Errors:** Return error message
3. **Validation Errors:** Pydantic validation messages
4. **Network Errors:** Proper HTTP status codes

### Frontend Errors

1. **API Errors:** User-friendly error messages
2. **Network Errors:** Connection failure handling
3. **Timeout Handling:** Request timeout management

---

## Deployment Architecture

### Development

```
Frontend (Vite Dev Server) → Backend (Uvicorn) → Model (Local)
```

### Production (Recommended)

```
┌─────────────┐
│   Nginx     │ (Reverse Proxy)
└──────┬──────┘
       │
┌──────▼──────┐
│   Backend   │ (Gunicorn + Uvicorn Workers)
└──────┬──────┘
       │
┌──────▼──────┐
│  GPU Server │ (Model Inference)
└─────────────┘
```

### Containerization (Future)

- **Docker:** Containerize backend and frontend
- **Docker Compose:** Orchestrate services
- **GPU Support:** NVIDIA Docker runtime

---

## Monitoring and Logging

### Current Logging

- **Startup Logs:** Model loading status
- **Error Logs:** Exception handling
- **API Logs:** Request/response (via FastAPI)

### Future Enhancements

- **Structured Logging:** JSON logs
- **Metrics:** Response times, confidence scores
- **Health Checks:** `/health` endpoint
- **Monitoring:** Prometheus + Grafana

---

## Scalability Considerations

### Current Limitations

- **Single Instance:** One backend instance
- **In-Memory Model:** Model loaded per instance
- **No Load Balancing:** Single server

### Scaling Options

1. **Horizontal Scaling:** Multiple backend instances
2. **Model Server:** Separate model inference service
3. **Caching Layer:** Redis for common queries
4. **Queue System:** Celery for async processing

---

## Technology Stack

### Backend

- **Framework:** FastAPI 0.104+
- **Python:** 3.8+
- **ML Framework:** PyTorch, Transformers
- **Fine-tuning:** PEFT (LoRA)
- **Validation:** Pydantic

### Frontend

- **Framework:** React 18+
- **Language:** TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **HTTP Client:** Fetch API

### AI/ML

- **Base Model:** microsoft/phi-2
- **Fine-tuning:** LoRA (PEFT)
- **Quantization:** BitsAndBytes (4-bit)
- **Inference:** Transformers library

---

## Future Enhancements

1. **Multi-turn Conversations:** Enhanced context management
2. **Streaming Responses:** Real-time response generation
3. **User Authentication:** User-specific conversations
4. **Analytics Dashboard:** Usage statistics
5. **Multi-language Support:** Internationalization
6. **Voice Interface:** Speech-to-text integration
7. **Mobile App:** React Native application

---

**Last Updated:** 2024-11-30
**Version:** 1.0.0

