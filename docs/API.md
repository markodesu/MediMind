# MediMind API Documentation

Complete API reference for the MediMind backend.

## Base URL

```
http://localhost:8000
```

## API Version

All endpoints are prefixed with `/api/v1`

---

## Endpoints

### 1. Root Endpoint

**GET** `/`

Get API information.

**Response:**
```json
{
  "message": "MediMind API - University of Central Asia",
  "description": "AI Health Guidance Chatbot for UCA students and staff",
  "version": "1.0.0"
}
```

---

### 2. Chat Endpoint

**POST** `/api/v1/chat`

Send a message to the MediMind chatbot and receive a response.

#### Request Body

```json
{
  "message": "What is a headache?",
  "history": [
    {
      "role": "user",
      "content": "I have been feeling unwell"
    },
    {
      "role": "assistant",
      "content": "I'm sorry to hear that. Can you tell me more about your symptoms?"
    }
  ]
}
```

**Fields:**
- `message` (string, required): The user's current message
- `history` (array, optional): Previous conversation messages
  - `role` (string): Either "user" or "assistant"
  - `content` (string): The message content

#### Response

**Success (200 OK):**
```json
{
  "answer": "A headache is a pain or discomfort in the head, scalp, or neck...",
  "confidence": 0.75,
  "safe": true
}
```

**Low Confidence Response:**
```json
{
  "answer": "I am not fully confident in this assessment. Please visit Dr. Kyal at the University of Central Asia (UCA).\n\nContact Information:\n📞 Phone: +996XXXXXXXXX\n📍 Location: 1st floor, Academic Block, near GYM\n\nDr. Kyal is available to provide professional medical consultation for UCA students and staff.",
  "confidence": 0.35,
  "safe": false
}
```

**Fields:**
- `answer` (string): The AI-generated response
- `confidence` (float): Confidence score between 0 and 1
- `safe` (boolean): Whether the response is considered safe (confidence >= threshold)

#### Example Requests

**cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is a headache?",
    "history": []
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "What is a headache?",
        "history": []
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Confidence: {data['confidence']}")
```

**JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/api/v1/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'What is a headache?',
    history: []
  })
});

const data = await response.json();
console.log(data.answer);
```

---

## Conversation History

The API supports conversation history to maintain context across multiple messages.

### Example Conversation Flow

**Message 1:**
```json
{
  "message": "I have a headache",
  "history": []
}
```

**Response 1:**
```json
{
  "answer": "Headaches can be caused by various factors...",
  "confidence": 0.72,
  "safe": true
}
```

**Message 2 (with history):**
```json
{
  "message": "It started this morning",
  "history": [
    {
      "role": "user",
      "content": "I have a headache"
    },
    {
      "role": "assistant",
      "content": "Headaches can be caused by various factors..."
    }
  ]
}
```

**Response 2:**
```json
{
  "answer": "Since it started this morning, you might want to monitor...",
  "confidence": 0.68,
  "safe": true
}
```

### Best Practices

1. **Limit History:** Send only the last 10-15 messages to avoid token limits
2. **Maintain Order:** Keep messages in chronological order
3. **Include Both Roles:** Include both user and assistant messages
4. **Clear on Reset:** Send empty array `[]` to start a new conversation

---

## Error Responses

### 400 Bad Request

Invalid request format or missing required fields.

```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 422 Unprocessable Entity

Validation error.

```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

### 500 Internal Server Error

Server error (model loading, generation failure, etc.).

```json
{
  "detail": "Internal server error"
}
```

---

## Configuration

### Environment Variables

Set these in `backend/.env`:

```env
# Model Configuration
MODEL_NAME=microsoft/phi-2
LORA_MODEL_PATH=./models/medimind-phi2-lora  # Optional: path to trained LoRA adapter
MAX_NEW_TOKENS=150

# Confidence Threshold
CONFIDENCE_THRESHOLD=0.3

# UCA Medical Contact
UCA_MEDICAL_CONTACT_NAME=Dr. Kyal
UCA_MEDICAL_PHONE=+996XXXXXXXXX  # Set your actual phone number here
UCA_MEDICAL_LOCATION=1st floor, Academic Block, near GYM
```

### Confidence Threshold

The confidence threshold is set to `0.3` (30%) to allow the model to showcase its capabilities. However, safety is maintained through multiple layers:

1. **Urgent cases** are redirected before model inference (via `should_redirect_to_doctor()`)
2. **Low confidence responses** are marked as `safe: false` but still returned
3. **Dangerous keywords** in user messages significantly lower confidence scores
4. **Model instructions** explicitly avoid diagnoses and prescriptions

The threshold primarily affects the `safe` flag in responses, not routing decisions (which are handled by urgent case detection).

---

## Rate Limiting

Currently, there are no rate limits. However, consider implementing rate limiting for production use.

---

## CORS

CORS is enabled for:
- `http://localhost:3000`
- `http://localhost:5173`

To add more origins, update `backend/app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://yourdomain.com"],
    ...
)
```

---

## WebSocket Support

WebSocket support is not currently implemented. All communication is via HTTP POST requests.

---

## Testing

### Using the Test Script

```bash
cd backend
python -m app.test_chat
```

### Manual Testing

```bash
# Test basic chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a fever?"}'

# Test with history
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me more",
    "history": [
      {"role": "user", "content": "I have a headache"},
      {"role": "assistant", "content": "Headaches can be..."}
    ]
  }'
```

---

## Response Times

Typical response times:

- **GPU (CUDA):** 1-3 seconds
- **CPU:** 3-10 seconds (depending on hardware)

First request may take longer due to model loading.

---

## Model Information

- **Base Model:** microsoft/phi-2
- **Fine-tuning:** LoRA (Low-Rank Adaptation)
- **Max Tokens:** 150 (configurable)
- **Context Length:** 512 tokens (configurable)

---

## Support

For API issues:
1. Check server logs
2. Verify model is loaded correctly
3. Check request format matches documentation
4. Ensure backend is running on correct port

---

**Last Updated:** 2024-11-30
**API Version:** 1.0.0

