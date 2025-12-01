# Backend Endpoints Discovery

This document describes the backend API endpoints discovered from the backend codebase.

## Base URL

The backend API is expected to run on `http://localhost:8000` by default. This can be overridden using the `VITE_API_URL` environment variable.

## Endpoints

### POST /chat

Sends a chat message to the MediMind AI assistant.

**Request:**
```json
{
  "question": "string"
}
```

**Response:**
```json
{
  "answer": "string",
  "confidence": 0.85,
  "safe": true
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "I have a headache"}'
```

**Response Fields:**
- `answer` (string): The AI-generated response to the user's question
- `confidence` (number): A confidence score between 0 and 1 indicating the model's confidence in the response
- `safe` (boolean): Indicates whether the response is considered safe (confidence > 0.6)

## Implementation Notes

- The frontend uses the `/chat` endpoint to send user messages
- The API expects a JSON body with a `question` field
- The response includes the answer, confidence score, and safety flag
- Error handling is implemented in the frontend API layer (`src/lib/api.ts`)

## Assumptions

1. The backend runs on port 8000 by default
2. CORS is properly configured on the backend to allow frontend requests
3. The backend uses FastAPI (as seen in `backend/app/main.py`)
4. No authentication is required for the `/chat` endpoint (based on current implementation)

## Future Considerations

If the backend API changes, update this document and the corresponding TypeScript types in `src/lib/types.ts` and API implementation in `src/lib/api.ts`.

