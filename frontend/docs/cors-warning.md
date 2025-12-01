# CORS Configuration Warning

## Issue

If you see CORS (Cross-Origin Resource Sharing) errors in the browser console when the frontend tries to connect to the backend, the backend needs to be configured to allow requests from the frontend origin.

## Error Example

```
Access to fetch at 'http://localhost:8000/chat' from origin 'http://localhost:3000' 
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present 
on the requested resource.
```

## Solution (Backend Configuration Required)

The backend FastAPI application needs to include CORS middleware. This should be added to `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Alternative: Using Vite Proxy

The frontend is configured with a Vite proxy in `vite.config.ts` that proxies `/chat` requests to the backend. However, the frontend API client (`src/lib/api.ts`) uses the full URL from `VITE_API_URL`, so the proxy may not be used.

To use the Vite proxy instead:
- Change `src/lib/api.ts` to use relative URLs: `fetch('/chat', ...)`
- Or ensure CORS is configured on the backend (recommended)

## Current Configuration

- **Frontend URL**: `http://localhost:3000`
- **Backend URL**: `http://localhost:8000` (from `VITE_API_URL`)
- **API Endpoint**: `POST http://localhost:8000/chat`

## Testing CORS

After configuring CORS on the backend:

1. Start the backend server
2. Start the frontend dev server
3. Open browser DevTools → Network tab
4. Send a message from the frontend
5. Check the response headers for `Access-Control-Allow-Origin: http://localhost:3000`

