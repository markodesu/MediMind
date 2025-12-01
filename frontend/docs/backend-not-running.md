# Backend Not Running

## Issue Detected

The backend server is **not currently running** on `http://localhost:8000`.

## Diagnosis

- **Date**: $(Get-Date)
- **Port Check**: Port 8000 is not accessible
- **Status**: Backend server needs to be started

## How to Start Backend

To start the backend server, navigate to the `backend/` directory and run:

```bash
# If using Python with uvicorn
cd backend
uvicorn app.main:app --reload --port 8000

# Or if using a different setup, check backend/README.md
```

## Frontend Behavior

The frontend will:
- ✅ Start successfully on `http://localhost:3000`
- ⚠️ Show error messages when trying to send chat messages
- ⚠️ Display: "Sorry, I encountered an error. Please try again later."

## Next Steps

1. Start the backend server on port 8000
2. Verify backend is accessible: `curl http://localhost:8000/` should return `{"message":"MediMind API is running!"}`
3. Refresh the frontend and try sending a message

## CORS Configuration

If you encounter CORS errors after starting the backend, the backend needs to allow requests from `http://localhost:3000`. See `frontend/docs/cors-warning.md` for details.

