from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.chat import router as chat_router
from app.services.llm import load_model

app = FastAPI(
    title="MediMind API - University of Central Asia",
    description="AI Health Guidance Chatbot for UCA students and staff",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Load model when server starts (not on first request)."""
    print("Starting MediMind backend...")
    load_model()
    print("Backend ready!")

# Add a root endpoint
@app.get("/")
async def root():
    return {
        "message": "MediMind API - University of Central Asia",
        "description": "AI Health Guidance Chatbot for UCA students and staff",
        "version": "1.0.0"
    }
