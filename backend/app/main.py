from fastapi import FastAPI
from app.routers.chat import router as chat_router

app = FastAPI()

app.include_router(chat_router, prefix="/api/v1")

# Add a root endpoint
@app.get("/")
async def root():
    return {"message": "MediMind backend is running!"}
