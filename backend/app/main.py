from fastapi import FastAPI
from model import generate_response
from schemas import ChatRequest

app = FastAPI()

@app.get("/")
def home():
    return {"message": "MediMind API is running!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    response, confidence = generate_response(request.question)
    return {
        "answer": response,
        "confidence": confidence,
        "safe": confidence > 0.6
    }
