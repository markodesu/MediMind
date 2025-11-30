from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm import generate_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    
    # Model call
    response, confidence = generate_response(user_message)
    
    # Safety check
    if confidence < 0.5:
        return {
            "answer": "I am not fully confident in this assessment. Please visit a hospital or contact a medical professional immediately.",
            "confidence": confidence
        }
    
    return {"answer": response, "confidence": confidence}
