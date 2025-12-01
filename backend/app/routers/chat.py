from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm import generate_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

# UCA-specific medical contact information
UCA_MEDICAL_CONTACT = """Please visit Dr. Kyal at the University of Central Asia (UCA).

Contact Information:
📞 Phone: +996708136013
📍 Location: 1st floor, Academic Block, near GYM

Dr. Kyal is available to provide professional medical consultation for UCA students and staff."""


@router.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    
    # Model call
    response, confidence = generate_response(user_message)
    
    # Safety check - low confidence redirects to UCA medical services
    if confidence < 0.5:
        return {
            "answer": f"I am not fully confident in this assessment. {UCA_MEDICAL_CONTACT}",
            "confidence": confidence,
            "safe": False
        }
    
    return {
        "answer": response,
        "confidence": confidence,
        "safe": confidence >= 0.5
    }
