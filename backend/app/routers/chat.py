from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm import generate_response
from app.config import settings

router = APIRouter()

# UCA-specific medical contact information
UCA_MEDICAL_CONTACT = """Please visit Dr. Kyal at the University of Central Asia (UCA).

Contact Information:
📞 Phone: +996708136013
📍 Location: 1st floor, Academic Block, near GYM

Dr. Kyal is available to provide professional medical consultation for UCA students and staff."""


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_message = request.message
    history = request.history or []  # Get conversation history
    
    # Model call with history
    response, confidence = generate_response(user_message, history)
    
    # Safety check - low confidence redirects to UCA medical services
    if confidence < settings.CONFIDENCE_THRESHOLD:
        return ChatResponse(
            answer=f"I am not fully confident in this assessment. {UCA_MEDICAL_CONTACT}",
            confidence=confidence,
            safe=False
        )
    
    return ChatResponse(
        answer=response,
        confidence=confidence,
        safe=confidence >= settings.CONFIDENCE_THRESHOLD
    )
