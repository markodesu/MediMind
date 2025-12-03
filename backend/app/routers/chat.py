from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm import generate_response, should_redirect_to_doctor
from app.config import settings

router = APIRouter()

# UCA-specific medical contact information (dynamically generated from settings)
def get_uca_medical_contact() -> str:
    """Generate UCA medical contact information from settings."""
    return f"""Please visit {settings.UCA_MEDICAL_CONTACT_NAME} at the University of Central Asia (UCA).

Contact Information:
📞 Phone: {settings.UCA_MEDICAL_PHONE}
📍 Location: {settings.UCA_MEDICAL_LOCATION}

{settings.UCA_MEDICAL_CONTACT_NAME} is available to provide professional medical consultation for UCA students and staff."""


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_message = request.message
    history = request.history or []  # Get conversation history
    
    # Debug: Log history (only in development)
    import os
    if os.getenv("DEBUG", "").lower() == "true":
        print(f"📝 Received history: {len(history)} messages")
        for i, msg in enumerate(history):
            print(f"  {i+1}. {msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:50]}...")
    
    # Check if question requires doctor referral (urgent/complex cases)
    if should_redirect_to_doctor(user_message):
        return ChatResponse(
            answer=f"For medical diagnosis, prescriptions, or complex symptoms, {get_uca_medical_contact()}",
            confidence=0.3,
            safe=False
        )
    
    # Model call with history
    response, confidence = generate_response(user_message, history)
    
    # Safety check - low confidence redirects to UCA medical services
    if confidence < settings.CONFIDENCE_THRESHOLD:
        return ChatResponse(
            answer=f"I'm not fully confident in my assessment. {get_uca_medical_contact()}",
            confidence=confidence,
            safe=False
        )
    
    return ChatResponse(
        answer=response,
        confidence=confidence,
        safe=confidence >= settings.CONFIDENCE_THRESHOLD
    )
