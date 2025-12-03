import time
from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm import generate_response, should_redirect_to_doctor
from app.config import settings
from app.utils.logger import log_interaction

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
    
    start_time = time.time()
    
    # Check if question requires doctor referral (urgent/complex cases)
    if should_redirect_to_doctor(user_message):
        response_text = f"For medical diagnosis, prescriptions, or complex symptoms, {UCA_MEDICAL_CONTACT}"
        confidence = 0.3
        safe = False
        
        # Log interaction
        response_time_ms = (time.time() - start_time) * 1000
        log_interaction(
            user_message=user_message,
            bot_response=response_text,
            confidence=confidence,
            safe=safe,
            history_length=len(history),
            response_time_ms=response_time_ms,
            metadata={"redirect_reason": "urgent_symptom"}
        )
        
        return ChatResponse(
            answer=response_text,
            confidence=confidence,
            safe=safe
        )
    
    # Model call with history
    response, confidence = generate_response(user_message, history)
    
    # Safety check - low confidence redirects to UCA medical services
    if confidence < settings.CONFIDENCE_THRESHOLD:
        response_text = f"I'm not fully confident in my assessment. {UCA_MEDICAL_CONTACT}"
        safe = False
    else:
        response_text = response
        safe = confidence >= settings.CONFIDENCE_THRESHOLD
    
    # Log interaction
    response_time_ms = (time.time() - start_time) * 1000
    log_interaction(
        user_message=user_message,
        bot_response=response_text,
        confidence=confidence,
        safe=safe,
        history_length=len(history),
        response_time_ms=response_time_ms,
        metadata={"redirected": confidence < settings.CONFIDENCE_THRESHOLD}
    )
    
    return ChatResponse(
        answer=response_text,
        confidence=confidence,
        safe=safe
    )
