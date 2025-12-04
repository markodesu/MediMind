import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings

# Initialize model and tokenizer (lazy loading)
tokenizer = None
model = None

# Campus doctor routing sentence (used everywhere for safety)
UCA_DOCTOR_LINE = (
    f"If your symptoms continue, get worse, or worry you, please visit "
    f"{settings.UCA_MEDICAL_CONTACT_NAME} at the UCA medical office on campus."
)

# Short conversational hint to collect basic intake details
INTAKE_HINT = (
    "If you are comfortable, please also tell me when this started, how strong the "
    "symptoms feel (mild, moderate, or strong), where exactly you feel it, and any "
    "other symptoms like fever, cough, dizziness, nausea, or rash."
)


# -------- Static responses for very common, simple issues (no LLM call) --------

def get_static_response(message: str) -> str | None:
    """
    For very common, simple questions, return a predefined response
    to reduce latency and keep answers consistent and safe.

    IMPORTANT:
    - Uses Kyrgyzstan-appropriate common medicines (paracetamol, Taylol Hot)
    - Never provides dosages
    - Always ends with campus doctor routing line
    """
    msg = message.lower().strip()

    # Direct request for campus doctor / Dr. Kyal contact details
    if any(
        phrase in msg
        for phrase in [
            "dr kyal",
            "doctor kyal",
            "campus doctor contact",
            "medical office contact",
            "uca doctor contact",
            "campus clinic contact",
            "medical office phone",
            "doctor phone number",
        ]
    ):
        return (
            f"The UCA campus doctor is {settings.UCA_MEDICAL_CONTACT_NAME}. "
            f"You can contact them at {settings.UCA_MEDICAL_PHONE}, "
            f"and the medical office is located at {settings.UCA_MEDICAL_LOCATION}. "
            "If you feel very unwell, seek help in person as soon as you can."
        )

    # Headache / mild pain
    if any(k in msg for k in ["headache", "head pain", "migraine"]):
        return (
            "For a mild headache, rest in a quiet, dark room and drink enough water. "
            "Many people in Kyrgyzstan also use simple pain relievers like paracetamol, "
            "but a doctor or pharmacist should advise what is best for you. "
            f"{UCA_DOCTOR_LINE} "
            f"{INTAKE_HINT}"
        )

    # Common cold – be stricter so we don't trigger on unrelated 'cold' uses
    cold_symptoms = ["runny nose", "sore throat", "sneezing", "cough"]
    has_cold_symptom = any(k in msg for k in cold_symptoms)
    if has_cold_symptom or ("cold" in msg and has_cold_symptom):
        return (
            "For a common cold, rest, drink warm fluids, and use home remedies like honey with warm tea. "
            "Many people also use simple medicines such as paracetamol or locally available hot drink powders for colds, "
            "but a doctor or pharmacist should guide what is safe for you. "
            f"{UCA_DOCTOR_LINE} "
            f"{INTAKE_HINT}"
        )

    # Flu
    if "flu" in msg or "gripp" in msg:
        return (
            "Flu usually causes fever, body aches, cough, and fatigue; rest at home, drink plenty of fluids, and stay warm. "
            "In Kyrgyzstan, people often use simple medicines like paracetamol and some hot flu powders to relieve fever and aches, "
            "but it is safest to follow a doctor or pharmacist’s advice. "
            f"{UCA_DOCTOR_LINE} "
            f"{INTAKE_HINT}"
        )

    # Mild stomach issues / food poisoning
    if any(k in msg for k in ["stomach ache", "stomach pain", "stomachache", "stomach-ache", "food poisoning", "diarrhea", "vomit", "vomiting"]):
        return (
            "For mild stomach upset or possible food poisoning, rest and sip water or oral rehydration slowly to avoid "
            "dehydration, and avoid heavy or spicy food for a while. "
            f"{UCA_DOCTOR_LINE} "
            f"{INTAKE_HINT}"
        )

    # Period pain / menstrual cramps
    if any(k in msg for k in ["period pain", "menstrual pain", "cramps", "dysmenorrhea", "period cramps"]):
        return (
            "Period cramps are very common, but the pain should usually be manageable. Gentle heat on the lower abdomen, "
            "light stretching, rest, and relaxation can help ease cramps. Some people also use simple pain relievers like "
            "paracetamol, but a doctor or pharmacist should confirm what is safe for you. If the pain is much stronger than "
            "your usual periods, comes with very heavy bleeding, large clots, dizziness, or you might be pregnant, please see "
            "a gynecologist or the campus doctor as soon as you can. "
            f"{UCA_DOCTOR_LINE} "
            f"{INTAKE_HINT}"
        )

    # Stress / anxiety
    if any(k in msg for k in ["stress", "stressed", "anxiety", "panic", "worried", "depressed"]):
        return (
            "When feeling stressed or anxious, try slowing your breathing, taking short walks, and talking with a "
            "trusted friend, family member, or counselor. "
            f"{UCA_DOCTOR_LINE} "
            f"{INTAKE_HINT}"
        )

    # Insomnia / trouble sleeping
    if any(k in msg for k in ["cannot sleep", "can't sleep", "insomnia", "trouble sleeping", "sleep problem"]):
        return (
            "For trouble sleeping, keep a regular sleep schedule, limit screens before bed, avoid caffeine late in the "
            "day, and create a calm, dark environment for sleep. "
            f"{UCA_DOCTOR_LINE} "
            f"{INTAKE_HINT}"
        )

    return None


def load_model():
    """Load model and tokenizer once at startup. Supports LoRA adapters."""
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading model...")
        model_name = settings.MODEL_NAME
        lora_path = settings.LORA_MODEL_PATH
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        # Load LoRA adapter if specified
        if lora_path:
            try:
                from peft import PeftModel
                print(f"Loading LoRA adapter from: {lora_path}")
                model = PeftModel.from_pretrained(base_model, lora_path)
                print("✅ LoRA adapter loaded successfully!")
            except ImportError:
                print("⚠️ PEFT not installed. Install with: pip install peft")
                print("   Falling back to base model...")
                model = base_model
            except Exception as e:
                print(f"⚠️ Error loading LoRA adapter: {e}")
                print("   Falling back to base model...")
                model = base_model
        else:
            model = base_model
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Optimize for inference
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model loaded on GPU")
        else:
            print("⚠️ Model on CPU - responses will be slow")
            # Use CPU optimizations
            torch.set_num_threads(4)  # Use multiple CPU cores
        
        print("Model loaded successfully!")
    return tokenizer, model


def should_redirect_to_doctor(message: str) -> bool:
    """
    Determine if the message requires immediate doctor referral.
    Returns True if question is too complex or requires diagnosis.
    """
    message_lower = message.lower()
    
    # Keywords that indicate need for professional help
    urgent_keywords = [
        "severe",
        "emergency",
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "difficulty breathing",
        "unconscious",
        "bleeding heavily",
        "nose is bleeding",
        "nose bleed",
        "bleeding nose",
        "bleeding from nose",
        "blood from nose",
        "vomiting blood",
        "coughing blood",
        "severe allergic reaction",
        "overdose",
        "diagnose",
        "diagnosis",
        "what disease",
        "what condition",
        "test results",
        "prescription",
        "medication",
        "what medicine",
        "what drug",
    ]
    
    # Check for urgent patterns
    for keyword in urgent_keywords:
        if keyword in message_lower:
            return True
    
    # Check for diagnostic questions (only specific diagnostic patterns, not informational "what is X")
    diagnostic_patterns = [
        r'do i have .+',
        r'am i .+',
        r'what disease do i have',
        r'what condition do i have',
        r'should i take .+',
        r'what medicine should i',
        r'what drug should i',
        r'prescribe',
        r'what is my diagnosis',
        r'do i need .+ medicine',
    ]
    
    for pattern in diagnostic_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False


def is_complex_question(message: str, history: list = None) -> bool:
    """
    Detect if the question is complex and needs a longer response.
    """
    message_lower = message.lower()
    
    # Complex question indicators
    complex_indicators = [
        'explain', 'tell me about', 'describe', 'why', 'how does', 'what causes',
        'difference between', 'compare', 'multiple', 'several', 'and', 'also'
    ]
    
    # Check for multiple questions or long questions
    question_marks = message.count('?')
    if question_marks > 1 or len(message.split()) > 15:
        return True
    
    # Check for complex keywords
    if any(indicator in message_lower for indicator in complex_indicators):
        return True
    
    # Check if it's a follow-up requiring context
    if history and len(history) > 0:
        # If user is asking follow-up questions, it might need more context
        follow_up_words = ['that', 'this', 'it', 'they', 'what about', 'and', 'also']
        # Temporal context words that refer to previous messages
        temporal_words = ['now', 'today', 'yesterday', 'better', 'worse', 'improved', 'still', 'gone', 'went away', 
                         'feeling', 'feels', 'was', 'were', 'is now', 'has been', 'got better', 'got worse']
        # Pronouns and references that need context
        reference_words = ['it', 'that', 'this', 'they', 'them', 'the pain', 'the symptom', 'the headache', 
                          'the pain', 'my head', 'my stomach', 'my throat']
        
        if any(word in message_lower for word in follow_up_words + temporal_words + reference_words):
            return True
    
    return False


def format_conversation(message: str, history: list = None) -> str:
    """
    Format conversation history and current message for phi-2 model.
    Includes system instructions that adapt to question complexity.
    
    Args:
        message: Current user message
        history: List of previous messages (can be dicts or Pydantic models)
    """
    is_complex = is_complex_question(message, history)
    
    # System instructions - adapt based on complexity
    if is_complex:
        system_instruction = """You are MediMind, a friendly and calm health information assistant for university students.
Give clear, informative answers that fully address the question. You can use 3-5 sentences for complex questions.
Focus on general health information and basic self-care.

IMPORTANT: Always respond to the CURRENT message the user just sent. 

TEMPORAL CONTEXT: Pay close attention to temporal references and pronouns. When the user says words like "it", "that", "now", 
"better", "worse", "yesterday", "today", "still", "gone", "feels", "was", "is now" - these refer to previous symptoms or 
topics mentioned in the conversation history. Use the conversation history to understand what "it" or "that" refers to.

If the user mentions a NEW symptom or topic (completely different from history), respond to that NEW topic. But if they use 
temporal words or pronouns ("it is better now", "that's gone", "still hurts"), they are referring to a previous symptom - 
use the history to understand the context.

When a user describes current symptoms, first ask 1-2 short follow-up questions like a doctor (for example:
when it started, how strong it feels, where exactly it is, whether it changed over time, other symptoms, medicines taken,
allergies, or if they were around anyone sick). Be especially kind and non-judgmental for questions about periods,
menstrual cramps, pregnancy, or gynecology, and remind users they can always talk to a doctor in person.
Then give a short, helpful summary and what they can do next. Do not diagnose, prescribe, or give medical advice beyond general information.

If the user asks for contact information for Dr. Kyal or the campus doctor, provide the contact details.

"""
    else:
        system_instruction = """You are MediMind, a friendly health information assistant for university students.

IMPORTANT: Always respond directly to the user's CURRENT question. Read their message carefully.

If the user asks "what is [something]" (like "what is flu"), explain what it is in 2-3 sentences.
If the user asks "what to do" or "what should I do", give practical self-care steps.
If the user describes a symptom (like "my head hurts"), acknowledge it and ask 1-2 follow-up questions.

Examples:
Human: what is flu
Assistant: Flu (influenza) is a viral infection that affects your respiratory system. It typically causes fever, body aches, cough, fatigue, and sometimes chills. Most people recover with rest and fluids.

Human: i have flu what to do
Assistant: If you have flu, rest at home, drink plenty of fluids like water and warm tea, and get enough sleep. You can use simple pain relievers like paracetamol for fever and body aches (check with a pharmacist first). If symptoms get worse, see a doctor.

Human: my head hurts
Assistant: I'm sorry to hear that. When did the headache start, and how strong is it (mild, moderate, or strong)? Also, do you have any other symptoms like fever or nausea?

Do not diagnose or prescribe. Be kind and helpful.

"""
    
    # Build conversation context with history
    # Format system instruction with actual contact information
    contact_info = (
        f"If the user asks for contact information for Dr. Kyal or the campus doctor, "
        f"provide: {settings.UCA_MEDICAL_CONTACT_NAME}, phone: {settings.UCA_MEDICAL_PHONE}, "
        f"location: {settings.UCA_MEDICAL_LOCATION}.\n\n"
    )
    conversation = system_instruction + contact_info
    
    if history and len(history) > 0:
        # Include recent history for context (last 6 messages = 3 exchanges)
        recent_history = history[-6:] if len(history) > 6 else history
        
        # Add context instruction if there's history
        if len(recent_history) > 0:
            conversation += "Previous conversation (use this to understand temporal references like 'it', 'that', 'now', 'better', 'yesterday' in the current message):\n"
        
        for msg in recent_history:
            # Handle both dict and Pydantic model formats
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            
            # Skip empty messages
            if not content or not content.strip():
                continue
            
            if role == "user":
                conversation += f"Human: {content}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n"
        
        conversation += "\n"
    
    # Add current message with emphasis
    conversation += f"Human: {message}\nAssistant:"
    
    return conversation


def post_process_response(text: str, user_message: str, is_complex: bool = False) -> str:
    """
    Post-process response to ensure it's clear and complete.
    Adapts length based on question complexity.
    
    Args:
        text: Raw model response
        user_message: Original user question for context
        is_complex: Whether the question is complex (allows longer response)
    
    Returns:
        Cleaned response with appropriate length
    """
    if not text:
        return (
            "I'm not sure how to help with that in a safe way through chat. "
            f"{UCA_DOCTOR_LINE}"
        )

    text = text.strip()

    # Remove explicit dosage information (safety rule - no dosages)
    text = re.sub(
        r"\b\d+\s*(mg|ml|milligram|milliliter|tablet|pill|capsule)s?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Avoid strong prescription-style words
    text = re.sub(r"\b(antibiotic|amoxicillin|ibuprofen|nsaid|steroid)s?\b", "medicine", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(prescribe|prescription|diagnose|diagnosis)\b", "assess", text, flags=re.IGNORECASE)

    # Find all complete sentences
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    complete_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # If no complete sentences found, try to fix it
    if not complete_sentences:
        # Try splitting on newlines or common patterns
        if '\n' in text:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            complete_sentences = lines[:5]  # Take up to 5 lines
        else:
            # Ensure it ends with punctuation
            if text[-1] not in '.!?':
                text += '.'
            complete_sentences = [text]
    
    # For complex questions, allow more sentences
    max_sentences = 5 if is_complex else 2
    max_length = 400 if is_complex else 200
    
    # For "what to do" questions, prioritize action-oriented sentences
    user_lower = user_message.lower()
    is_action_question = any(phrase in user_lower for phrase in [
        'what should i do', 'what to do', 'how should i', 'how to treat',
        'what can i do', 'what do i do'
    ])
    
    # Select relevant sentences
    if is_action_question:
        # Prefer sentences with action words
        action_sentences = [
            s for s in complete_sentences
            if any(word in s.lower() for word in ['drink', 'rest', 'take', 'avoid', 'seek', 'see', 'visit', 'eat', 'apply', 'should', 'can'])
        ]
        if action_sentences:
            selected = action_sentences[:max_sentences]
        else:
            selected = complete_sentences[:max_sentences]
    else:
        # For informational questions, take first sentences
        selected = complete_sentences[:max_sentences]
    
    # Join selected sentences
    result = ' '.join(selected).strip()
    
    # Ensure it ends with proper punctuation
    if result and result[-1] not in '.!?':
        result += '.'
    
    # Add campus doctor routing if not already present (only for actionable advice)
    lower_result = result.lower()
    if is_action_question and settings.UCA_MEDICAL_CONTACT_NAME.lower() not in lower_result:
        # Remove generic “seek medical help” phrasing if present
        result = re.sub(
            r"seek (medical )?help( if [^.]+)?\.?",
            "",
            result,
            flags=re.IGNORECASE,
        ).strip()
        if result and result[-1] not in ".!?":
            result += "."
        if result:
            result += " "
        result += UCA_DOCTOR_LINE
    
    # Limit length - but be more lenient for complex questions
    if len(result) > max_length:
        if is_complex:
            # For complex questions, truncate more gracefully
            truncated = result[:max_length-3].rsplit('.', 1)[0]
            if truncated:
                result = truncated + '.'
            else:
                # Fallback: truncate at word boundary
                result = result[:max_length-3].rsplit(' ', 1)[0] + '...'
        else:
            # For simple questions, keep it short
            first_sentence = complete_sentences[0] if complete_sentences else result
            if len(first_sentence) <= max_length:
                result = first_sentence
            else:
                truncated = result[:max_length-3].rsplit(' ', 1)[0]
                result = truncated + '...'
    
    return result.strip()


def generate_response(message: str, history: list = None):
    """
    Generate response with conversation history support.
    Uses improved prompts and post-processing for concise, clear answers.
    
    Args:
        message: Current user message
        history: List of previous messages in format [{"role": "user|assistant", "content": "..."}, ...]
    
    Returns:
        tuple: (response_text, confidence_score)
    """
    # All queries now go through the LLM to showcase model intelligence
    # Hard safety redirect for potentially serious issues (e.g., bleeding, chest pain)
    if should_redirect_to_doctor(message):
        return (
            "This sounds like something that should be checked in person. "
            "I cannot safely assess possible serious or bleeding symptoms over chat. "
            f"{UCA_DOCTOR_LINE}",
            0.3,
        )

    # Load model if not already loaded
    if tokenizer is None or model is None:
        load_model()
    
    # Check if question is complex
    is_complex = is_complex_question(message, history)
    
    # Format conversation with history and system instructions
    formatted_prompt = format_conversation(message, history)
    
    # Adjust token limit based on complexity
    token_limit = settings.MAX_NEW_TOKENS * 2 if is_complex else settings.MAX_NEW_TOKENS
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=token_limit,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Reduce repetition
        )
    
    # Decode response - extract only the new assistant response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (everything after the last "Assistant:")
    if "Assistant:" in full_text:
        # Get everything after the last "Assistant:" marker
        parts = full_text.split("Assistant:")
        raw_text = parts[-1].strip()
        
        # Remove any remaining "Human:" references that might appear
        if "Human:" in raw_text:
            raw_text = raw_text.split("Human:")[0].strip()
        
        # Remove any leading/trailing whitespace and newlines
        raw_text = raw_text.strip()
    else:
        # Fallback: if format is different, try to extract from prompt
        # Remove the original prompt to get just the generated text
        prompt_length = len(formatted_prompt)
        if len(full_text) > prompt_length:
            raw_text = full_text[prompt_length:].strip()
        else:
            raw_text = full_text.strip()
        
        # Still try to clean up
        if "Human:" in raw_text:
            raw_text = raw_text.split("Human:")[0].strip()
    
    # Post-process to ensure complete sentences (with complexity awareness)
    text = post_process_response(raw_text, message, is_complex)
    
    # Calculate confidence score
    response_length = len(text.strip())
    response_lower = text.lower().strip()
    user_message_lower = message.lower()
    
    # Base confidence from length (optimized for concise responses)
    # Start with higher base confidence for model-generated responses
    # Be more lenient - even short responses can be valid
    if response_length < 20:
        confidence = 0.5  # Short but could be valid (e.g., "I'm sorry to hear that.")
    elif response_length < 50:
        confidence = 0.55 + ((response_length - 20) / 30) * 0.15  # 0.55-0.7
    elif response_length < 150:
        confidence = 0.7 + ((response_length - 50) / 100) * 0.15  # 0.7-0.85
    else:
        confidence = 0.85  # Good length, cap at 0.85 for pre-trained model
    
    # Analyze user message for medical relevance and danger indicators
    medical_keywords = ['symptom', 'health', 'medical', 'doctor', 'patient', 'treatment',
                       'disease', 'condition', 'pain', 'fever', 'headache', 'cough', 'cold',
                       'flu', 'stomach', 'ache', 'nausea', 'dizziness', 'rash', 'sore',
                       'throat', 'runny', 'nose', 'sneezing', 'fatigue', 'tired', 'weak',
                       'rest', 'drink', 'fluids', 'seek help', 'cramp', 'menstrual', 'period']
    
    # Dangerous/urgent keywords (even if not triggering immediate redirect)
    dangerous_keywords = [
        'severe', 'emergency', 'chest pain', 'breathing', 'unconscious', 'bleeding',
        'blood', 'allergic', 'overdose', 'severe pain', 'extreme', 'intense',
        'cannot', "can't", 'difficulty', 'worsening', 'getting worse', 'rapid',
        'sudden', 'sharp', 'stabbing', 'crushing', 'pressure'
    ]
    
    # Unknown/non-medical keywords (exclude "what" since it's common in medical questions)
    unknown_patterns = [
        r'\b(xyz|abc|test|random)\b',
        r'\b\d{4,}\b',  # Long numbers (likely not medical)
        r'[^\w\s]{3,}',  # Multiple special characters
    ]
    
    # Check user message for medical relevance
    has_medical_keywords = any(keyword in user_message_lower for keyword in medical_keywords)
    has_dangerous_keywords = any(keyword in user_message_lower for keyword in dangerous_keywords)
    has_unknown_patterns = any(re.search(pattern, user_message_lower) for pattern in unknown_patterns)
    
    # Check for simple informational questions (should have higher confidence)
    is_info_question = any(phrase in user_message_lower for phrase in ['what is', 'what are', 'what does', 'what can'])
    is_what_to_do = 'what to do' in user_message_lower or 'what should i' in user_message_lower
    
    # Adjust confidence based on user message analysis
    if has_dangerous_keywords:
        confidence *= 0.5  # Significantly lower for dangerous symptoms
        confidence -= 0.15  # Additional penalty
    elif is_info_question or is_what_to_do:
        # Informational or "what to do" questions should have good confidence
        confidence += 0.1  # Boost for clear question types
    elif not has_medical_keywords:
        # No medical keywords found - lower confidence, but less harsh
        confidence -= 0.05  # Reduced penalty
        if has_unknown_patterns:
            confidence -= 0.1  # Reduced penalty for unknown patterns
    
    # Quality adjustments based on response content
    response_medical_keywords = ['symptom', 'health', 'medical', 'doctor', 'patient', 'treatment',
                       'disease', 'condition', 'pain', 'fever', 'headache', 'cough',
                       'rest', 'drink', 'fluids', 'seek help']
    has_medical_context = any(keyword in response_lower for keyword in response_medical_keywords)
    
    if has_medical_context:
        confidence += 0.1  # Boost for medical relevance in response
    else:
        confidence -= 0.05  # Lower penalty if response not medical-related
    
    # Check for complete sentences
    has_punctuation = any(char in text for char in '.!?')
    if not has_punctuation:
        confidence *= 0.8  # Penalize incomplete responses
    
    # Penalize repetitive content
    if response_length > 20:
        words = text.split()
        if len(words) > 3 and words.count(words[0]) > 2:
            confidence *= 0.7  # Repetitive
    
    # Penalize code/unrelated content
    if any(indicator in response_lower[:50] for indicator in ['def ', 'import ', 'function', 'class ', '```']):
        confidence *= 0.4  # Code = very low confidence
    
    # Normalize to reasonable range for pre-trained model
    # Allow higher confidence for good responses
    confidence = max(0.3, min(confidence, 0.85))
    
    return text, round(confidence, 2)

