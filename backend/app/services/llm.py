import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings

# Initialize model and tokenizer (lazy loading)
tokenizer = None
model = None


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
        'severe', 'emergency', 'chest pain', 'can\'t breathe', 'difficulty breathing',
        'unconscious', 'bleeding heavily', 'severe allergic reaction', 'overdose',
        'diagnose', 'diagnosis', 'what disease', 'what condition', 'test results',
        'prescription', 'medication', 'what medicine', 'what drug'
    ]
    
    # Check for urgent patterns
    for keyword in urgent_keywords:
        if keyword in message_lower:
            return True
    
    # Check for diagnostic questions
    diagnostic_patterns = [
        r'what (is|are|do i have|disease|condition)',
        r'do i have .+',
        r'am i .+',
        r'should i take .+',
        r'what medicine',
        r'what drug',
        r'prescribe',
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
        if any(word in message_lower for word in follow_up_words):
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
        system_instruction = """You are MediMind, a helpful health information assistant for university students.
Give clear, informative answers that fully address the question. You can use 3-5 sentences for complex questions.
Focus on general health information and basic self-care. Use the conversation history to provide context-aware responses.
For "what to do" questions, give actionable steps. Do not diagnose, prescribe, or give medical advice beyond general information.

"""
    else:
        system_instruction = """You are MediMind, a helpful health information assistant for university students.
Give clear, concise answers in 1-2 complete sentences. Focus on general health information and basic self-care.
For "what to do" questions, give brief actionable steps. Do not diagnose, prescribe, or give medical advice beyond general information.

Examples:
Human: What should I do if I have mild food poisoning?
Assistant: Drink fluids, rest, and seek help if symptoms last more than 2 days.

Human: What is a headache?
Assistant: A headache is pain in the head or neck area, often caused by tension, dehydration, or illness.

"""
    
    # Build conversation context with history
    conversation = system_instruction
    
    if history and len(history) > 0:
        # Include recent history for context (last 6 messages = 3 exchanges)
        recent_history = history[-6:] if len(history) > 6 else history
        
        # Add context instruction if there's history
        if len(recent_history) > 0:
            conversation += "Previous conversation:\n"
        
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
    
    # Add current message
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
        return "I'm not sure how to help with that. Please consult a healthcare professional."
    
    text = text.strip()
    
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
    
    # Add safety disclaimer if not already present (only for actionable advice)
    if is_action_question and 'seek help' not in result.lower() and 'see a doctor' not in result.lower():
        result += " Seek help if symptoms worsen or persist."
    
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
        )
    
    # Decode response - extract only the new assistant response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (everything after the last "Assistant:")
    if "Assistant:" in full_text:
        raw_text = full_text.split("Assistant:")[-1].strip()
    else:
        # Fallback: if format is different, use the full text
        raw_text = full_text.strip()
    
    # Post-process to ensure complete sentences (with complexity awareness)
    text = post_process_response(raw_text, message, is_complex)
    
    # Calculate confidence score
    response_length = len(text.strip())
    response_lower = text.lower().strip()
    
    # Base confidence from length (optimized for concise responses)
    if response_length < 15:
        confidence = 0.3  # Too short = low confidence
    elif response_length < 50:
        confidence = 0.4 + ((response_length - 15) / 35) * 0.15  # 0.4-0.55
    elif response_length < 150:
        confidence = 0.55 + ((response_length - 50) / 100) * 0.2  # 0.55-0.75
    else:
        confidence = 0.75  # Good length, cap at 0.75 for pre-trained model
    
    # Quality adjustments based on content
    medical_keywords = ['symptom', 'health', 'medical', 'doctor', 'patient', 'treatment',
                       'disease', 'condition', 'pain', 'fever', 'headache', 'cough',
                       'rest', 'drink', 'fluids', 'seek help']
    has_medical_context = any(keyword in response_lower for keyword in medical_keywords)
    
    if has_medical_context:
        confidence += 0.05  # Boost for medical relevance
    else:
        confidence -= 0.1  # Lower if not medical-related
    
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
    confidence = max(0.2, min(confidence, 0.75))
    
    return text, round(confidence, 2)

