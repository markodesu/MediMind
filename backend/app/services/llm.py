import torch
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


def format_conversation(message: str, history: list = None) -> str:
    """
    Format conversation history and current message for phi-2 model.
    Phi-2 uses a simple chat format.
    
    Args:
        message: Current user message
        history: List of previous messages (can be dicts or Pydantic models)
    """
    if not history:
        # No history, just return the current message
        return f"Human: {message}\nAssistant:"
    
    # Build conversation context
    conversation = ""
    for msg in history:
        # Handle both dict and Pydantic model formats
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            # Pydantic model
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        
        if role == "user":
            conversation += f"Human: {content}\n"
        elif role == "assistant":
            conversation += f"Assistant: {content}\n"
    
    # Add current message
    conversation += f"Human: {message}\nAssistant:"
    
    return conversation


def generate_response(message: str, history: list = None):
    """
    Generate response with conversation history support.
    
    Args:
        message: Current user message
        history: List of previous messages in format [{"role": "user|assistant", "content": "..."}, ...]
    
    Returns:
        tuple: (response_text, confidence_score)
    """
    # Load model if not already loaded
    if tokenizer is None or model is None:
        load_model()
    
    # Format conversation with history
    formatted_prompt = format_conversation(message, history)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate response (this is the slow part on CPU)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # Faster: use greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response - extract only the new assistant response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (everything after the last "Assistant:")
    if "Assistant:" in full_text:
        text = full_text.split("Assistant:")[-1].strip()
    else:
        # Fallback: if format is different, use the full text
        text = full_text.strip()
    
    # OPTIMIZED: Calculate confidence from generation logits (no extra forward pass)
    # Use a more nuanced heuristic based on response quality and relevance
    # This avoids the expensive second forward pass
    response_length = len(text.strip())
    response_lower = text.lower().strip()
    
    # Base confidence from length (more granular)
    if response_length < 10:
        confidence = 0.2  # Very short = very low confidence
    elif response_length < 30:
        confidence = 0.3 + (response_length / 30) * 0.15  # 0.3-0.45
    elif response_length < 100:
        confidence = 0.45 + ((response_length - 30) / 70) * 0.2  # 0.45-0.65
    elif response_length < 200:
        confidence = 0.65 + ((response_length - 100) / 100) * 0.1  # 0.65-0.75
    else:
        confidence = 0.75  # Very long responses cap at 0.75
    
    # Quality adjustments based on content
    # Check for medical relevance indicators
    medical_keywords = ['symptom', 'health', 'medical', 'doctor', 'patient', 'treatment', 
                       'disease', 'condition', 'pain', 'fever', 'headache', 'cough']
    has_medical_context = any(keyword in response_lower for keyword in medical_keywords)
    
    if has_medical_context:
        confidence += 0.05  # Slight boost for medical relevance
    else:
        confidence -= 0.1  # Lower if not medical-related
    
    # Penalize repetitive or incomplete responses
    if response_length > 20 and text.count(text[:20]) > 2:  # Repetitive content
        confidence *= 0.6
    
    # Penalize if response seems like code or unrelated content
    if any(indicator in response_lower[:50] for indicator in ['def ', 'import ', 'function', 'class ', '```']):
        confidence *= 0.5  # Code snippets = low confidence for medical
    
    # Normalize to reasonable range for untrained model
    confidence = max(0.2, min(confidence, 0.75))  # Clamp between 0.2 and 0.75
    
    return text, round(confidence, 2)

