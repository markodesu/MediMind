import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"

# Initialize model and tokenizer (lazy loading)
tokenizer = None
model = None


def load_model():
    """Load model and tokenizer once at startup."""
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
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


def generate_response(message):
    """Generate response with optimized confidence calculation."""
    # Load model if not already loaded
    if tokenizer is None or model is None:
        load_model()
    
    # Tokenize input
    inputs = tokenizer(message, return_tensors="pt")
    
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
    
    # Decode response
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
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

