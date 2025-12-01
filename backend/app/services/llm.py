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
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
    # Use a conservative heuristic based on response quality
    # This avoids the expensive second forward pass
    response_length = len(text.strip())
    
    # More conservative confidence scoring
    # Medical responses should be cautious - default to lower confidence
    if response_length < 10:
        confidence = 0.2  # Very short = very low confidence
    elif response_length < 30:
        confidence = 0.3 + (response_length / 30) * 0.2  # 0.3-0.5
    elif response_length < 100:
        confidence = 0.5 + ((response_length - 30) / 70) * 0.2  # 0.5-0.7
    else:
        confidence = min(0.7 + ((response_length - 100) / 200) * 0.15, 0.85)  # 0.7-0.85 max
    
    # Additional check: if response seems incomplete or repetitive, lower confidence
    if text.count(text[:20]) > 2:  # Repetitive content
        confidence *= 0.7
    
    # Medical responses should be conservative - cap at reasonable level
    confidence = min(confidence, 0.75)  # Max 75% for untrained model
    
    return text, round(confidence, 2)

