from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import settings


# Initialize model and tokenizer
tokenizer = None
model = None


def load_model():
    """Load the model and tokenizer. Called once at startup."""
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_NAME)
    return tokenizer, model


def generate_response(question: str):
    """
    Generate a response to a medical question using the loaded model.
    
    Args:
        question: The user's question string
        
    Returns:
        tuple: (response_text, confidence_score)
    """
    if tokenizer is None or model is None:
        load_model()
    
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=settings.MAX_NEW_TOKENS)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate confidence using utility function
    from app.utils.confidence import calculate_confidence
    confidence = calculate_confidence(text)
    
    return text, round(confidence, 2)

