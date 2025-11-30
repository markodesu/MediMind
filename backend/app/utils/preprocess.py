import re


def preprocess_text(text: str) -> str:
    """
    Preprocess input text for better model performance.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Ensure question ends with a question mark if it's a question
    if text and not text.endswith(('?', '.', '!')):
        # Simple heuristic: if it contains question words, add ?
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'should']
        if any(text.lower().startswith(word) for word in question_words):
            text += '?'
    
    return text

