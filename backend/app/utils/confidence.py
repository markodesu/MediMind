def calculate_confidence(text: str) -> float:
    """
    Calculate confidence score for a generated response.
    
    Args:
        text: The generated response text
        
    Returns:
        Confidence score between 0 and 1
    """
    # Simple heuristic: longer outputs (up to a point) indicate higher confidence
    # This is a placeholder - in production, you might use model logits or other metrics
    text_length = len(text)
    
    # Normalize to 0-1 range (assuming 50 chars is a good response length)
    confidence = min(text_length / 50.0, 1.0)
    
    # Ensure minimum confidence for non-empty responses
    if text_length > 0:
        confidence = max(confidence, 0.1)
    else:
        confidence = 0.0
    
    return confidence

