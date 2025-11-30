import json
from typing import Dict, List, Optional
from pathlib import Path
from app.config import settings


def load_knowledge_base() -> Dict:
    """
    Load the symptoms knowledge base from JSON file.
    
    Returns:
        Dictionary containing symptom-to-advice mappings
    """
    # Get the path relative to the backend directory
    backend_path = Path(__file__).parent.parent.parent
    kb_path = backend_path / settings.KNOWLEDGE_BASE_PATH
    
    if not kb_path.exists():
        # Return empty dict if file doesn't exist
        return {}
    
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading knowledge base: {e}")
        return {}


def query_symptom(symptom: str) -> Optional[str]:
    """
    Query the knowledge base for advice on a given symptom.
    
    Args:
        symptom: The symptom to look up (case-insensitive)
        
    Returns:
        Advice string if found, None otherwise
    """
    kb = load_knowledge_base()
    
    # Case-insensitive lookup
    symptom_lower = symptom.lower().strip()
    
    # Try exact match first
    if symptom_lower in kb:
        return kb[symptom_lower]
    
    # Try partial match
    for key, value in kb.items():
        if symptom_lower in key.lower() or key.lower() in symptom_lower:
            return value
    
    return None


def get_all_symptoms() -> List[str]:
    """
    Get a list of all symptoms in the knowledge base.
    
    Returns:
        List of symptom strings
    """
    kb = load_knowledge_base()
    return list(kb.keys())

