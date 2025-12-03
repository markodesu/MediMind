from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model configuration
    MODEL_NAME: str = "microsoft/phi-2"  # Base model name
    LORA_MODEL_PATH: Optional[str] = None  # Path to trained LoRA adapter (optional, set in .env)
    MAX_NEW_TOKENS: int = 180  # Base for simple questions, doubled for complex questions
    
    # Confidence threshold
    CONFIDENCE_THRESHOLD: float = 0.6
    
    # API configuration
    API_TITLE: str = "MediMind API - University of Central Asia"
    API_VERSION: str = "1.0.0"
    
    # UCA Medical Services (must be set in .env file)
    UCA_MEDICAL_CONTACT_NAME: str = "Dr. Kyal"
    UCA_MEDICAL_PHONE: str = "SET_IN_ENV_FILE"  # Must be set in .env file - will raise error if not set
    UCA_MEDICAL_LOCATION: str = "1st floor, Academic Block, near GYM"
    
    # Knowledge base
    KNOWLEDGE_BASE_PATH: str = "app/knowledge_base/symptoms.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

