from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model configuration
    MODEL_NAME: str = "google/flan-t5-small"
    MAX_NEW_TOKENS: int = 60
    
    # Confidence threshold
    CONFIDENCE_THRESHOLD: float = 0.6
    
    # API configuration
    API_TITLE: str = "MediMind API - University of Central Asia"
    API_VERSION: str = "1.0.0"
    
    # UCA Medical Services
    UCA_MEDICAL_CONTACT_NAME: str = "Dr. Kyal"
    UCA_MEDICAL_PHONE: str = "+996708136013"
    UCA_MEDICAL_LOCATION: str = "1st floor, Academic Block, near GYM"
    
    # Knowledge base
    KNOWLEDGE_BASE_PATH: str = "app/knowledge_base/symptoms.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

