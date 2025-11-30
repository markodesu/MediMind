from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    question: str = Field(..., description="The user's medical question", min_length=1)


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    answer: str = Field(..., description="The AI-generated response")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    safe: bool = Field(..., description="Whether the response is considered safe to display")

