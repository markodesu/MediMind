from pydantic import BaseModel, Field
from typing import List, Optional


class MessageHistory(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., description="The user's medical question", min_length=1)
    history: Optional[List[MessageHistory]] = Field(default=None, description="Conversation history")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    answer: str = Field(..., description="The AI-generated response")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    safe: bool = Field(default=True, description="Whether the response is considered safe to display")

