from pydantic import BaseModel, Field
from app.schemas.common import Source

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_tools: list[str] = []
    confidence: float | None = None 