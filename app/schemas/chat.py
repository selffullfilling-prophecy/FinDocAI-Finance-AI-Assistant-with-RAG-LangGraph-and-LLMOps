from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filter: dict[str, Any] | None = None


class SourceChunk(BaseModel):
    chunk_id: str | None = None
    section_item: str | None = None
    section_title: str | None = None
    chunk_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    collection_name: str
    top_k: int
    sources: list[SourceChunk]
