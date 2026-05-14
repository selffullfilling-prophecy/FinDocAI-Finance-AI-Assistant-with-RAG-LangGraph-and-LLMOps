from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=50)
    metadata_filter: dict[str, Any] | None = None
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    rerank: bool = True
    session_id: str | None = None
    use_memory: bool = True
    use_memory_for_retrieval: bool = False


class SourceChunk(BaseModel):
    source_number: int | None = None
    chunk_id: str | None = None
    section_item: str | None = None
    section_title: str | None = None
    chunk_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    final_score: float | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    answer_status: Literal["answered", "insufficient_context", "unverified_sources"]
    collection_name: str
    top_k: int
    candidate_k: int
    retrieval_mode: str
    rerank: bool
    session_id: str
    sources: list[SourceChunk]
    retrieved_context: list[SourceChunk] | None = None
    debug: dict[str, Any] | None = None
