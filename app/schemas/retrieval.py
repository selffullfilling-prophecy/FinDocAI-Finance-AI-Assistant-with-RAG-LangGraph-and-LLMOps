from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    page_content: str
    metadata: dict
    score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    final_score: float | None = None


class RetrievalRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=50)
    metadata_filter: dict[str, Any] | None = None
    with_score: bool = False
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    rerank: bool = False


class RetrievalResponse(BaseModel):
    collection_name: str
    top_k: int
    candidate_k: int | None = None
    retrieval_mode: str | None = None
    rerank: bool | None = None
    chunks: list[RetrievedChunk]
