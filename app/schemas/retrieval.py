from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    page_content: str
    metadata: dict
    score: float | None = None


class RetrievalRequest(BaseModel):
    question: str
    collection_name: str
    top_k: int = 5
    metadata_filter: dict | None = None
    with_score: bool = False


class RetrievalResponse(BaseModel):
    collection_name: str
    top_k: int
    chunks: list[RetrievedChunk]
