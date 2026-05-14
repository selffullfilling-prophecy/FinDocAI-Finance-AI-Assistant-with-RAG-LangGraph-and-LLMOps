from pydantic import BaseModel

class UploadResponse(BaseModel):
    file_name: str
    status: int
    total_chunks: int
    message: str
    processed_path: str | None = None
    latest_processed_path: str | None = None
    eval_report_path: str | None = None
    latest_eval_report_path: str | None = None
    chunk_quality_score: int | None = None
    indexed: bool = False
    collection_name: str | None = None
    vector_count: int | None = None
    indexing_error: str | None = None
