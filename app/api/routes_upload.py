from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.chunk_artifacts import (
    build_chunk_artifact_paths,
    evaluate_chunk_records,
    load_chunks_jsonl,
    write_chunks_jsonl,
    write_eval_report,
)
from app.rag.chunk_pipeline import chunk_10k_file
from app.rag.loader import SUPPORTED_EXTENSIONS
from app.rag.vector_store import collection_name_from_file, index_documents
from app.schemas.upload import UploadResponse


router = APIRouter(tags=["Upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    index_to_chroma: bool = Form(False),
) -> UploadResponse:
    """Upload a 10-K PDF/TXT file, save it, and run the chunking pipeline.

    Current scope:
    - Save raw file to `data/raw`
    - Run `chunk_10k_file`
    - Save chunk debug output to `data/processed/*.chunks.jsonl`

    Next scope:
    - Send chunks to `embeddings.py`
    - Store embeddings in Chroma
    """

    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(status_code=400, detail="Missing upload filename.")

    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    settings = get_settings()
    raw_dir = Path(settings.raw_data_dir)
    processed_dir = Path(settings.processed_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / original_name
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    raw_path.write_bytes(contents)

    try:
        chunks = chunk_10k_file(raw_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Document was saved, but no extractable chunks were created.",
        )

    artifacts = _write_processed_chunks(processed_dir, raw_path, chunks)
    collection_name = collection_name_from_file(original_name)
    indexed = False
    vector_count: int | None = None
    indexing_error: str | None = None

    if index_to_chroma:
        try:
            index_result = index_documents(chunks, collection_name)
            indexed = True
            vector_count = index_result["vector_count"]
        except Exception as exc:
            indexing_error = str(exc)

    return UploadResponse(
        file_name=original_name,
        status=200,
        total_chunks=len(chunks),
        message=f"Uploaded and chunked successfully. Debug chunks: {artifacts['versioned_chunks']}",
        processed_path=str(artifacts["versioned_chunks"]),
        latest_processed_path=str(artifacts["latest_chunks"]),
        eval_report_path=str(artifacts["versioned_eval"]),
        latest_eval_report_path=str(artifacts["latest_eval"]),
        chunk_quality_score=artifacts["eval_report"]["score"],
        indexed=indexed,
        collection_name=collection_name,
        vector_count=vector_count,
        indexing_error=indexing_error,
    )


def _write_processed_chunks(
    processed_dir: Path,
    raw_path: Path,
    chunks: list[Document],
) -> dict:
    """Persist versioned/latest chunk artifacts and eval reports."""

    paths = build_chunk_artifact_paths(processed_dir, raw_path)
    write_chunks_jsonl(paths["versioned_chunks"], chunks)
    write_chunks_jsonl(paths["latest_chunks"], chunks)

    eval_report = evaluate_chunk_records(load_chunks_jsonl(paths["versioned_chunks"]))
    write_eval_report(paths["versioned_eval"], eval_report)
    write_eval_report(paths["latest_eval"], eval_report)

    return {**paths, "eval_report": eval_report}
