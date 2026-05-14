from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from app.rag.llm_client import generate_answer
from app.rag.prompt_builder import build_rag_prompt
from app.rag.vector_store import similarity_search_with_score


INSUFFICIENT_CONTEXT_ANSWER = (
    "The provided documents do not contain enough information to answer this question."
)


def answer_question(
    question: str,
    collection_name: str,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a question using retrieved chunks from a Chroma collection."""

    question = question.strip()
    if not question:
        raise ValueError("Question is required.")

    retrieved_chunks = similarity_search_with_score(
        query=question,
        collection_name=collection_name,
        k=top_k,
        metadata_filter=metadata_filter,
    )
    sources = format_source_chunks(retrieved_chunks)

    if not retrieved_chunks:
        return {
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "collection_name": collection_name,
            "top_k": top_k,
            "sources": [],
        }

    prompt = build_rag_prompt(question, retrieved_chunks)
    answer = generate_answer(prompt).strip() or INSUFFICIENT_CONTEXT_ANSWER

    return {
        "answer": answer,
        "collection_name": collection_name,
        "top_k": top_k,
        "sources": sources,
    }


def format_source_chunks(
    retrieved_chunks: list[tuple[Document, float | None]],
) -> list[dict[str, Any]]:
    """Convert retrieved LangChain documents into API source payloads."""

    sources: list[dict[str, Any]] = []
    for document, score in retrieved_chunks:
        metadata = document.metadata or {}
        page_start = _as_int_or_none(metadata.get("page_start") or metadata.get("page_number"))
        page_end = _as_int_or_none(metadata.get("page_end") or page_start)
        sources.append(
            {
                "chunk_id": _as_str_or_none(metadata.get("chunk_id")),
                "section_item": _as_str_or_none(metadata.get("section_item")),
                "section_title": _as_str_or_none(metadata.get("section_title")),
                "chunk_type": _as_str_or_none(metadata.get("chunk_type")),
                "page_start": page_start,
                "page_end": page_end,
                "score": score,
                "preview": _preview(document.page_content),
            }
        )
    return sources


def _preview(text: str, max_length: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _as_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
