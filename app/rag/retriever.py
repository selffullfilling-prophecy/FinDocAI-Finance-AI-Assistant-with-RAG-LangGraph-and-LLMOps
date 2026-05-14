from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from app.rag.vector_store import similarity_search


def retrieve_relevant_chunks(
    question: str,
    collection_name: str,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Document]:
    """Retrieve top-k chunks from an indexed document collection."""

    return similarity_search(
        question,
        collection_name=collection_name,
        k=top_k,
        metadata_filter=metadata_filter,
    )
