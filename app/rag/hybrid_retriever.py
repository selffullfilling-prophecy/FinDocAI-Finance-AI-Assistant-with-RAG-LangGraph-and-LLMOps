from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.rag.keyword_retriever import score_chunk_for_query
from app.rag.vector_store import get_collection_documents, similarity_search_with_score


@dataclass
class RetrievedCandidate:
    document: Document
    chunk_id: str | None
    vector_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None


def retrieve_candidates(
    query: str,
    collection_name: str,
    candidate_k: int = 20,
    metadata_filter: dict[str, Any] | None = None,
    retrieval_mode: str = "hybrid",
) -> list[RetrievedCandidate]:
    """Retrieve vector, keyword, or merged hybrid candidates from a Chroma collection."""

    query = query.strip()
    collection_name = collection_name.strip()
    if not query:
        return []
    if not collection_name:
        raise ValueError("Collection name is required.")

    retrieval_mode = retrieval_mode.lower().strip()
    if retrieval_mode not in {"vector", "keyword", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    candidate_k = max(1, candidate_k)
    vector_candidates = (
        _vector_candidates(query, collection_name, candidate_k, metadata_filter)
        if retrieval_mode in {"vector", "hybrid"}
        else []
    )
    keyword_candidates = (
        _keyword_candidates(query, collection_name, candidate_k, metadata_filter)
        if retrieval_mode in {"keyword", "hybrid"}
        else []
    )

    if retrieval_mode == "vector":
        return _score_vector_only(vector_candidates)[:candidate_k]
    if retrieval_mode == "keyword":
        return _score_keyword_only(keyword_candidates)[:candidate_k]

    merged = _merge_candidates(vector_candidates, keyword_candidates)
    _assign_hybrid_scores(merged)
    return sorted(merged, key=lambda candidate: candidate.hybrid_score or 0.0, reverse=True)[:candidate_k]


def _vector_candidates(
    query: str,
    collection_name: str,
    candidate_k: int,
    metadata_filter: dict[str, Any] | None,
) -> list[RetrievedCandidate]:
    results = similarity_search_with_score(
        query=query,
        collection_name=collection_name,
        k=candidate_k,
        metadata_filter=metadata_filter,
    )
    return [
        RetrievedCandidate(
            document=document,
            chunk_id=_chunk_id(document),
            vector_score=score,
        )
        for document, score in results
    ]


def _keyword_candidates(
    query: str,
    collection_name: str,
    candidate_k: int,
    metadata_filter: dict[str, Any] | None,
) -> list[RetrievedCandidate]:
    documents = get_collection_documents(collection_name, metadata_filter=metadata_filter)
    scored: list[RetrievedCandidate] = []
    for document in documents:
        chunk = {"page_content": document.page_content, "metadata": document.metadata}
        keyword_result = score_chunk_for_query(chunk, query)
        if keyword_result["score"] <= 0:
            continue
        scored.append(
            RetrievedCandidate(
                document=document,
                chunk_id=_chunk_id(document),
                keyword_score=float(keyword_result["score"]),
            )
        )

    return sorted(scored, key=lambda candidate: candidate.keyword_score or 0.0, reverse=True)[:candidate_k]


def _merge_candidates(
    vector_candidates: list[RetrievedCandidate],
    keyword_candidates: list[RetrievedCandidate],
) -> list[RetrievedCandidate]:
    merged: dict[str, RetrievedCandidate] = {}

    for candidate in vector_candidates + keyword_candidates:
        key = candidate.chunk_id or f"content:{hash(candidate.document.page_content)}"
        existing = merged.get(key)
        if existing is None:
            merged[key] = RetrievedCandidate(
                document=candidate.document,
                chunk_id=candidate.chunk_id,
                vector_score=candidate.vector_score,
                keyword_score=candidate.keyword_score,
            )
            continue

        if candidate.vector_score is not None:
            existing.vector_score = candidate.vector_score
        if candidate.keyword_score is not None:
            existing.keyword_score = candidate.keyword_score

    return list(merged.values())


def _assign_hybrid_scores(candidates: list[RetrievedCandidate]) -> None:
    vector_norms = _normalize_vector_scores(candidates)
    keyword_norms = _normalize_keyword_scores(candidates)
    for candidate in candidates:
        key = id(candidate)
        candidate.hybrid_score = 0.65 * vector_norms.get(key, 0.0) + 0.35 * keyword_norms.get(key, 0.0)


def _score_vector_only(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    vector_norms = _normalize_vector_scores(candidates)
    for candidate in candidates:
        candidate.hybrid_score = vector_norms.get(id(candidate), 0.0)
    return sorted(candidates, key=lambda candidate: candidate.hybrid_score or 0.0, reverse=True)


def _score_keyword_only(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    keyword_norms = _normalize_keyword_scores(candidates)
    for candidate in candidates:
        candidate.hybrid_score = keyword_norms.get(id(candidate), 0.0)
    return sorted(candidates, key=lambda candidate: candidate.hybrid_score or 0.0, reverse=True)


def _normalize_vector_scores(candidates: list[RetrievedCandidate]) -> dict[int, float]:
    scored = [candidate for candidate in candidates if candidate.vector_score is not None]
    if not scored:
        return {}

    scores = [float(candidate.vector_score) for candidate in scored]
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return {id(candidate): 1.0 for candidate in scored}

    return {
        id(candidate): 1.0 - ((float(candidate.vector_score) - min_score) / (max_score - min_score))
        for candidate in scored
    }


def _normalize_keyword_scores(candidates: list[RetrievedCandidate]) -> dict[int, float]:
    scored = [candidate for candidate in candidates if candidate.keyword_score is not None]
    if not scored:
        return {}

    max_score = max(float(candidate.keyword_score) for candidate in scored)
    if max_score <= 0:
        return {}

    return {id(candidate): float(candidate.keyword_score) / max_score for candidate in scored}


def _chunk_id(document: Document) -> str | None:
    value = (document.metadata or {}).get("chunk_id")
    if value is None:
        return None
    return str(value)
