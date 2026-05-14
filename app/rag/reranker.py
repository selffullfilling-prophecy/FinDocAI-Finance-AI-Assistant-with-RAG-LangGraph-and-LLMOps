from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from app.rag.hybrid_retriever import RetrievedCandidate
from app.rag.keyword_retriever import tokenize_query


TABLE_QUERY_TERMS = {
    "table",
    "cash flow",
    "balance sheet",
    "statement",
    "operating activities",
}
SECTION_7_QUERY_TERMS = {
    "revenue",
    "net sales",
    "growth",
    "margin",
    "management discussion",
}
SECTION_8_QUERY_TERMS = {
    "cash flow",
    "assets",
    "liabilities",
    "consolidated statement",
}


def rerank_candidates(
    query: str,
    candidates: list[RetrievedCandidate],
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float | None]]:
    """Rerank retrieved candidates with deterministic finance-aware heuristics."""

    query = query.strip()
    if not query or not candidates:
        return []

    scored = [
        (_enrich_document(candidate, final_score), final_score)
        for candidate, final_score in (
            (candidate, _score_candidate(query, candidate, metadata_filter))
            for candidate in candidates
        )
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def candidates_to_ranked_documents(
    candidates: list[RetrievedCandidate],
    top_k: int = 5,
) -> list[tuple[Document, float | None]]:
    scored = [
        (_enrich_document(candidate, _base_score(candidate)), _base_score(candidate))
        for candidate in candidates
    ]
    return sorted(scored, key=lambda item: item[1] or 0.0, reverse=True)[:top_k]


def _score_candidate(
    query: str,
    candidate: RetrievedCandidate,
    metadata_filter: dict[str, Any] | None,
) -> float:
    metadata = candidate.document.metadata or {}
    content_lower = candidate.document.page_content.lower()
    query_lower = query.lower()
    tokens = tokenize_query(query)

    score = _base_score(candidate)
    token_hits = sum(1 for token in tokens if token in content_lower)
    score += min(token_hits, 8) * 0.04

    compact_query = " ".join(tokens)
    if compact_query and compact_query in content_lower:
        score += 0.18

    if _metadata_section_matches(metadata, metadata_filter):
        score += 0.2

    section_item = str(metadata.get("section_item", ""))
    chunk_type = str(metadata.get("chunk_type", "")).lower()

    if _contains_any(query_lower, TABLE_QUERY_TERMS) and chunk_type == "table":
        score += 0.25
    if _contains_any(query_lower, SECTION_7_QUERY_TERMS) and section_item == "7":
        score += 0.15
    if _contains_any(query_lower, SECTION_8_QUERY_TERMS) and section_item == "8":
        score += 0.15

    return round(score, 6)


def _base_score(candidate: RetrievedCandidate) -> float:
    if candidate.hybrid_score is not None:
        return float(candidate.hybrid_score)
    if candidate.keyword_score is not None:
        return float(candidate.keyword_score)
    if candidate.vector_score is not None:
        return 1.0 / (1.0 + max(float(candidate.vector_score), 0.0))
    return 0.0


def _metadata_section_matches(
    metadata: dict[str, Any],
    metadata_filter: dict[str, Any] | None,
) -> bool:
    if not metadata_filter or "section_item" not in metadata_filter:
        return False
    return str(metadata.get("section_item")) == str(metadata_filter["section_item"])


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _enrich_document(candidate: RetrievedCandidate, final_score: float | None) -> Document:
    metadata = dict(candidate.document.metadata or {})
    metadata["vector_score"] = candidate.vector_score
    metadata["keyword_score"] = candidate.keyword_score
    metadata["hybrid_score"] = candidate.hybrid_score
    metadata["final_score"] = final_score
    return Document(page_content=candidate.document.page_content, metadata=metadata)
