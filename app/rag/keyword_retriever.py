from __future__ import annotations

import json
import re
from typing import Any


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "where",
    "when",
    "which",
    "were",
    "was",
    "are",
    "how",
    "did",
    "does",
    "cua",
    "la",
    "gi",
    "trong",
    "nam",
}


def tokenize_query(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{1,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def score_chunk_for_query(chunk: dict[str, Any], query: str) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    content = chunk.get("page_content", "")
    content_lower = content.lower()
    metadata_text = json.dumps(metadata, ensure_ascii=False).lower()
    tokens = tokenize_query(query)

    matched_terms = sorted(
        {
            token
            for token in tokens
            if token in content_lower or token in metadata_text
        }
    )
    phrase_bonus = 8 if query.strip().lower() in content_lower else 0
    table_bonus = 2 if metadata.get("chunk_type") == "table" else 0
    section_bonus = 2 if str(metadata.get("section_item", "")).lower() in tokens else 0
    score = len(matched_terms) * 3 + phrase_bonus + table_bonus + section_bonus

    return {
        "score": score,
        "matched_terms": matched_terms,
        "chunk": chunk,
    }


def retrieve_from_chunk_records(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []

    scored = [score_chunk_for_query(chunk, query) for chunk in chunks]
    scored = [item for item in scored if item["score"] > 0]
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]
