from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.conversation_memory import memory_store, normalize_session_id
from app.rag.hybrid_retriever import RetrievedCandidate, retrieve_candidates
from app.rag.llm_client import generate_answer, stream_answer
from app.rag.prompt_builder import build_rag_prompt
from app.rag.query_rewriter import build_retrieval_query
from app.rag.reranker import candidates_to_ranked_documents, rerank_candidates


INSUFFICIENT_CONTEXT_ANSWER = (
    "The provided documents do not contain enough information to answer this question."
)
ANSWER_STATUS_ANSWERED = "answered"
ANSWER_STATUS_INSUFFICIENT = "insufficient_context"
ANSWER_STATUS_UNVERIFIED = "unverified_sources"

INSUFFICIENT_ANSWER_MARKERS = [
    "provided documents do not contain enough information",
    "does not contain enough information",
    "do not contain enough information",
    "not enough information",
    "cannot determine",
    "not provided",
    "not available",
    "not present in the provided context",
]


def answer_question(
    question: str,
    collection_name: str,
    top_k: int = 5,
    candidate_k: int = 20,
    metadata_filter: dict[str, Any] | None = None,
    retrieval_mode: str = "hybrid",
    rerank: bool = True,
    session_id: str | None = None,
    use_memory: bool = True,
    use_memory_for_retrieval: bool = False,
) -> dict[str, Any]:
    """Answer a question using retrieved chunks from a Chroma collection."""

    question, collection_name = _validate_inputs(question, collection_name)
    normalized_session_id = normalize_session_id(session_id)
    history_text = memory_store.build_history_text(normalized_session_id) if use_memory else ""
    retrieval_query, rewrite_debug = build_retrieval_query(
        question=question,
        session_id=normalized_session_id,
        use_memory=use_memory,
        use_memory_for_retrieval=use_memory_for_retrieval,
    )
    candidates = retrieve_candidates(
        query=retrieval_query,
        collection_name=collection_name,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
        retrieval_mode=retrieval_mode,
    )
    retrieved_chunks = _select_retrieved_chunks(
        query=retrieval_query,
        candidates=candidates,
        top_k=top_k,
        metadata_filter=metadata_filter,
        rerank=rerank,
    )
    retrieved_context = format_source_chunks(retrieved_chunks)

    if not retrieved_chunks:
        _remember_turn(normalized_session_id, question, INSUFFICIENT_CONTEXT_ANSWER, [], use_memory)
        return {
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "answer_status": ANSWER_STATUS_INSUFFICIENT,
            "collection_name": collection_name,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "retrieval_mode": retrieval_mode,
            "rerank": rerank,
            "session_id": normalized_session_id,
            "sources": [],
            "retrieved_context": [],
            "debug": {"candidate_count": len(candidates), **rewrite_debug},
        }

    settings = get_settings()
    prompt = build_rag_prompt(
        question,
        retrieved_chunks,
        conversation_history=history_text,
        max_chars_per_chunk=settings.rag_max_chars_per_chunk,
        max_total_context_chars=settings.rag_max_total_context_chars,
    )
    answer = generate_answer(prompt).strip() or INSUFFICIENT_CONTEXT_ANSWER
    cited_numbers = extract_cited_source_numbers(answer)
    cited_sources, citation_debug = filter_cited_sources_with_debug(retrieved_context, cited_numbers)
    answer_status = classify_answer_status(answer, cited_sources, retrieved_context)
    sources = [] if answer_status == ANSWER_STATUS_INSUFFICIENT else cited_sources
    _remember_turn(normalized_session_id, question, answer, sources, use_memory)

    return {
        "answer": answer,
        "answer_status": answer_status,
        "collection_name": collection_name,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "retrieval_mode": retrieval_mode,
        "rerank": rerank,
        "session_id": normalized_session_id,
        "sources": sources,
        "retrieved_context": retrieved_context,
        "debug": {
            "candidate_count": len(candidates),
            "cited_source_numbers": sorted(cited_numbers),
            **rewrite_debug,
            **citation_debug,
        },
    }


def stream_answer_question(
    question: str,
    collection_name: str,
    top_k: int = 5,
    candidate_k: int = 20,
    metadata_filter: dict[str, Any] | None = None,
    retrieval_mode: str = "hybrid",
    rerank: bool = True,
    session_id: str | None = None,
    use_memory: bool = True,
    use_memory_for_retrieval: bool = False,
) -> Iterator[dict[str, Any]]:
    question, collection_name = _validate_inputs(question, collection_name)
    normalized_session_id = normalize_session_id(session_id)
    history_text = memory_store.build_history_text(normalized_session_id) if use_memory else ""
    yield {
        "type": "status",
        "stage": "retrieving",
        "message": "Retrieving relevant context...",
    }
    retrieval_query, rewrite_debug = build_retrieval_query(
        question=question,
        session_id=normalized_session_id,
        use_memory=use_memory,
        use_memory_for_retrieval=use_memory_for_retrieval,
    )
    candidates = retrieve_candidates(
        query=retrieval_query,
        collection_name=collection_name,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
        retrieval_mode=retrieval_mode,
    )
    retrieved_chunks = _select_retrieved_chunks(
        query=retrieval_query,
        candidates=candidates,
        top_k=top_k,
        metadata_filter=metadata_filter,
        rerank=rerank,
    )
    retrieved_context = format_source_chunks(retrieved_chunks)

    if not retrieved_chunks:
        _remember_turn(normalized_session_id, question, INSUFFICIENT_CONTEXT_ANSWER, [], use_memory)
        yield {"type": "token", "content": INSUFFICIENT_CONTEXT_ANSWER}
        yield {
            "type": "metadata",
            "answer_status": ANSWER_STATUS_INSUFFICIENT,
            "retrieved_context": [],
            "debug": {"candidate_count": len(candidates), **rewrite_debug},
        }
        yield {"type": "sources", "sources": []}
        yield {"type": "done"}
        return

    yield {
        "type": "status",
        "stage": "generating",
        "message": "Generating answer...",
    }
    settings = get_settings()
    prompt = build_rag_prompt(
        question,
        retrieved_chunks,
        conversation_history=history_text,
        max_chars_per_chunk=settings.rag_max_chars_per_chunk,
        max_total_context_chars=settings.rag_max_total_context_chars,
    )

    answer_parts: list[str] = []
    for token in stream_answer(prompt):
        answer_parts.append(token)
        yield {"type": "token", "content": token}

    answer = "".join(answer_parts).strip() or INSUFFICIENT_CONTEXT_ANSWER
    cited_numbers = extract_cited_source_numbers(answer)
    cited_sources, citation_debug = filter_cited_sources_with_debug(retrieved_context, cited_numbers)
    answer_status = classify_answer_status(answer, cited_sources, retrieved_context)
    sources = [] if answer_status == ANSWER_STATUS_INSUFFICIENT else cited_sources
    _remember_turn(normalized_session_id, question, answer, sources, use_memory)
    yield {
        "type": "metadata",
        "answer_status": answer_status,
        "retrieved_context": retrieved_context,
        "debug": {
            "candidate_count": len(candidates),
            "cited_source_numbers": sorted(cited_numbers),
            **rewrite_debug,
            **citation_debug,
        },
    }
    yield {"type": "sources", "sources": sources}
    yield {"type": "done"}


def format_source_chunks(
    retrieved_chunks: list[tuple[Document, float | None]],
) -> list[dict[str, Any]]:
    """Convert retrieved LangChain documents into API source payloads."""

    sources: list[dict[str, Any]] = []
    for source_number, (document, score) in enumerate(retrieved_chunks, start=1):
        metadata = document.metadata or {}
        page_start = _as_int_or_none(metadata.get("page_start") or metadata.get("page_number"))
        page_end = _as_int_or_none(metadata.get("page_end") or page_start)
        sources.append(
            {
                "source_number": source_number,
                "chunk_id": _as_str_or_none(metadata.get("chunk_id")),
                "section_item": _as_str_or_none(metadata.get("section_item")),
                "section_title": _as_str_or_none(metadata.get("section_title")),
                "chunk_type": _as_str_or_none(metadata.get("chunk_type")),
                "page_start": page_start,
                "page_end": page_end,
                "score": score,
                "vector_score": _as_float_or_none(metadata.get("vector_score")),
                "keyword_score": _as_float_or_none(metadata.get("keyword_score")),
                "final_score": _as_float_or_none(metadata.get("final_score") or score),
                "preview": _source_preview(document),
            }
        )
    return sources


def is_insufficient_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(marker in answer_lower for marker in INSUFFICIENT_ANSWER_MARKERS)


def extract_cited_source_numbers(answer: str) -> set[int]:
    cited_numbers: set[int] = set()
    for match in re.finditer(r"\[Source\s+(\d+)\]", answer, flags=re.IGNORECASE):
        try:
            source_number = int(match.group(1))
        except ValueError:
            continue
        if source_number > 0:
            cited_numbers.add(source_number)
    return cited_numbers


def filter_cited_sources(
    all_context_sources: list[dict[str, Any]],
    cited_numbers: set[int],
) -> list[dict[str, Any]]:
    sources, _debug = filter_cited_sources_with_debug(all_context_sources, cited_numbers)
    return sources


def filter_cited_sources_with_debug(
    all_context_sources: list[dict[str, Any]],
    cited_numbers: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cited_sources: list[dict[str, Any]] = []
    invalid_citations: list[int] = []
    seen: set[int] = set()

    for source_number in sorted(cited_numbers):
        if source_number in seen:
            continue
        seen.add(source_number)
        index = source_number - 1
        if 0 <= index < len(all_context_sources):
            source = dict(all_context_sources[index])
            source["source_number"] = source_number
            cited_sources.append(source)
        else:
            invalid_citations.append(source_number)

    return cited_sources, {"invalid_citation_numbers": invalid_citations}


def classify_answer_status(
    answer: str,
    cited_sources: list[dict[str, Any]],
    retrieved_context: list[dict[str, Any]],
) -> str:
    if is_insufficient_answer(answer):
        return ANSWER_STATUS_INSUFFICIENT
    if not answer.strip():
        return ANSWER_STATUS_INSUFFICIENT
    if cited_sources:
        return ANSWER_STATUS_ANSWERED
    return ANSWER_STATUS_UNVERIFIED


def _validate_inputs(question: str, collection_name: str) -> tuple[str, str]:
    question = question.strip()
    collection_name = collection_name.strip()
    if not question:
        raise ValueError("Question is required.")
    if not collection_name:
        raise ValueError("Collection name is required.")
    return question, collection_name


def _select_retrieved_chunks(
    query: str,
    candidates: list[RetrievedCandidate],
    top_k: int,
    metadata_filter: dict[str, Any] | None,
    rerank: bool,
) -> list[tuple[Document, float | None]]:
    if rerank:
        return rerank_candidates(query, candidates, top_k=top_k, metadata_filter=metadata_filter)
    return candidates_to_ranked_documents(candidates, top_k=top_k)


def _remember_turn(
    session_id: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    use_memory: bool,
) -> None:
    if not use_memory:
        return
    memory_store.add_user_message(session_id, question)
    memory_store.add_assistant_message(session_id, answer, sources=sources)


def _preview(text: str, max_length: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _source_preview(document: Document) -> str:
    metadata = document.metadata or {}
    table_context = metadata.get("table_context")
    if table_context:
        return _preview(str(table_context), 700)

    table_headers = metadata.get("table_headers")
    if table_headers and metadata.get("chunk_type") == "table":
        header_text = str(table_headers)
        content = document.page_content
        if header_text not in content:
            return _preview(f"{header_text}\n{content}", 700)

    return _preview(document.page_content)


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


def _as_float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sse_encode(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
