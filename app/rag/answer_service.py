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
from app.rag.reranker import candidates_to_ranked_documents, rerank_candidates


INSUFFICIENT_CONTEXT_ANSWER = (
    "The provided documents do not contain enough information to answer this question."
)


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
) -> dict[str, Any]:
    """Answer a question using retrieved chunks from a Chroma collection."""

    question, collection_name = _validate_inputs(question, collection_name)
    normalized_session_id = normalize_session_id(session_id)
    history_text = memory_store.build_history_text(normalized_session_id) if use_memory else ""
    retrieval_query = _build_retrieval_query(question, normalized_session_id, use_memory)
    candidates = retrieve_candidates(
        query=retrieval_query,
        collection_name=collection_name,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
        retrieval_mode=retrieval_mode,
    )
    retrieved_chunks = _select_retrieved_chunks(
        query=question,
        candidates=candidates,
        top_k=top_k,
        metadata_filter=metadata_filter,
        rerank=rerank,
    )
    sources = format_source_chunks(retrieved_chunks)

    if not retrieved_chunks:
        _remember_turn(normalized_session_id, question, INSUFFICIENT_CONTEXT_ANSWER, [], use_memory)
        return {
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "collection_name": collection_name,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "retrieval_mode": retrieval_mode,
            "rerank": rerank,
            "session_id": normalized_session_id,
            "sources": [],
            "debug": {"retrieval_query": retrieval_query, "candidate_count": len(candidates)},
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
    answer = apply_citation_fallback(answer, sources)
    _remember_turn(normalized_session_id, question, answer, sources, use_memory)

    return {
        "answer": answer,
        "collection_name": collection_name,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "retrieval_mode": retrieval_mode,
        "rerank": rerank,
        "session_id": normalized_session_id,
        "sources": sources,
        "debug": {"retrieval_query": retrieval_query, "candidate_count": len(candidates)},
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
) -> Iterator[dict[str, Any]]:
    question, collection_name = _validate_inputs(question, collection_name)
    normalized_session_id = normalize_session_id(session_id)
    history_text = memory_store.build_history_text(normalized_session_id) if use_memory else ""
    retrieval_query = _build_retrieval_query(question, normalized_session_id, use_memory)
    candidates = retrieve_candidates(
        query=retrieval_query,
        collection_name=collection_name,
        candidate_k=candidate_k,
        metadata_filter=metadata_filter,
        retrieval_mode=retrieval_mode,
    )
    retrieved_chunks = _select_retrieved_chunks(
        query=question,
        candidates=candidates,
        top_k=top_k,
        metadata_filter=metadata_filter,
        rerank=rerank,
    )
    sources = format_source_chunks(retrieved_chunks)

    if not retrieved_chunks:
        _remember_turn(normalized_session_id, question, INSUFFICIENT_CONTEXT_ANSWER, [], use_memory)
        yield {"type": "token", "content": INSUFFICIENT_CONTEXT_ANSWER}
        yield {"type": "sources", "sources": []}
        yield {"type": "done"}
        return

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
    answer = apply_citation_fallback(answer, sources)
    if "[Source" not in "".join(answer_parts) and sources:
        yield {"type": "token", "content": answer[len("".join(answer_parts)) :]}

    _remember_turn(normalized_session_id, question, answer, sources, use_memory)
    yield {"type": "sources", "sources": sources}
    yield {"type": "done"}


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
                "vector_score": _as_float_or_none(metadata.get("vector_score")),
                "keyword_score": _as_float_or_none(metadata.get("keyword_score")),
                "final_score": _as_float_or_none(metadata.get("final_score") or score),
                "preview": _preview(document.page_content),
            }
        )
    return sources


def apply_citation_fallback(answer: str, sources: list[dict[str, Any]]) -> str:
    if not sources or "[Source" in answer:
        return answer
    citations = ", ".join(f"[Source {index}]" for index, _source in enumerate(sources, start=1))
    return f"{answer}\n\nSources: {citations}"


def _validate_inputs(question: str, collection_name: str) -> tuple[str, str]:
    question = question.strip()
    collection_name = collection_name.strip()
    if not question:
        raise ValueError("Question is required.")
    if not collection_name:
        raise ValueError("Collection name is required.")
    return question, collection_name


def _build_retrieval_query(question: str, session_id: str, use_memory: bool) -> str:
    if not use_memory:
        return question
    turns = memory_store.get_recent_history(session_id, max_turns=4)
    if not turns:
        return question
    history = "\n".join(f"{turn.role}: {_preview(turn.content, 300)}" for turn in turns)
    return f"{history}\ncurrent question: {question}"


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
