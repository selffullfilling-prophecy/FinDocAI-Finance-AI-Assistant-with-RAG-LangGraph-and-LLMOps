from langchain_core.documents import Document

from app.rag import answer_service
from app.rag.conversation_memory import memory_store
from app.rag.hybrid_retriever import RetrievedCandidate


def _document() -> Document:
    return Document(
        page_content="Net sales increased due to higher iPhone revenue and Services revenue.",
        metadata={
            "chunk_id": "item-7-text-001",
            "section_item": "7",
            "section_title": "Management's Discussion and Analysis",
            "chunk_type": "section_text",
            "page_start": 12,
            "page_end": 13,
        },
    )


def _candidate() -> RetrievedCandidate:
    return RetrievedCandidate(
        document=_document(),
        chunk_id="item-7-text-001",
        vector_score=0.2,
        keyword_score=12.0,
        hybrid_score=0.9,
    )


def test_answer_question_blank_question_raises_value_error():
    try:
        answer_service.answer_question("   ", "test_collection")
    except ValueError as exc:
        assert str(exc) == "Question is required."
    else:
        raise AssertionError("Expected ValueError for blank question")


def test_answer_question_blank_collection_raises_value_error():
    try:
        answer_service.answer_question("What drove net sales?", "   ")
    except ValueError as exc:
        assert str(exc) == "Collection name is required."
    else:
        raise AssertionError("Expected ValueError for blank collection")


def test_answer_question_no_chunks_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [])

    def fail_generate_answer(prompt):
        raise AssertionError("LLM should not be called when no chunks are retrieved")

    monkeypatch.setattr(answer_service, "generate_answer", fail_generate_answer)

    result = answer_service.answer_question(
        "What drove net sales?",
        "test_collection",
        use_memory=False,
    )

    assert result["answer"] == answer_service.INSUFFICIENT_CONTEXT_ANSWER
    assert result["collection_name"] == "test_collection"
    assert result["top_k"] == 5
    assert result["candidate_k"] == 20
    assert result["retrieval_mode"] == "hybrid"
    assert result["sources"] == []


def test_answer_question_hybrid_rerank_calls_llm_and_returns_sources(monkeypatch):
    calls = {}

    def fake_retrieve_candidates(query, collection_name, candidate_k, metadata_filter, retrieval_mode):
        calls["retrieval"] = {
            "query": query,
            "collection_name": collection_name,
            "candidate_k": candidate_k,
            "metadata_filter": metadata_filter,
            "retrieval_mode": retrieval_mode,
        }
        return [_candidate()]

    def fake_rerank_candidates(query, candidates, top_k, metadata_filter):
        calls["rerank"] = {"query": query, "top_k": top_k, "metadata_filter": metadata_filter}
        document = Document(
            page_content=candidates[0].document.page_content,
            metadata={**candidates[0].document.metadata, "final_score": 1.23},
        )
        return [(document, 1.23)]

    def fake_generate_answer(prompt):
        calls["prompt"] = prompt
        return "Net sales were driven by iPhone and Services revenue."

    monkeypatch.setattr(answer_service, "retrieve_candidates", fake_retrieve_candidates)
    monkeypatch.setattr(answer_service, "rerank_candidates", fake_rerank_candidates)
    monkeypatch.setattr(answer_service, "generate_answer", fake_generate_answer)

    result = answer_service.answer_question(
        "  What drove net sales?  ",
        "test_collection",
        top_k=3,
        candidate_k=9,
        metadata_filter={"section_item": "7"},
        retrieval_mode="hybrid",
        rerank=True,
        session_id="unit-test-answer",
        use_memory=False,
    )

    assert calls["retrieval"] == {
        "query": "What drove net sales?",
        "collection_name": "test_collection",
        "candidate_k": 9,
        "metadata_filter": {"section_item": "7"},
        "retrieval_mode": "hybrid",
    }
    assert calls["rerank"] == {
        "query": "What drove net sales?",
        "top_k": 3,
        "metadata_filter": {"section_item": "7"},
    }
    assert "What drove net sales?" in calls["prompt"]
    assert result["answer"].endswith("Sources: [Source 1]")
    assert result["sources"][0]["chunk_id"] == "item-7-text-001"
    assert result["sources"][0]["section_item"] == "7"
    assert result["sources"][0]["score"] == 1.23
    assert result["sources"][0]["final_score"] == 1.23


def test_answer_question_without_rerank_uses_candidate_scores(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(answer_service, "generate_answer", lambda prompt: "Answer. [Source 1]")

    result = answer_service.answer_question(
        "What drove net sales?",
        "test_collection",
        rerank=False,
        use_memory=False,
    )

    assert result["rerank"] is False
    assert result["sources"][0]["vector_score"] == 0.2
    assert result["sources"][0]["keyword_score"] == 12.0


def test_answer_question_updates_memory(monkeypatch):
    session_id = "unit-test-memory"
    memory_store.clear_session(session_id)
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(answer_service, "generate_answer", lambda prompt: "Answer. [Source 1]")

    answer_service.answer_question(
        "What drove net sales?",
        "test_collection",
        session_id=session_id,
        use_memory=True,
    )

    turns = memory_store.get_recent_history(session_id, max_turns=10)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "What drove net sales?"
    memory_store.clear_session(session_id)


def test_stream_answer_question_yields_tokens_sources_done(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(answer_service, "stream_answer", lambda prompt: iter(["Answer", "."]))

    events = list(
        answer_service.stream_answer_question(
            "What drove net sales?",
            "test_collection",
            use_memory=False,
        )
    )

    assert events[0] == {"type": "token", "content": "Answer"}
    assert events[-2]["type"] == "sources"
    assert events[-1] == {"type": "done"}


def test_format_source_chunks_uses_page_number_fallback_and_truncates_preview():
    document = Document(
        page_content=("cash flows " * 80).strip(),
        metadata={
            "chunk_id": "item-8-table-001",
            "section_item": "8",
            "chunk_type": "table",
            "page_number": 44,
        },
    )

    sources = answer_service.format_source_chunks([(document, None)])

    assert sources[0]["page_start"] == 44
    assert sources[0]["page_end"] == 44
    assert sources[0]["score"] is None
    assert len(sources[0]["preview"]) <= 280
    assert sources[0]["preview"].endswith("...")
