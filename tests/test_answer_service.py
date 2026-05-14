from langchain_core.documents import Document

from app.rag import answer_service
from app.rag.conversation_memory import memory_store
from app.rag.hybrid_retriever import RetrievedCandidate
from app.rag.query_rewriter import is_follow_up_question


def _document(chunk_id: str = "item-7-text-001", content: str | None = None) -> Document:
    return Document(
        page_content=content or "Net sales increased due to higher iPhone revenue and Services revenue.",
        metadata={
            "chunk_id": chunk_id,
            "section_item": "7",
            "section_title": "Management's Discussion and Analysis",
            "chunk_type": "section_text",
            "page_start": 12,
            "page_end": 13,
        },
    )


def _candidate() -> RetrievedCandidate:
    return _candidate_with_id(
        "item-7-text-001",
        "Net sales increased due to higher iPhone revenue and Services revenue.",
    )


def _candidate_with_id(chunk_id: str, content: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        document=_document(chunk_id=chunk_id, content=content),
        chunk_id=chunk_id,
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
    assert result["answer_status"] == "insufficient_context"
    assert result["collection_name"] == "test_collection"
    assert result["top_k"] == 5
    assert result["candidate_k"] == 20
    assert result["retrieval_mode"] == "hybrid"
    assert result["sources"] == []
    assert result["retrieved_context"] == []


def test_answer_question_cited_answer_returns_only_supporting_sources(monkeypatch):
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
        return "Net sales were driven by iPhone and Services revenue. [Source 1]"

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
    assert result["answer_status"] == "answered"
    assert result["answer"].endswith("[Source 1]")
    assert result["sources"][0]["source_number"] == 1
    assert result["sources"][0]["chunk_id"] == "item-7-text-001"
    assert result["sources"][0]["section_item"] == "7"
    assert result["sources"][0]["score"] == 1.23
    assert result["sources"][0]["final_score"] == 1.23
    assert result["retrieved_context"][0]["chunk_id"] == "item-7-text-001"


def test_insufficient_answer_does_not_get_fallback_citations(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(
        answer_service,
        "generate_answer",
        lambda prompt: "The provided documents do not contain enough information to answer this question.",
    )

    result = answer_service.answer_question(
        "What was Apple's weighted average interest rate in 2024?",
        "test_collection",
        use_memory=False,
    )

    assert result["answer_status"] == "insufficient_context"
    assert result["sources"] == []
    assert len(result["retrieved_context"]) == 1
    assert "Sources:" not in result["answer"]


def test_cited_source_filtering_returns_only_cited_context_source(monkeypatch):
    candidates = [
        _candidate_with_id("source-1", "First related passage."),
        _candidate_with_id("source-2", "Mac and iPhone sales decreased."),
        _candidate_with_id("source-3", "Third related passage."),
    ]
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: candidates)
    monkeypatch.setattr(
        answer_service,
        "generate_answer",
        lambda prompt: "Net sales decreased due to lower Mac and iPhone sales [Source 2].",
    )

    result = answer_service.answer_question(
        "Why did net sales decrease?",
        "test_collection",
        rerank=False,
        use_memory=False,
    )

    assert result["answer_status"] == "answered"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_number"] == 2
    assert result["sources"][0]["chunk_id"] == "source-2"
    assert len(result["retrieved_context"]) == 3


def test_invalid_citation_number_is_ignored(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(answer_service, "generate_answer", lambda prompt: "Answer [Source 99].")

    result = answer_service.answer_question("Question?", "test_collection", use_memory=False)

    assert result["answer_status"] == "unverified_sources"
    assert result["sources"] == []
    assert result["debug"]["invalid_citation_numbers"] == [99]


def test_no_citation_answer_is_unverified_sources(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(answer_service, "generate_answer", lambda prompt: "Net sales decreased due to lower Mac sales.")

    result = answer_service.answer_question("Why did net sales decrease?", "test_collection", use_memory=False)

    assert result["answer_status"] == "unverified_sources"
    assert result["sources"] == []


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


def test_answer_question_updates_memory_with_filtered_sources(monkeypatch):
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
    assert turns[1].sources[0]["chunk_id"] == "item-7-text-001"
    memory_store.clear_session(session_id)


def test_stream_answer_question_yields_metadata_sources_done(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(answer_service, "stream_answer", lambda prompt: iter(["Answer [Source 1]", "."]))

    events = list(
        answer_service.stream_answer_question(
            "What drove net sales?",
            "test_collection",
            use_memory=False,
        )
    )

    assert events[0] == {
        "type": "status",
        "stage": "retrieving",
        "message": "Retrieving relevant context...",
    }
    assert events[1] == {
        "type": "status",
        "stage": "generating",
        "message": "Generating answer...",
    }
    assert events[2] == {"type": "token", "content": "Answer [Source 1]"}
    assert events[-3]["type"] == "metadata"
    assert events[-3]["answer_status"] == "answered"
    assert events[-2]["type"] == "sources"
    assert events[-2]["sources"][0]["chunk_id"] == "item-7-text-001"
    assert events[-1] == {"type": "done"}


def test_stream_answer_question_insufficient_context_has_empty_sources(monkeypatch):
    monkeypatch.setattr(answer_service, "retrieve_candidates", lambda **kwargs: [_candidate()])
    monkeypatch.setattr(
        answer_service,
        "stream_answer",
        lambda prompt: iter(["The provided documents do not contain enough information to answer this question."]),
    )

    events = list(
        answer_service.stream_answer_question(
            "What was Apple's weighted average interest rate in 2024?",
            "test_collection",
            use_memory=False,
        )
    )

    metadata_event = next(event for event in events if event["type"] == "metadata")
    sources_event = next(event for event in events if event["type"] == "sources")
    assert metadata_event["answer_status"] == "insufficient_context"
    assert sources_event["sources"] == []


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


def test_format_source_chunks_uses_table_context_for_preview():
    document = Document(
        page_content="Total gross margin percentage 44.1% 43.3% 41.8%",
        metadata={
            "chunk_id": "item-7-table-009",
            "section_item": "7",
            "chunk_type": "table",
            "page_number": 24,
            "table_context": (
                "2023 2022 2021\n"
                "Total gross margin percentage: 2023 44.1%, 2022 43.3%, 2021 41.8%"
            ),
        },
    )

    sources = answer_service.format_source_chunks([(document, 0.4)])

    assert sources[0]["preview"].startswith("2023 2022 2021")
    assert "44.1%" in sources[0]["preview"]


def test_standalone_question_with_memory_retrieves_only_current_question(monkeypatch):
    session_id = "unit-test-standalone"
    memory_store.clear_session(session_id)
    memory_store.add_user_message(session_id, "What was Apple's weighted average interest rate in 2024?")
    calls = {}

    def fake_retrieve_candidates(query, **kwargs):
        calls["query"] = query
        return [_candidate()]

    monkeypatch.setattr(answer_service, "retrieve_candidates", fake_retrieve_candidates)
    monkeypatch.setattr(answer_service, "generate_answer", lambda prompt: "Answer. [Source 1]")

    answer_service.answer_question(
        "What was Apple's gross margin percentage in 2023?",
        "test_collection",
        session_id=session_id,
        use_memory=True,
        use_memory_for_retrieval=False,
    )

    assert calls["query"] == "What was Apple's gross margin percentage in 2023?"
    memory_store.clear_session(session_id)


def test_follow_up_uses_history_only_when_enabled(monkeypatch):
    session_id = "unit-test-follow-up"
    memory_store.clear_session(session_id)
    memory_store.add_user_message(session_id, "What was Apple's gross margin percentage in 2023?")
    calls = []

    def fake_retrieve_candidates(query, **kwargs):
        calls.append(query)
        return [_candidate()]

    monkeypatch.setattr(answer_service, "retrieve_candidates", fake_retrieve_candidates)
    monkeypatch.setattr(answer_service, "generate_answer", lambda prompt: "Answer. [Source 1]")

    answer_service.answer_question(
        "What about 2022?",
        "test_collection",
        session_id=session_id,
        use_memory=True,
        use_memory_for_retrieval=False,
    )
    memory_store.clear_session(session_id)
    memory_store.add_user_message(session_id, "What was Apple's gross margin percentage in 2023?")
    answer_service.answer_question(
        "What about 2022?",
        "test_collection",
        session_id=session_id,
        use_memory=True,
        use_memory_for_retrieval=True,
    )

    assert calls[0] == "What about 2022?"
    assert calls[1] == "What was Apple's gross margin percentage in 2022?"
    memory_store.clear_session(session_id)


def test_is_follow_up_question_false_for_standalone_metric_year_question():
    assert is_follow_up_question("What was Apple's gross margin percentage in 2023?") is False
    assert is_follow_up_question("What about 2022?") is True
