from langchain_core.documents import Document

from app.rag import answer_service


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


def test_answer_question_blank_question_raises_value_error():
    try:
        answer_service.answer_question("   ", "test_collection")
    except ValueError as exc:
        assert str(exc) == "Question is required."
    else:
        raise AssertionError("Expected ValueError for blank question")


def test_answer_question_no_chunks_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(answer_service, "similarity_search_with_score", lambda **kwargs: [])

    def fail_generate_answer(prompt):
        raise AssertionError("LLM should not be called when no chunks are retrieved")

    monkeypatch.setattr(answer_service, "generate_answer", fail_generate_answer)

    result = answer_service.answer_question("What drove net sales?", "test_collection")

    assert result["answer"] == answer_service.INSUFFICIENT_CONTEXT_ANSWER
    assert result["collection_name"] == "test_collection"
    assert result["top_k"] == 5
    assert result["sources"] == []


def test_answer_question_with_chunks_calls_llm_and_returns_sources(monkeypatch):
    retrieved = [(_document(), 0.42)]
    calls = {}

    def fake_similarity_search_with_score(query, collection_name, k, metadata_filter):
        calls["retrieval"] = {
            "query": query,
            "collection_name": collection_name,
            "k": k,
            "metadata_filter": metadata_filter,
        }
        return retrieved

    def fake_generate_answer(prompt):
        calls["prompt"] = prompt
        return "Net sales were driven by iPhone and Services revenue. [Source 1]"

    monkeypatch.setattr(answer_service, "similarity_search_with_score", fake_similarity_search_with_score)
    monkeypatch.setattr(answer_service, "generate_answer", fake_generate_answer)

    result = answer_service.answer_question(
        "  What drove net sales?  ",
        "test_collection",
        top_k=3,
        metadata_filter={"section_item": "7"},
    )

    assert calls["retrieval"] == {
        "query": "What drove net sales?",
        "collection_name": "test_collection",
        "k": 3,
        "metadata_filter": {"section_item": "7"},
    }
    assert "What drove net sales?" in calls["prompt"]
    assert result["answer"] == "Net sales were driven by iPhone and Services revenue. [Source 1]"
    assert result["sources"][0]["chunk_id"] == "item-7-text-001"
    assert result["sources"][0]["section_item"] == "7"
    assert result["sources"][0]["score"] == 0.42
    assert "Net sales increased" in result["sources"][0]["preview"]


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
