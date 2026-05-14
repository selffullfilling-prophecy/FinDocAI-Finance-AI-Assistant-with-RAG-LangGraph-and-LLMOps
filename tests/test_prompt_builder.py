from langchain_core.documents import Document

from app.rag.prompt_builder import build_rag_prompt


def test_build_rag_prompt_includes_question_sources_and_citation_guardrails():
    document = Document(
        page_content="Net sales increased due to higher iPhone revenue.",
        metadata={
            "chunk_id": "item-7-text-001",
            "section_item": "7",
            "section_title": "Management's Discussion and Analysis",
            "chunk_type": "section_text",
            "page_start": 10,
            "page_end": 12,
        },
    )

    prompt = build_rag_prompt(
        "What drove net sales?",
        [(document, 0.123456)],
    )

    assert "What drove net sales?" in prompt
    assert "[Source 1]" in prompt
    assert "chunk_id: item-7-text-001" in prompt
    assert "section: Item 7 - Management's Discussion and Analysis" in prompt
    assert "page: 10-12" in prompt
    assert "Does the context directly answer the exact question?" in prompt
    assert "requested company, year, period, metric, and unit" in prompt
    assert "do not cite sources" in prompt


def test_build_rag_prompt_includes_conversation_history_without_hiding_current_question():
    document = Document(page_content="Revenue increased.", metadata={"chunk_id": "chunk-1"})

    prompt = build_rag_prompt(
        "What changed in gross margin?",
        [(document, None)],
        conversation_history="User: What drove revenue?\nAssistant: Revenue increased. [Source 1]",
    )

    assert "Conversation history:" in prompt
    assert "User question:\nWhat changed in gross margin?" in prompt


def test_build_rag_prompt_preserves_source_header_when_truncating():
    document = Document(
        page_content="A" * 1000,
        metadata={
            "chunk_id": "chunk-1",
            "section_item": "7",
            "section_title": "MD&A",
            "chunk_type": "section_text",
            "page_start": 1,
        },
    )

    prompt = build_rag_prompt(
        "What happened?",
        [(document, 0.1)],
        max_chars_per_chunk=900,
        max_total_context_chars=350,
    )

    assert "[Source 1]" in prompt
    assert "chunk_id: chunk-1" in prompt
