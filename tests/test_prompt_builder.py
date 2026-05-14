from langchain_core.documents import Document

from app.rag.prompt_builder import build_rag_prompt


def test_build_rag_prompt_includes_question_sources_and_grounding_instructions():
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
    assert "Answer using only the provided context." in prompt
    assert "Do not invent numbers, dates, financial metrics, or claims." in prompt
    assert "Cite relevant sources using [Source 1]" in prompt
