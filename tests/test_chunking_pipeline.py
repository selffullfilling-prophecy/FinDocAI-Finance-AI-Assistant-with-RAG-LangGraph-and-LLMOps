from langchain_core.documents import Document

from app.rag.chunk_models import ChunkConfig
from app.rag.chunk_pipeline import chunk_loaded_10k_documents
from app.rag.section_detector import detect_10k_sections, join_pages


def test_detect_10k_sections_ignores_table_of_contents_items():
    text = """
Table of Contents
Item 1. Business ........................................ 5
Item 1A. Risk Factors ................................... 12
Item 7. Management Discussion and Analysis .............. 45

Item 1. Business
The company sells software and cloud services.

Item 1A. Risk Factors
The company faces competition, regulation, and execution risks.

Item 7. Management Discussion and Analysis
Revenue increased because of higher demand.
"""

    sections = detect_10k_sections(text)

    assert [section.item_id for section in sections] == ["1", "1A", "7"]
    assert "........" not in sections[0].title
    assert sections[0].text.startswith("Item 1. Business")


def test_detect_10k_sections_skips_backward_item_order():
    text = """
Item 1. Business
Business text.

Item 7. Management Discussion and Analysis
Management text.

Item 1. Business
This line appears after Item 7, so it should not start a new section.
"""

    sections = detect_10k_sections(text)

    assert [section.item_id for section in sections] == ["1", "7"]


def test_detect_10k_sections_adds_page_range_metadata():
    docs = [
        Document(
            page_content="Item 1. Business\nBusiness text.",
            metadata={"file_name": "sample-10k.txt", "source_path": "sample-10k.txt", "page_number": 4},
        ),
        Document(
            page_content="Item 7. Management Discussion and Analysis\nManagement text.",
            metadata={"file_name": "sample-10k.txt", "source_path": "sample-10k.txt", "page_number": 9},
        ),
    ]
    full_text, page_offsets = join_pages(docs)

    sections = detect_10k_sections(full_text, page_offsets)

    assert sections[0].metadata["page_start"] == 4
    assert sections[0].metadata["page_end"] == 4
    assert sections[1].metadata["page_start"] == 9


def test_chunk_loaded_10k_documents_keeps_section_metadata():
    docs = [
        Document(
            page_content=(
                "Item 1. Business\n"
                "The company sells software and cloud services. " * 40
            ),
            metadata={"file_name": "sample-10k.txt", "source_path": "sample-10k.txt", "page_number": 1},
        ),
        Document(
            page_content=(
                "Item 7. Management Discussion and Analysis\n"
                "Revenue increased because of higher demand. " * 40
            ),
            metadata={"file_name": "sample-10k.txt", "source_path": "sample-10k.txt", "page_number": 2},
        ),
    ]

    chunks = chunk_loaded_10k_documents(docs, ChunkConfig(chunk_size=500, chunk_overlap=50))

    assert chunks
    assert {chunk.metadata["section_item"] for chunk in chunks} == {"1", "7"}
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_chunk_loaded_10k_documents_marks_table_chunks():
    docs = [
        Document(
            page_content=(
                "Item 8. Financial Statements and Supplementary Data\n"
                "Revenue              2025        100,000        42%\n"
                "Revenue              2024         90,000        38%\n"
                "Operating income     2025         10,000        10%\n"
                "The notes describe accounting policies. " * 30
            ),
            metadata={"file_name": "sample-10k.txt", "source_path": "sample-10k.txt", "page_number": 3},
        )
    ]

    chunks = chunk_loaded_10k_documents(docs, ChunkConfig(chunk_size=500, chunk_overlap=50))

    # TODO(student): After improving `table_detector.py`, add assertions for
    # table title, currency units, and statement type.
    assert any(chunk.metadata["chunk_type"] == "table" for chunk in chunks)
    assert any(
        "[TABLE 0:" in chunk.page_content
        for chunk in chunks
        if chunk.metadata["chunk_type"] == "section_text"
    )
