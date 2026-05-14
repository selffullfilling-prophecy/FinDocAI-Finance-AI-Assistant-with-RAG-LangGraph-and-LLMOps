from langchain_core.documents import Document

from app.rag.chunk_models import ChunkConfig
from app.rag.chunk_pipeline import chunk_loaded_10k_documents
from app.rag.section_detector import detect_10k_sections, join_pages
from app.rag.splitter import split_large_table_text, split_section
from app.rag.chunk_models import SectionSpan


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


def test_detect_10k_sections_ignores_index_page_items_with_plain_page_numbers():
    text = """
AMAZON.COM, INC.
FORM 10-K
INDEX

  Page
PART I
Item 1. Business 3
Item 1A. Risk Factors 6
PART II
Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations 19
PART IV
Item 16. Form 10-K Summary 74

AMAZON.COM, INC.
PART I
Item 1. Business
Business text.

Item 1A. Risk Factors
Risk text.

PART II
Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Net cash provided by operating activities.

PART IV
Item 16. Form 10-K Summary
Summary text.
"""

    sections = detect_10k_sections(text)

    assert [section.item_id for section in sections] == ["1", "1A", "7", "16"]
    assert sections[0].title == "Business"
    assert sections[2].title.startswith("Management's Discussion")
    assert sections[2].text.startswith("Item 7.")


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


def test_detect_10k_sections_ignores_inline_item_cross_reference():
    text = """
Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
Further information can be found in Part II,
Item 8 of this Form 10-K in the Notes to Consolidated Financial Statements.
The following table shows net sales by reportable segment.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk
Market risk text.

Item 8. Financial Statements and Supplementary Data
Consolidated statements text.
"""

    sections = detect_10k_sections(text)

    assert [section.item_id for section in sections] == ["7", "7A", "8"]
    assert sections[0].text.startswith("Item 7.")
    assert "net sales by reportable segment" in sections[0].text
    assert sections[2].title == "Financial Statements and Supplementary Data"


def test_detect_10k_sections_ignores_item_reference_to_part_ii():
    text = """
Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
See Item 8 of Part II, "Financial Statements and Supplementary Data - Note 1 - Description of Business,"
for additional discussion.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk
Market risk text.

Item 8. Financial Statements and Supplementary Data
Consolidated statements text.
"""

    sections = detect_10k_sections(text)

    assert [section.item_id for section in sections] == ["7", "7A", "8"]
    assert "additional discussion" in sections[0].text


def test_split_large_table_text_keeps_parts_under_limit():
    table_text = "\n".join(f"Exhibit {index} long description text" for index in range(30))

    parts = split_large_table_text(table_text, max_chars=180)

    assert len(parts) > 1
    assert all(len(part) <= 220 for part in parts)
    assert "Exhibit 0" in parts[0]


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


def test_split_table_chunks_carry_year_headers_into_later_parts():
    table_text = "\n".join(
        [
            "2023 2022 2021",
            "Products 36.5% 36.3% 35.3%",
            "Services 70.8% 71.7% 69.7%",
            "Total gross margin percentage 44.1% 43.3% 41.8%",
        ]
    )
    section = SectionSpan(
        item_id="7",
        title="Management's Discussion and Analysis",
        text=table_text,
        start_char=0,
        end_char=len(table_text),
        metadata={"page_number": 24},
    )

    chunks = split_section(section, ChunkConfig(max_table_chunk_chars=55, table_min_rows=2))
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type.value == "table"]

    assert len(table_chunks) > 1
    percentage_chunk = next(chunk for chunk in table_chunks if "Total gross margin percentage" in chunk.content)
    assert "2023" in percentage_chunk.content
    assert "2022" in percentage_chunk.content
    assert "2021" in percentage_chunk.content
    assert "44.1%" in percentage_chunk.metadata["table_context"]
