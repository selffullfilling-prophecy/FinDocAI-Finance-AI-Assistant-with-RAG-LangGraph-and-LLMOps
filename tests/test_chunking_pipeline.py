from langchain_core.documents import Document

from app.rag.chunk_models import ChunkConfig
from app.rag.chunk_pipeline import chunk_loaded_10k_documents


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
