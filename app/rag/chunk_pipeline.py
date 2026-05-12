from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.rag.chunk_models import ChunkConfig, ChunkRecord
from app.rag.loader import load_document
from app.rag.section_detector import detect_10k_sections, join_pages
from app.rag.splitter import chunk_records_to_documents, split_section


def chunk_loaded_10k_documents(
    documents: list[Document],
    config: ChunkConfig | None = None,
) -> list[Document]:
    """Chunk already-loaded page Documents from a 10-K report.

    This is the main function to call after `loader.py`.

    Expected flow from General-Flow.png:
        loader.py -> chunk_pipeline.py/splitter.py -> embeddings.py -> Chroma

    TODO(student):
    - Add company/ticker/form/year metadata from filename or SEC metadata.
    - Persist processed chunks to `data/processed` for debugging.
    - Add quality checks: no empty chunks, Item 7 exists, Item 8 exists, etc.
    """

    cfg = config or ChunkConfig()
    if not documents:
        return []

    full_text, page_offsets = join_pages(documents)
    sections = detect_10k_sections(full_text, page_offsets)

    if not sections:
        # Fallback keeps the app usable while you improve section detection.
        # TODO(student): Replace this fallback with a stronger document-based parser.
        source_metadata = _source_metadata_from_pages(documents)
        fallback_section = _make_fallback_section(full_text, source_metadata)
        records = split_section(fallback_section, cfg)
        return chunk_records_to_documents(records, source_metadata)

    records: list[ChunkRecord] = []
    for section in sections:
        records.extend(split_section(section, cfg))

    return chunk_records_to_documents(records, _source_metadata_from_pages(documents))


def chunk_10k_file(file_path: str | Path, config: ChunkConfig | None = None) -> list[Document]:
    """Load and chunk a PDF/TXT 10-K file."""

    return chunk_loaded_10k_documents(load_document(file_path), config)


def _source_metadata_from_pages(documents: list[Document]) -> dict:
    first_doc = documents[0]
    return {
        "file_name": first_doc.metadata.get("file_name"),
        "source_path": first_doc.metadata.get("source_path"),
    }


def _make_fallback_section(full_text: str, source_metadata: dict) -> object:
    """Create a SectionSpan-like object for documents without detected Item headings."""

    from app.rag.chunk_models import SectionSpan

    return SectionSpan(
        item_id="UNKNOWN",
        title="Unclassified Document",
        text=full_text,
        start_char=0,
        end_char=len(full_text),
        metadata={"page_number": 1, **source_metadata},
    )
