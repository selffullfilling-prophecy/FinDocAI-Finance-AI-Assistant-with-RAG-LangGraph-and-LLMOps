from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.chunk_models import ChunkConfig, ChunkRecord, ChunkType, SectionSpan
from app.rag.table_detector import detect_tables, remove_table_spans


def build_recursive_splitter(config: ChunkConfig | None = None) -> RecursiveCharacterTextSplitter:
    """Create the recursive splitter used inside each 10-K section."""

    cfg = config or ChunkConfig()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def split_section(section: SectionSpan, config: ChunkConfig | None = None) -> list[ChunkRecord]:
    """Split one 10-K section into table chunks and narrative chunks.

    Flow:
    1. Detect financial table-like blocks.
    2. Store each table as its own chunk.
    3. Remove table text from the section narrative.
    4. Recursively split the remaining narrative text.

    TODO(student):
    - Add special treatment for Item 8 tables so statements are not fragmented.
    - Add section-specific chunk size tuning, e.g. larger chunks for Item 7 MD&A.
    - Add better chunk titles/summaries for retrieval.
    """

    cfg = config or ChunkConfig()
    splitter = build_recursive_splitter(cfg)
    chunks: list[ChunkRecord] = []

    base_metadata = {
        "section_item": section.item_id,
        "section_title": section.title,
        **section.metadata,
    }

    tables = detect_tables(section.text, cfg)
    for table_index, table in enumerate(tables):
        chunks.append(
            ChunkRecord(
                chunk_id=f"item-{section.item_id}-table-{table_index:03d}",
                content=table.text,
                chunk_type=ChunkType.TABLE,
                metadata={
                    **base_metadata,
                    "table_index": table_index,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                },
            )
        )

    narrative_text = remove_table_spans(section.text, tables)
    narrative_parts = splitter.split_text(narrative_text)

    for text_index, part in enumerate(narrative_parts):
        content = part.strip()
        if len(content) < cfg.min_chunk_chars:
            continue

        chunks.append(
            ChunkRecord(
                chunk_id=f"item-{section.item_id}-text-{text_index:03d}",
                content=content,
                chunk_type=ChunkType.SECTION_TEXT,
                metadata={
                    **base_metadata,
                    "text_index": text_index,
                },
            )
        )

    return chunks


def chunk_records_to_documents(records: list[ChunkRecord], source_metadata: dict | None = None) -> list[Document]:
    """Convert internal chunk records into LangChain Documents."""

    documents: list[Document] = []
    source_metadata = source_metadata or {}

    for record in records:
        documents.append(
            Document(
                page_content=record.content,
                metadata={
                    **source_metadata,
                    **record.metadata,
                    "chunk_id": record.chunk_id,
                    "chunk_type": record.chunk_type.value,
                },
            )
        )

    return documents

