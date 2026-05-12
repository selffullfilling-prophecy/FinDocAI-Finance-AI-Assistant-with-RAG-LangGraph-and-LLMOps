from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    """Type of content stored in one vector-store chunk."""

    SECTION_TEXT = "section_text"
    TABLE = "table"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChunkConfig:
    """Central tuning knobs for 10-K chunking.

    Keep these values here so you can run experiments without changing the
    section/table parsing logic.
    """

    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 120
    table_min_columns: int = 3
    table_min_rows: int = 2


@dataclass(frozen=True)
class SectionSpan:
    """A detected 10-K Item section inside a document."""

    item_id: str
    title: str
    text: str
    start_char: int
    end_char: int
    page_start: int | None = None
    page_end: int | None = None 
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableSpan:
    """A table-like text block detected inside a section."""

    text: str
    start_char: int
    end_char: int
    page_start: int | None = None 
    page_end: int | None = None
    row_count: int
    column_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    """Intermediate chunk representation before converting to LangChain Document."""

    chunk_id: str
    content: str
    chunk_type: ChunkType
    metadata: dict[str, Any] = field(default_factory=dict)
