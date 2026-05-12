from __future__ import annotations

import re

from app.rag.chunk_models import ChunkConfig, TableSpan


NUMBER_RE = re.compile(r"[$(]?\d[\d,]*(?:\.\d+)?%?\)?")


def looks_like_table_row(line: str, min_columns: int) -> bool:
    """Heuristic for table rows in extracted PDF text.

    TODO(student): Replace this with a stronger parser later. Good options:
    - Camelot/Tabula for born-digital PDFs
    - layout-aware extraction if you add OCR/layout models
    - custom SEC statement parser for Item 8 financial statements
    """

    stripped = line.strip()
    if not stripped:
        return False

    # Financial statement rows usually contain several number-like cells.
    number_cells = NUMBER_RE.findall(stripped)
    if len(number_cells) >= min_columns - 1:
        return True

    # Some extracted tables preserve multi-space column separators.
    columns = [part for part in re.split(r"\s{2,}", stripped) if part]
    return len(columns) >= min_columns


def detect_tables(text: str, config: ChunkConfig | None = None) -> list[TableSpan]:
    """Find contiguous table-like blocks inside section text."""

    cfg = config or ChunkConfig()
    tables: list[TableSpan] = []
    lines = text.splitlines(keepends=True)

    current_lines: list[str] = []
    current_start: int | None = None
    cursor = 0

    def flush(end_char: int) -> None:
        nonlocal current_lines, current_start

        if current_start is None:
            return

        rows = [line for line in current_lines if line.strip()]
        if len(rows) >= cfg.table_min_rows:
            column_count = max(
                len([part for part in re.split(r"\s{2,}", row.strip()) if part])
                for row in rows
            )
            tables.append(
                TableSpan(
                    text="".join(current_lines).strip(),
                    start_char=current_start,
                    end_char=end_char,
                    row_count=len(rows),
                    column_count=column_count,
                )
            )

        current_lines = []
        current_start = None

    for line in lines:
        line_start = cursor
        cursor += len(line)

        if looks_like_table_row(line, cfg.table_min_columns):
            if current_start is None:
                current_start = line_start
            current_lines.append(line)
            continue

        flush(line_start)

    flush(len(text))
    return tables


def build_table_placeholder(table: TableSpan, table_index: int) -> str:
    """Create a short marker kept in narrative chunks where a table was found."""

    return (
        f"[TABLE {table_index}: {table.row_count} rows x "
        f"{table.column_count} columns stored as separate table chunk]"
    )


def replace_table_spans_with_placeholders(text: str, tables: list[TableSpan]) -> str:
    """Replace table blocks before recursive text splitting.

    The table is still stored as its own chunk. The placeholder keeps nearby
    text useful for retrieval, because the LLM can see that a financial table
    occurred between two narrative paragraphs.
    """

    if not tables:
        return text

    pieces: list[str] = []
    cursor = 0
    for table_index, table in enumerate(sorted(tables, key=lambda span: span.start_char)):
        pieces.append(text[cursor : table.start_char])
        pieces.append(build_table_placeholder(table, table_index))
        cursor = table.end_char
    pieces.append(text[cursor:])

    return "\n".join(piece.strip() for piece in pieces if piece.strip())


def remove_table_spans(text: str, tables: list[TableSpan]) -> str:
    """Backward-compatible alias for older code.

    Prefer `replace_table_spans_with_placeholders`.
    """

    return replace_table_spans_with_placeholders(text, tables)
