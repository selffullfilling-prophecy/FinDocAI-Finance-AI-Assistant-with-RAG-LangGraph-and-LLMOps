from __future__ import annotations

from langchain_core.documents import Document


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[tuple[Document, float | None]],
) -> str:
    """Build a grounded RAG prompt from retrieved chunks."""

    context_blocks = [
        _format_source_block(index=index, document=document, score=score)
        for index, (document, score) in enumerate(retrieved_chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)

    return (
        "You are a financial document assistant.\n"
        "Answer using only the provided context.\n"
        "If the context does not contain enough information, say that the provided documents do not contain enough information.\n"
        "Do not invent numbers, dates, financial metrics, or claims.\n"
        "Cite relevant sources using [Source 1], [Source 2], etc.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Answer:"
    )


def _format_source_block(index: int, document: Document, score: float | None) -> str:
    metadata = document.metadata or {}
    section_item = metadata.get("section_item")
    section_title = metadata.get("section_title")
    chunk_type = metadata.get("chunk_type")
    page_start = metadata.get("page_start") or metadata.get("page_number")
    page_end = metadata.get("page_end") or page_start
    page_text = _format_page_range(page_start, page_end)
    score_text = "None" if score is None else f"{score:.6f}"

    return (
        f"[Source {index}]\n"
        f"chunk_id: {metadata.get('chunk_id')}\n"
        f"section: Item {section_item} - {section_title}\n"
        f"chunk_type: {chunk_type}\n"
        f"page: {page_text}\n"
        f"score: {score_text}\n"
        "content:\n"
        f"{document.page_content}"
    )


def _format_page_range(page_start: object, page_end: object) -> str:
    if page_start is None and page_end is None:
        return "unknown"
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"
