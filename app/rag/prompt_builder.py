from __future__ import annotations

from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.prompt_templates import RAG_USER_PROMPT_TEMPLATE


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[tuple[Document, float | None]],
    conversation_history: str | None = None,
    max_chars_per_chunk: int | None = None,
    max_total_context_chars: int | None = None,
) -> str:
    """Build a grounded RAG prompt from retrieved chunks."""

    settings = get_settings()
    per_chunk_limit = max_chars_per_chunk or settings.rag_max_chars_per_chunk
    total_limit = max_total_context_chars or settings.rag_max_total_context_chars

    context_blocks = [
        _format_source_block(
            index=index,
            document=document,
            score=score,
            max_content_chars=per_chunk_limit,
        )
        for index, (document, score) in enumerate(retrieved_chunks, start=1)
    ]
    context = _limit_context_blocks(context_blocks, total_limit)
    history = (conversation_history or "").strip()
    history_block = f"Conversation history:\n{history}\n\n" if history else ""

    return RAG_USER_PROMPT_TEMPLATE.format(
        context=context,
        conversation_history_block=history_block,
        question=question.strip(),
    )


def _format_source_block(
    index: int,
    document: Document,
    score: float | None,
    max_content_chars: int,
) -> str:
    metadata = document.metadata or {}
    section_item = metadata.get("section_item")
    section_title = metadata.get("section_title")
    chunk_type = metadata.get("chunk_type")
    page_start = metadata.get("page_start") or metadata.get("page_number")
    page_end = metadata.get("page_end") or page_start
    page_text = _format_page_range(page_start, page_end)
    score_text = "None" if score is None else f"{score:.6f}"
    content = _truncate(document.page_content, max_content_chars)

    return (
        f"[Source {index}]\n"
        f"chunk_id: {metadata.get('chunk_id')}\n"
        f"section: Item {section_item} - {section_title}\n"
        f"chunk_type: {chunk_type}\n"
        f"page: {page_text}\n"
        f"score: {score_text}\n"
        "content:\n"
        f"{content}"
    )


def _format_page_range(page_start: object, page_end: object) -> str:
    if page_start is None and page_end is None:
        return "unknown"
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _limit_context_blocks(blocks: list[str], max_total_context_chars: int) -> str:
    selected: list[str] = []
    used = 0
    separator_chars = 2
    for block in blocks:
        remaining = max_total_context_chars - used
        if remaining <= 0:
            break

        block_with_separator = len(block) + (separator_chars if selected else 0)
        if block_with_separator <= remaining:
            selected.append(block)
            used += block_with_separator
            continue

        if remaining > 300:
            header = block.split("\n", 1)[0]
            if header.startswith("[Source "):
                selected.append(_truncate(block, remaining))
        break

    return "\n\n".join(selected)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
