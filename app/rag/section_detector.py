from __future__ import annotations

import re
from collections.abc import Iterable

from langchain_core.documents import Document

from app.rag.chunk_models import SectionSpan


# Common 10-K Item headings. This is intentionally explicit so learners can see
# which sections matter most for financial QA.
TEN_K_ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}


ITEM_HEADING_RE = re.compile(
    r"(?im)^\s*item\s+"
    r"(?P<item>1A|1B|1C|7A|9A|9B|9C|1[0-6]|[1-9])"
    r"\.?\s+"
    r"(?P<title>[^\n]{0,160})$"
)


def normalize_item_id(item_id: str) -> str:
    return item_id.strip().upper().rstrip(".")


def join_pages(documents: Iterable[Document]) -> tuple[str, dict[int, int]]:
    """Join page documents and keep char offset -> page number anchor.

    Returns:
        full_text: document text with page separators
        page_offsets: mapping from starting char offset to page number

    TODO(student): Improve this mapping so every chunk can include a more exact
    page range instead of only a start-page anchor.
    """

    parts: list[str] = []
    page_offsets: dict[int, int] = {}
    cursor = 0

    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            continue

        page_number = int(doc.metadata.get("page_number") or 1)
        page_offsets[cursor] = page_number
        parts.append(text)
        cursor += len(text) + 2

    return "\n\n".join(parts), page_offsets


def find_page_for_offset(page_offsets: dict[int, int], char_offset: int) -> int | None:
    """Return the nearest page number at or before char_offset."""

    if not page_offsets:
        return None

    eligible_offsets = [offset for offset in page_offsets if offset <= char_offset]
    if not eligible_offsets:
        return None

    return page_offsets[max(eligible_offsets)]


def detect_10k_sections(full_text: str, page_offsets: dict[int, int] | None = None) -> list[SectionSpan]:
    """Detect 10-K sections using Item headings.

    This starter implementation uses regex because 10-K reports have stable
    headings. It is enough for practice, but not enough for all SEC filings.

    TODO(student):
    - Ignore table-of-contents matches near the beginning of the report.
    - Handle headings like "ITEM 7. MD&A" where text extraction collapses spaces.
    - Add validation that Item order is plausible: 1, 1A, 1B, ..., 7, 7A, 8.
    """

    matches = list(ITEM_HEADING_RE.finditer(full_text))
    sections: list[SectionSpan] = []

    for index, match in enumerate(matches):
        item_id = normalize_item_id(match.group("item"))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        section_text = full_text[start:end].strip()

        if not section_text:
            continue

        fallback_title = TEN_K_ITEM_TITLES.get(item_id, "Unknown Section")
        title = (match.group("title") or fallback_title).strip(" .:-") or fallback_title
        page_number = find_page_for_offset(page_offsets or {}, start)

        sections.append(
            SectionSpan(
                item_id=item_id,
                title=title,
                text=section_text,
                start_char=start,
                end_char=end,
                metadata={"page_number": page_number},
            )
        )

    return sections
