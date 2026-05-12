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

ITEM_ORDER: dict[str, int] = {
    item_id: index
    for index, item_id in enumerate(TEN_K_ITEM_TITLES.keys(), start=1)
}


ITEM_HEADING_RE = re.compile(
    r"(?im)^\s*item\s*"
    r"(?P<item>1A|1B|1C|7A|9A|9B|9C|1[0-6]|[1-9])"
    r"(?:\s*[\.\-:]\s*|\s+)"
    r"(?P<title>[^\n]{0,160})$"
)

TOC_WINDOW_CHARS = 15_000
TOC_TITLE_RE = re.compile(r"(?:\.{2,}|\s{2,})\s*\d{1,4}\s*$")


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


def find_page_range_for_offsets(
    page_offsets: dict[int, int],
    start_char: int,
    end_char: int,
) -> tuple[int | None, int | None]:
    """Return approximate start/end page numbers for a char span."""

    page_start = find_page_for_offset(page_offsets, start_char)
    page_end = find_page_for_offset(page_offsets, max(start_char, end_char - 1))
    return page_start, page_end


def _has_toc_context(match: re.Match[str], full_text: str) -> bool:
    before_match = full_text[max(0, match.start() - 2_000) : match.start()].lower()
    return "table of contents" in before_match or "contents" in before_match


def _looks_like_toc_heading(match: re.Match[str], full_text: str) -> bool:
    """Return True when an Item heading is probably from Table of Contents.

    10-K table-of-contents lines often look like:
        Item 1. Business ........................................ 5
        Item 1A. Risk Factors                                  12

    Real section headings usually do not end with a page number.
    """

    if match.start() > TOC_WINDOW_CHARS:
        return False

    line = full_text[match.start() : match.end()].strip()
    title = (match.group("title") or "").strip()

    has_page_number_pattern = bool(TOC_TITLE_RE.search(title)) or bool(TOC_TITLE_RE.search(line))

    return _has_toc_context(match, full_text) and has_page_number_pattern


def _has_later_same_item(match: re.Match[str], all_matches: list[re.Match[str]]) -> bool:
    """Check whether the same Item appears again later in the report."""

    item_id = normalize_item_id(match.group("item"))
    return any(
        normalize_item_id(other.group("item")) == item_id and other.start() > match.start()
        for other in all_matches
    )


def _filter_table_of_contents_matches(
    matches: list[re.Match[str]],
    full_text: str,
) -> list[re.Match[str]]:
    """Remove likely Table of Contents Item headings.

    Heuristic:
    - Obvious TOC lines near "Table of Contents" are removed.
    - Early duplicate Item headings are removed when the same Item appears again
      later. This catches TOC rows extracted without dot leaders.

    TODO(student): Try logging skipped headings here while tuning against real
    10-K files from SEC EDGAR.
    """

    filtered: list[re.Match[str]] = []

    for match in matches:
        is_early_duplicate = (
            match.start() <= TOC_WINDOW_CHARS
            and _has_toc_context(match, full_text)
            and _has_later_same_item(match, matches)
        )
        if _looks_like_toc_heading(match, full_text) or is_early_duplicate:
            continue

        filtered.append(match)

    return filtered


def _filter_implausible_item_order(matches: list[re.Match[str]]) -> list[re.Match[str]]:
    """Drop headings that move backwards in the expected 10-K Item order.

    After the real report reaches Item 7 or Item 8, a line beginning with
    "Item 1." is usually a cross-reference or extraction artifact, not a new
    section heading.
    """

    filtered: list[re.Match[str]] = []
    highest_order = 0

    for match in matches:
        item_id = normalize_item_id(match.group("item"))
        item_order = ITEM_ORDER.get(item_id)

        if item_order is None:
            continue

        if item_order < highest_order:
            continue

        highest_order = max(highest_order, item_order)
        filtered.append(match)

    return filtered


def detect_10k_sections(full_text: str, page_offsets: dict[int, int] | None = None) -> list[SectionSpan]:
    """Detect 10-K sections using Item headings.

    This starter implementation uses regex because 10-K reports have stable
    headings. It is enough for practice, but not enough for all SEC filings.

    The detector removes likely table-of-contents headings and skips headings
    that move backwards in the expected Item order.
    """

    raw_matches = list(ITEM_HEADING_RE.finditer(full_text))
    toc_filtered_matches = _filter_table_of_contents_matches(raw_matches, full_text)
    matches = _filter_implausible_item_order(toc_filtered_matches)
    sections: list[SectionSpan] = []

    for index, match in enumerate(matches):
        item_id = normalize_item_id(match.group("item"))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        page_anchor = match.start("item")
        section_text = full_text[start:end].strip()

        if not section_text:
            continue

        fallback_title = TEN_K_ITEM_TITLES.get(item_id, "Unknown Section")
        title = (match.group("title") or fallback_title).strip(" .:-") or fallback_title
        page_start, page_end = find_page_range_for_offsets(page_offsets or {}, page_anchor, end)

        sections.append(
            SectionSpan(
                item_id=item_id,
                title=title,
                text=section_text,
                start_char=start,
                end_char=end,
                page_start=page_start,
                page_end=page_end,
                metadata={
                    "page_number": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                },
            )
        )

    return sections
