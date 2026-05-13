from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.rag.section_detector import ITEM_ORDER


CORE_10K_ITEMS = {"1", "1A", "7", "8"}


def build_chunk_artifact_paths(processed_dir: Path, raw_path: Path) -> dict[str, Path]:
    """Return versioned and latest artifact paths for one uploaded document."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = raw_path.stem
    return {
        "versioned_chunks": processed_dir / f"{stem}.{timestamp}.chunks.jsonl",
        "latest_chunks": processed_dir / f"{stem}.chunks.latest.jsonl",
        "versioned_eval": processed_dir / f"{stem}.{timestamp}.chunk_eval.json",
        "latest_eval": processed_dir / f"{stem}.chunk_eval.latest.json",
    }


def write_chunks_jsonl(path: Path, chunks: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                chunks.append(json.loads(stripped))
    return chunks


def evaluate_chunk_records(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic chunk quality checks.

    These checks do not prove semantic correctness, but they catch common
    chunking failures: TOC leakage, missing core sections, duplicate chunk IDs,
    bad metadata, and pathological chunk sizes.
    """

    metadata_list = [chunk.get("metadata", {}) for chunk in chunks]
    contents = [chunk.get("page_content", "") for chunk in chunks]
    section_counts = Counter(str(meta.get("section_item", "UNKNOWN")) for meta in metadata_list)
    type_counts = Counter(str(meta.get("chunk_type", "UNKNOWN")) for meta in metadata_list)
    chunk_ids = [str(meta.get("chunk_id", "")) for meta in metadata_list]
    duplicate_chunk_ids = sorted([chunk_id for chunk_id, count in Counter(chunk_ids).items() if chunk_id and count > 1])
    lengths = [len(content) for content in contents]

    section_ranges = _section_page_ranges(metadata_list)
    issues: list[dict[str, Any]] = []

    missing_core = sorted(CORE_10K_ITEMS - set(section_counts))
    if missing_core:
        issues.append(
            {
                "severity": "high",
                "code": "missing_core_sections",
                "message": f"Missing common 10-K sections: {', '.join(missing_core)}.",
            }
        )

    if "UNKNOWN" in section_counts:
        issues.append(
            {
                "severity": "medium",
                "code": "unknown_section_chunks",
                "message": f"{section_counts['UNKNOWN']} chunks have UNKNOWN section metadata.",
            }
        )

    if duplicate_chunk_ids:
        issues.append(
            {
                "severity": "high",
                "code": "duplicate_chunk_ids",
                "message": f"Duplicate chunk IDs: {', '.join(duplicate_chunk_ids[:10])}.",
            }
        )

    suspicious_titles = _find_suspicious_section_titles(metadata_list)
    if suspicious_titles:
        issues.append(
            {
                "severity": "high",
                "code": "toc_title_leakage",
                "message": "Some section titles look like Table of Contents rows with page numbers.",
                "examples": suspicious_titles[:10],
            }
        )

    non_monotonic = _find_non_monotonic_section_pages(section_ranges)
    if non_monotonic:
        issues.append(
            {
                "severity": "medium",
                "code": "non_monotonic_section_pages",
                "message": "Some section page ranges move backwards.",
                "examples": non_monotonic,
            }
        )

    very_short = sum(1 for length in lengths if length < 80)
    if very_short:
        issues.append(
            {
                "severity": "low",
                "code": "very_short_chunks",
                "message": f"{very_short} chunks are shorter than 80 characters.",
            }
        )

    very_long = sum(1 for length in lengths if length > 2_500)
    if very_long:
        issues.append(
            {
                "severity": "medium",
                "code": "very_long_chunks",
                "message": f"{very_long} chunks are longer than 2,500 characters.",
            }
        )

    toc_markers = _count_toc_marker_chunks(chunks)
    if toc_markers:
        issues.append(
            {
                "severity": "medium",
                "code": "toc_content_leakage",
                "message": f"{toc_markers} chunks contain INDEX/Table of Contents markers.",
            }
        )

    return {
        "score": _score_from_issues(issues),
        "total_chunks": len(chunks),
        "chunk_types": dict(type_counts),
        "section_counts": dict(section_counts),
        "section_page_ranges": section_ranges,
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "avg": int(sum(lengths) / len(lengths)) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "issues": issues,
    }


def write_eval_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _section_page_ranges(metadata_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}
    for meta in metadata_list:
        item = str(meta.get("section_item", "UNKNOWN"))
        title = str(meta.get("section_title", ""))
        page_start = meta.get("page_start") or meta.get("page_number")
        page_end = meta.get("page_end") or page_start
        if page_start is None:
            continue

        current = ranges.setdefault(
            item,
            {"section_item": item, "section_title": title, "page_start": page_start, "page_end": page_end},
        )
        current["page_start"] = min(current["page_start"], page_start)
        current["page_end"] = max(current["page_end"], page_end)

    return sorted(
        ranges.values(),
        key=lambda row: (
            row["page_start"],
            ITEM_ORDER.get(str(row["section_item"]), 999),
        ),
    )


def _find_suspicious_section_titles(metadata_list: list[dict[str, Any]]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for meta in metadata_list:
        item = str(meta.get("section_item", "UNKNOWN"))
        title = str(meta.get("section_title", ""))
        if title and title.split()[-1].isdigit():
            key = (item, title)
            if key not in seen:
                seen.add(key)
                examples.append({"section_item": item, "section_title": title})
    return examples


def _find_non_monotonic_section_pages(section_ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    last_page = 0
    ordered_ranges = sorted(
        section_ranges,
        key=lambda row: ITEM_ORDER.get(str(row["section_item"]), 999),
    )
    for row in ordered_ranges:
        page_start = row.get("page_start") or 0
        if page_start < last_page:
            issues.append(row)
        last_page = max(last_page, row.get("page_end") or page_start)
    return issues[:10]


def _count_toc_marker_chunks(chunks: list[dict[str, Any]]) -> int:
    count = 0
    for chunk in chunks:
        content = chunk.get("page_content", "").lower()
        if ("table of contents" in content or "\nindex\n" in content) and "item 1." in content:
            count += 1
    return count


def _score_from_issues(issues: list[dict[str, Any]]) -> int:
    score = 100
    penalties = {"high": 25, "medium": 12, "low": 5}
    for issue in issues:
        score -= penalties.get(issue.get("severity"), 5)
    return max(score, 0)
