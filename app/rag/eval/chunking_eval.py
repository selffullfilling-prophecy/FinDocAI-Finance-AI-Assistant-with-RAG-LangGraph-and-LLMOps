from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.rag.chunk_artifacts import load_chunks_jsonl
from app.rag.chunk_pipeline import chunk_10k_file


def run_chunking_golden_eval(cases_path: str | Path) -> dict[str, Any]:
    cases_path = Path(cases_path)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    case_results = [evaluate_chunking_case(case) for case in cases]
    return _build_report(case_results)


def evaluate_chunking_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case.get("case_id", "unknown_case")
    document_path = case.get("document_path")
    chunks_path = case.get("chunks_path")

    if chunks_path:
        path = Path(chunks_path)
        if not path.exists():
            return _skipped(case_id, chunks_path, f"Chunks file not found: {chunks_path}")
        chunks = load_chunks_jsonl(path)
        return evaluate_chunk_records_against_case(case, chunks, document_path or chunks_path)

    if not document_path:
        return _failed(case_id, "", ["Case must define document_path or chunks_path."])

    path = Path(document_path)
    if not path.exists():
        return _skipped(case_id, document_path, f"Document not found: {document_path}")

    try:
        documents = chunk_10k_file(path)
    except Exception as exc:
        return _failed(case_id, document_path, [f"Chunking failed: {exc}"])

    chunks = [
        {"page_content": document.page_content, "metadata": document.metadata}
        for document in documents
    ]
    return evaluate_chunk_records_against_case(case, chunks, document_path)


def evaluate_chunk_records_against_case(
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
    document_path: str,
) -> dict[str, Any]:
    case_id = case.get("case_id", "unknown_case")
    issues: list[str] = []
    metadata_list = [chunk.get("metadata", {}) for chunk in chunks]
    found_sections = sorted({str(meta.get("section_item", "UNKNOWN")) for meta in metadata_list})
    chunk_type_counts = Counter(str(meta.get("chunk_type", "UNKNOWN")) for meta in metadata_list)

    if not chunks:
        issues.append("No chunks were produced.")

    expected_sections = {str(item) for item in case.get("expected_sections", [])}
    missing_sections = sorted(expected_sections - set(found_sections))
    if missing_sections:
        issues.append(f"Missing expected sections: {', '.join(missing_sections)}.")

    required_fields = case.get("required_metadata_fields", [])
    missing_metadata = _find_missing_metadata_fields(chunks, required_fields)
    if missing_metadata:
        issues.append(f"Missing required metadata fields: {missing_metadata[:10]}.")

    duplicate_ids = _find_duplicate_chunk_ids(chunks)
    if duplicate_ids:
        issues.append(f"Duplicate chunk_id values: {duplicate_ids[:10]}.")

    unknown_ratio = _unknown_section_ratio(metadata_list)
    max_unknown = float(case.get("max_unknown_section_ratio", 0.05))
    if unknown_ratio > max_unknown:
        issues.append(f"UNKNOWN section ratio {unknown_ratio:.1%} exceeds max {max_unknown:.1%}.")

    min_table_chunks = int(case.get("min_table_chunks", 0))
    if chunk_type_counts.get("table", 0) < min_table_chunks:
        issues.append(
            f"Expected at least {min_table_chunks} table chunks, found {chunk_type_counts.get('table', 0)}."
        )

    for requirement in case.get("must_contain", []):
        section_item = str(requirement.get("section_item", ""))
        section_text = _section_text(chunks, section_item)
        for term in requirement.get("terms", []):
            if term.lower() not in section_text.lower():
                issues.append(f"Section {section_item} does not contain term: {term}.")

    global_text = "\n".join(chunk.get("page_content", "") for chunk in chunks).lower()
    for term in case.get("must_not_contain_global_terms", []):
        if term.lower() in global_text:
            issues.append(f"Global forbidden term found: {term}.")

    status = "failed" if issues else "passed"
    return {
        "case_id": case_id,
        "status": status,
        "document_path": document_path,
        "total_chunks": len(chunks),
        "found_sections": found_sections,
        "chunk_type_counts": dict(chunk_type_counts),
        "unknown_section_ratio": unknown_ratio,
        "issues": issues,
    }


def _build_report(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for result in case_results if result["status"] == "passed")
    failed = sum(1 for result in case_results if result["status"] == "failed")
    skipped = sum(1 for result in case_results if result["status"] == "skipped")
    runnable = total - skipped
    pass_rate = passed / runnable if runnable else 0.0
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "skipped_cases": skipped,
        "pass_rate": pass_rate,
        "summary_score": round(pass_rate * 100, 2),
        "case_results": case_results,
    }


def _skipped(case_id: str, document_path: str, message: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "skipped",
        "document_path": document_path,
        "total_chunks": 0,
        "found_sections": [],
        "chunk_type_counts": {},
        "issues": [message],
    }


def _failed(case_id: str, document_path: str, issues: list[str]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "failed",
        "document_path": document_path,
        "total_chunks": 0,
        "found_sections": [],
        "chunk_type_counts": {},
        "issues": issues,
    }


def _find_missing_metadata_fields(chunks: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        for field in fields:
            if metadata.get(field) in (None, ""):
                missing.append({"chunk_index": index, "field": field})
    return missing


def _find_duplicate_chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    ids = [str(chunk.get("metadata", {}).get("chunk_id", "")) for chunk in chunks]
    return sorted([chunk_id for chunk_id, count in Counter(ids).items() if chunk_id and count > 1])


def _unknown_section_ratio(metadata_list: list[dict[str, Any]]) -> float:
    if not metadata_list:
        return 0.0
    unknown_count = sum(1 for metadata in metadata_list if str(metadata.get("section_item", "UNKNOWN")) == "UNKNOWN")
    return unknown_count / len(metadata_list)


def _section_text(chunks: list[dict[str, Any]], section_item: str) -> str:
    return "\n".join(
        chunk.get("page_content", "")
        for chunk in chunks
        if str(chunk.get("metadata", {}).get("section_item")) == section_item
    )


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    print("Chunking Golden Eval")
    print(f"- Total: {report['total_cases']}")
    print(f"- Passed: {report['passed_cases']}")
    print(f"- Failed: {report['failed_cases']}")
    print(f"- Skipped: {report['skipped_cases']}")
    print(f"- Pass rate: {report['pass_rate']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden chunking evaluation.")
    parser.add_argument("--cases", required=True, help="Path to chunking golden cases JSON.")
    parser.add_argument("--output", help="Optional path to write report JSON.")
    args = parser.parse_args()

    report = run_chunking_golden_eval(args.cases)
    if args.output:
        _write_report(args.output, report)
    _print_summary(report)


if __name__ == "__main__":
    main()
