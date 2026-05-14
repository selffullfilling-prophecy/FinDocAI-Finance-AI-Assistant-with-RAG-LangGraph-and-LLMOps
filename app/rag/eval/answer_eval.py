from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.rag.answer_service import ANSWER_STATUS_INSUFFICIENT, INSUFFICIENT_CONTEXT_ANSWER, answer_question


def run_answer_golden_eval(cases_path: str | Path) -> dict[str, Any]:
    cases_path = Path(cases_path)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    case_results = [evaluate_answer_case(case) for case in cases]
    return _build_report(case_results)


def evaluate_answer_case(case: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("RUN_LLM_EVAL") != "1":
        return _skipped(case, "RUN_LLM_EVAL is not set to 1; skipping live LLM answer eval.")

    if not get_settings().nvidia_api_key:
        return _skipped(case, "NVIDIA_API_KEY is not configured; skipping live LLM answer eval.")

    question = str(case.get("question", "")).strip()
    collection_name = str(case.get("collection_name", "")).strip()
    if not question:
        return _failed(case, ["Question is required."])
    if not collection_name:
        return _failed(case, ["Collection name is required."])

    try:
        response = answer_question(
            question=question,
            collection_name=collection_name,
            top_k=int(case.get("top_k", 5)),
            candidate_k=int(case.get("candidate_k", 20)),
            metadata_filter=case.get("metadata_filter"),
            retrieval_mode=case.get("retrieval_mode", "hybrid"),
            rerank=bool(case.get("rerank", True)),
            session_id=case.get("session_id") or f"eval-{case.get('case_id', 'unknown')}",
            use_memory=False,
        )
    except Exception as exc:
        return _failed(case, [f"Answer generation failed: {exc}"])

    return evaluate_answer_result(case, response)


def evaluate_answer_result(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer", ""))
    answer_status = str(response.get("answer_status", ""))
    sources = response.get("sources", []) or []
    answer_lower = answer.lower()
    issues: list[str] = []

    expected_any = [str(term) for term in case.get("expected_terms_any", [])]
    expected_all = [str(term) for term in case.get("expected_terms_all", [])]
    matched_any = [term for term in expected_any if term.lower() in answer_lower]
    matched_all = [term for term in expected_all if term.lower() in answer_lower]
    missing_all = [term for term in expected_all if term.lower() not in answer_lower]
    forbidden_matches = [
        str(term) for term in case.get("forbidden_terms", []) if str(term).lower() in answer_lower
    ]

    citation_present = "[Source" in answer
    if case.get("require_citation", False):
        if not citation_present:
            issues.append("Citation is required but answer does not contain [Source ...].")
        if not sources:
            issues.append("Citation is required but response.sources is empty.")

    if expected_any and not matched_any:
        issues.append(f"Answer did not contain any expected_terms_any: {expected_any}.")

    if missing_all:
        issues.append(f"Answer is missing expected_terms_all: {missing_all}.")

    if forbidden_matches:
        issues.append(f"Answer contains forbidden terms: {forbidden_matches}.")

    allow_insufficient = bool(case.get("allow_insufficient_context", False))
    insufficient = answer_status == ANSWER_STATUS_INSUFFICIENT or INSUFFICIENT_CONTEXT_ANSWER.lower() in answer_lower
    if insufficient and not allow_insufficient:
        issues.append("Answer reported insufficient context but case does not allow it.")
    if insufficient and sources:
        issues.append("Insufficient-context answers must not return sources.")

    expected_status = case.get("expected_answer_status")
    if expected_status and answer_status != expected_status:
        issues.append(f"Expected answer_status={expected_status}, got {answer_status}.")

    expected_sections = {str(item) for item in case.get("expected_source_sections", [])}
    source_sections = {str(source.get("section_item")) for source in sources if source.get("section_item") is not None}
    source_section_hit = not expected_sections or bool(expected_sections & source_sections)
    if expected_sections and not source_section_hit:
        issues.append(f"No source section matched expected_source_sections: {sorted(expected_sections)}.")

    status = "failed" if issues else "passed"
    return {
        "case_id": case.get("case_id", "unknown_case"),
        "status": status,
        "question": case.get("question", ""),
        "collection_name": case.get("collection_name"),
        "answer": answer,
        "answer_status": answer_status,
        "sources": [_summarize_source(index, source) for index, source in enumerate(sources, start=1)],
        "matched_terms": sorted(set(matched_any + matched_all)),
        "missing_terms": missing_all,
        "citation_present": citation_present,
        "source_section_hit": source_section_hit,
        "term_match_rate": _term_match_rate(expected_any, expected_all, matched_any, matched_all),
        "issues": issues,
    }


def _build_report(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for result in case_results if result["status"] == "passed")
    failed = sum(1 for result in case_results if result["status"] == "failed")
    skipped = sum(1 for result in case_results if result["status"] == "skipped")
    runnable = [result for result in case_results if result["status"] != "skipped"]
    citation_rate = _rate(runnable, "citation_present")
    source_section_hit_rate = _rate(runnable, "source_section_hit")
    term_match_rate = (
        sum(float(result.get("term_match_rate", 0.0)) for result in runnable) / len(runnable)
        if runnable
        else 0.0
    )
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "skipped_cases": skipped,
        "citation_rate": citation_rate,
        "source_section_hit_rate": source_section_hit_rate,
        "term_match_rate": term_match_rate,
        "case_results": case_results,
    }


def _failed(case: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id", "unknown_case"),
        "status": "failed",
        "question": case.get("question", ""),
        "collection_name": case.get("collection_name"),
        "answer": "",
        "answer_status": "",
        "sources": [],
        "matched_terms": [],
        "missing_terms": [],
        "citation_present": False,
        "source_section_hit": False,
        "term_match_rate": 0.0,
        "issues": issues,
    }


def _skipped(case: dict[str, Any], message: str) -> dict[str, Any]:
    result = _failed(case, [message])
    result["status"] = "skipped"
    return result


def _summarize_source(index: int, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": index,
        "source_number": source.get("source_number"),
        "chunk_id": source.get("chunk_id"),
        "section_item": source.get("section_item"),
        "chunk_type": source.get("chunk_type"),
        "page_start": source.get("page_start"),
        "page_end": source.get("page_end"),
        "score": source.get("score"),
        "preview": source.get("preview"),
    }


def _term_match_rate(
    expected_any: list[str],
    expected_all: list[str],
    matched_any: list[str],
    matched_all: list[str],
) -> float:
    expected = set(expected_any + expected_all)
    if not expected:
        return 1.0
    matched = set(matched_any + matched_all)
    return len(matched) / len(expected)


def _rate(results: list[dict[str, Any]], field: str) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.get(field)) / len(results)


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    print("Answer Golden Eval")
    print(f"- Total: {report['total_cases']}")
    print(f"- Passed: {report['passed_cases']}")
    print(f"- Failed: {report['failed_cases']}")
    print(f"- Skipped: {report['skipped_cases']}")
    print(f"- Citation rate: {report['citation_rate']:.1%}")
    print(f"- Source section hit rate: {report['source_section_hit_rate']:.1%}")
    print(f"- Term match rate: {report['term_match_rate']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden answer evaluation.")
    parser.add_argument("--cases", required=True, help="Path to answer golden cases JSON.")
    parser.add_argument("--output", help="Optional path to write report JSON.")
    args = parser.parse_args()

    report = run_answer_golden_eval(args.cases)
    if args.output:
        _write_report(args.output, report)
    _print_summary(report)


if __name__ == "__main__":
    main()
