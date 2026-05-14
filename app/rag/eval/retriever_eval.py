from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.rag.chunk_artifacts import load_chunks_jsonl
from app.rag.keyword_retriever import retrieve_from_chunk_records
from app.rag.vector_store import read_vector_store_manifest, similarity_search_with_score


def run_retriever_golden_eval(cases_path: str | Path) -> dict[str, Any]:
    cases_path = Path(cases_path)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    case_results = [evaluate_retriever_case(case) for case in cases]
    return _build_report(case_results)


def evaluate_retriever_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case.get("case_id", "unknown_case")
    question = str(case.get("question", "")).strip()
    top_k = int(case.get("top_k", 5))
    retriever_type = case.get("retriever_type", "vector")

    if not question:
        return _failed(case, ["Question is required."])

    try:
        if retriever_type == "keyword_jsonl":
            results = _run_keyword_retrieval(case, question, top_k)
        elif retriever_type == "vector":
            results = _run_vector_retrieval(case, question, top_k)
        else:
            return _failed(case, [f"Unsupported retriever_type: {retriever_type}."])
    except FileNotFoundError as exc:
        return _skipped(case, str(exc))
    except ValueError as exc:
        message = str(exc)
        if _is_skippable_vector_error(message):
            return _skipped(case, message)
        return _failed(case, [message])
    except Exception as exc:
        message = str(exc)
        if _is_skippable_vector_error(message):
            return _skipped(case, message)
        return _failed(case, [f"Retriever failed: {message}"])

    return evaluate_retrieval_results(case, results)


def evaluate_retrieval_results(
    case: dict[str, Any],
    results: list[tuple[dict[str, Any], float | None]],
) -> dict[str, Any]:
    issues: list[str] = []
    first_match_rank: int | None = None
    matched_terms: list[str] = []

    if not results:
        issues.append("Retriever returned no results.")

    retrieved_summaries = [
        _summarize_retrieved_chunk(rank, chunk, score)
        for rank, (chunk, score) in enumerate(results, start=1)
    ]

    for rank, (chunk, _score) in enumerate(results, start=1):
        result_match, terms = _matches_expected_result(case, chunk)
        if result_match:
            first_match_rank = rank
            matched_terms = terms
            break

    hit = first_match_rank is not None
    mrr = 1.0 / first_match_rank if first_match_rank else 0.0
    term_match_rate = _term_match_rate(case, results)

    if not hit:
        issues.append("No retrieved chunk matched expected section/type/terms.")

    status = "passed" if hit and not issues else "failed"
    return {
        "case_id": case.get("case_id", "unknown_case"),
        "status": status,
        "question": case.get("question", ""),
        "collection_name": case.get("collection_name"),
        "retriever_type": case.get("retriever_type", "vector"),
        "top_k": int(case.get("top_k", 5)),
        "first_match_rank": first_match_rank,
        "hit": hit,
        "mrr": mrr,
        "term_match_rate": term_match_rate,
        "matched_terms": matched_terms,
        "retrieved_chunks": retrieved_summaries,
        "issues": issues,
    }


def _run_keyword_retrieval(
    case: dict[str, Any],
    question: str,
    top_k: int,
) -> list[tuple[dict[str, Any], float | None]]:
    chunks_path = case.get("chunks_path")
    if not chunks_path:
        raise ValueError("keyword_jsonl cases require chunks_path.")

    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = load_chunks_jsonl(path)
    scored = retrieve_from_chunk_records(chunks, question, top_k=top_k)
    return [(item["chunk"], float(item["score"])) for item in scored]


def _run_vector_retrieval(
    case: dict[str, Any],
    question: str,
    top_k: int,
) -> list[tuple[dict[str, Any], float | None]]:
    collection_name = case.get("collection_name")
    if not collection_name:
        raise ValueError("vector cases require collection_name.")

    if read_vector_store_manifest(collection_name) is None:
        raise ValueError(f"Collection '{collection_name}' is not indexed. Build or rebuild it before vector eval.")

    scored_documents = similarity_search_with_score(
        query=question,
        collection_name=collection_name,
        k=top_k,
        metadata_filter=case.get("metadata_filter"),
    )
    return [
        (
            {"page_content": document.page_content, "metadata": document.metadata},
            score,
        )
        for document, score in scored_documents
    ]


def _matches_expected_result(case: dict[str, Any], chunk: dict[str, Any]) -> tuple[bool, list[str]]:
    metadata = chunk.get("metadata", {})
    content_and_metadata = _content_and_metadata_text(chunk)
    expected_section = case.get("expected_section")
    expected_chunk_type = case.get("expected_chunk_type")

    if expected_section is not None and str(metadata.get("section_item")) != str(expected_section):
        return False, []

    if expected_chunk_type is not None and str(metadata.get("chunk_type")) != str(expected_chunk_type):
        return False, []

    all_terms = [str(term) for term in case.get("expected_terms_all", [])]
    any_terms = [str(term) for term in case.get("expected_terms_any", [])]
    matched_all = [term for term in all_terms if term.lower() in content_and_metadata]
    matched_any = [term for term in any_terms if term.lower() in content_and_metadata]

    if all_terms and len(matched_all) != len(all_terms):
        return False, matched_all + matched_any

    if any_terms and not matched_any:
        return False, matched_all

    return True, sorted(set(matched_all + matched_any))


def _term_match_rate(
    case: dict[str, Any],
    results: list[tuple[dict[str, Any], float | None]],
) -> float:
    expected_terms = {
        str(term).lower()
        for term in case.get("expected_terms_all", []) + case.get("expected_terms_any", [])
    }
    if not expected_terms:
        return 1.0

    combined = "\n".join(_content_and_metadata_text(chunk) for chunk, _score in results)
    matched = {term for term in expected_terms if term in combined}
    return len(matched) / len(expected_terms)


def _content_and_metadata_text(chunk: dict[str, Any]) -> str:
    return (
        chunk.get("page_content", "")
        + "\n"
        + json.dumps(chunk.get("metadata", {}), ensure_ascii=False)
    ).lower()


def _summarize_retrieved_chunk(
    rank: int,
    chunk: dict[str, Any],
    score: float | None,
) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    preview = " ".join(chunk.get("page_content", "").split())[:240]
    return {
        "rank": rank,
        "score": score,
        "chunk_id": metadata.get("chunk_id"),
        "section_item": metadata.get("section_item"),
        "chunk_type": metadata.get("chunk_type"),
        "page_start": metadata.get("page_start") or metadata.get("page_number"),
        "page_end": metadata.get("page_end"),
        "short_preview": preview,
    }


def _build_report(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for result in case_results if result["status"] == "passed")
    failed = sum(1 for result in case_results if result["status"] == "failed")
    skipped = sum(1 for result in case_results if result["status"] == "skipped")
    runnable = [result for result in case_results if result["status"] != "skipped"]
    hit_at_k = sum(1 for result in runnable if result.get("hit")) / len(runnable) if runnable else 0.0
    mrr_at_k = sum(float(result.get("mrr", 0.0)) for result in runnable) / len(runnable) if runnable else 0.0
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "skipped_cases": skipped,
        "hit_at_k": hit_at_k,
        "mrr_at_k": mrr_at_k,
        "case_results": case_results,
    }


def _failed(case: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id", "unknown_case"),
        "status": "failed",
        "question": case.get("question", ""),
        "collection_name": case.get("collection_name"),
        "retriever_type": case.get("retriever_type", "vector"),
        "top_k": int(case.get("top_k", 5)),
        "first_match_rank": None,
        "hit": False,
        "mrr": 0.0,
        "term_match_rate": 0.0,
        "matched_terms": [],
        "retrieved_chunks": [],
        "issues": issues,
    }


def _skipped(case: dict[str, Any], message: str) -> dict[str, Any]:
    result = _failed(case, [message])
    result["status"] = "skipped"
    return result


def _is_skippable_vector_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in [
            "embedding model changed",
            "rebuild the vector store",
            "does not exist",
            "not indexed",
            "not found",
            "collection",
        ]
    )


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    print("Retriever Golden Eval")
    print(f"- Total: {report['total_cases']}")
    print(f"- Hit@k: {report['hit_at_k']:.1%}")
    print(f"- MRR@k: {report['mrr_at_k']:.3f}")
    print(f"- Passed: {report['passed_cases']}")
    print(f"- Failed: {report['failed_cases']}")
    print(f"- Skipped: {report['skipped_cases']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden retriever evaluation.")
    parser.add_argument("--cases", required=True, help="Path to retriever golden cases JSON.")
    parser.add_argument("--output", help="Optional path to write report JSON.")
    args = parser.parse_args()

    report = run_retriever_golden_eval(args.cases)
    if args.output:
        _write_report(args.output, report)
    _print_summary(report)


if __name__ == "__main__":
    main()
