from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.rag.answer_service import answer_question


def run_ragas_eval(cases_path: str | Path) -> dict[str, Any]:
    cases_path = Path(cases_path)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    if os.getenv("RUN_RAGAS_EVAL") != "1":
        return _skipped_report(cases, "RUN_RAGAS_EVAL is not set to 1; skipping optional RAGAS eval.")
    if not get_settings().nvidia_api_key:
        return _skipped_report(cases, "NVIDIA_API_KEY is not configured; skipping optional RAGAS eval.")
    if importlib.util.find_spec("ragas") is None or importlib.util.find_spec("datasets") is None:
        return _skipped_report(cases, "RAGAS dependencies are not installed. Run: pip install -r requirements-eval.txt")

    rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        result = _run_case(case)
        case_results.append(result)
        if result["status"] != "completed":
            continue
        rows.append(
            {
                "question": result["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": result["ground_truth"],
            }
        )

    if not rows:
        return {
            "total_cases": len(cases),
            "completed_cases": 0,
            "skipped_cases": sum(1 for result in case_results if result["status"] == "skipped"),
            "failed_cases": sum(1 for result in case_results if result["status"] == "failed"),
            "metrics": {},
            "case_results": case_results,
            "issues": ["No completed RAGAS rows were available."],
        }

    try:
        metrics = _evaluate_with_ragas(rows)
    except Exception as exc:
        return {
            "total_cases": len(cases),
            "completed_cases": len(rows),
            "skipped_cases": 0,
            "failed_cases": len(cases) - len(rows),
            "metrics": {},
            "case_results": case_results,
            "issues": [f"RAGAS evaluation failed: {exc}"],
        }

    return {
        "total_cases": len(cases),
        "completed_cases": len(rows),
        "skipped_cases": sum(1 for result in case_results if result["status"] == "skipped"),
        "failed_cases": sum(1 for result in case_results if result["status"] == "failed"),
        "metrics": metrics,
        "case_results": case_results,
        "issues": [],
    }


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    question = str(case.get("question", "")).strip()
    collection_name = str(case.get("collection_name", "")).strip()
    if not question or not collection_name:
        return _case_result(case, "failed", ["Question and collection_name are required."])

    try:
        response = answer_question(
            question=question,
            collection_name=collection_name,
            top_k=int(case.get("top_k", 5)),
            candidate_k=int(case.get("candidate_k", 20)),
            metadata_filter=case.get("metadata_filter"),
            retrieval_mode=case.get("retrieval_mode", "hybrid"),
            rerank=bool(case.get("rerank", True)),
            session_id=case.get("session_id") or f"ragas-{case.get('case_id', 'unknown')}",
            use_memory=False,
        )
    except Exception as exc:
        return _case_result(case, "failed", [f"Answer generation failed: {exc}"])

    contexts = [
        str(source.get("preview", "")).strip()
        for source in response.get("retrieved_context", []) or []
        if str(source.get("preview", "")).strip()
    ]
    issues = []
    if response.get("answer_status") != "answered":
        issues.append(f"answer_status is {response.get('answer_status')}; expected answered for RAGAS case.")

    result = _case_result(case, "completed", issues)
    result.update(
        {
            "question": question,
            "answer": response.get("answer", ""),
            "answer_status": response.get("answer_status"),
            "contexts": contexts,
            "ground_truth": case.get("ground_truth", ""),
            "source_count": len(response.get("sources", []) or []),
            "retrieved_context_count": len(contexts),
        }
    )
    return result


def _evaluate_with_ragas(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
    )
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    return {str(key): _as_float_or_none(value) for key, value in scores.items()}


def _skipped_report(cases: list[dict[str, Any]], message: str) -> dict[str, Any]:
    return {
        "total_cases": len(cases),
        "completed_cases": 0,
        "skipped_cases": len(cases),
        "failed_cases": 0,
        "metrics": {},
        "case_results": [_case_result(case, "skipped", [message]) for case in cases],
        "issues": [message],
    }


def _case_result(case: dict[str, Any], status: str, issues: list[str]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id", "unknown_case"),
        "status": status,
        "question": case.get("question", ""),
        "collection_name": case.get("collection_name"),
        "issues": issues,
    }


def _as_float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    print("RAGAS Eval")
    print(f"- Total: {report['total_cases']}")
    print(f"- Completed: {report['completed_cases']}")
    print(f"- Failed: {report['failed_cases']}")
    print(f"- Skipped: {report['skipped_cases']}")
    for metric, value in report.get("metrics", {}).items():
        if value is not None:
            print(f"- {metric}: {value:.3f}")
    for issue in report.get("issues", []):
        print(f"- Issue: {issue}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional RAGAS evaluation.")
    parser.add_argument("--cases", required=True, help="Path to RAGAS golden cases JSON.")
    parser.add_argument("--output", help="Optional path to write report JSON.")
    args = parser.parse_args()

    report = run_ragas_eval(args.cases)
    if args.output:
        _write_report(args.output, report)
    _print_summary(report)


if __name__ == "__main__":
    main()
