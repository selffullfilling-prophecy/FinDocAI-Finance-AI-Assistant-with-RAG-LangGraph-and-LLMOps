from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORTS = {
    "chunking": Path("data/eval/chunking_golden_report.json"),
    "retriever": Path("data/eval/retriever_golden_report.json"),
    "answer": Path("data/eval/answer_golden_report.json"),
    "ragas": Path("data/eval/ragas_report.json"),
}


def build_benchmark_summary(
    output_path: str | Path = "data/eval/benchmark_summary.md",
    report_paths: dict[str, Path] | None = None,
) -> str:
    report_paths = report_paths or DEFAULT_REPORTS
    reports = {name: _load_json(path) for name, path in report_paths.items()}
    markdown = _render_markdown(reports, report_paths)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return markdown


def _render_markdown(reports: dict[str, dict[str, Any] | None], report_paths: dict[str, Path]) -> str:
    lines = [
        "# FinDocAI MVP Benchmark Summary",
        "",
        "## Chunking Eval",
        *_summary_lines(reports.get("chunking"), report_paths["chunking"]),
        "",
        "## Retriever Eval",
        *_summary_lines(reports.get("retriever"), report_paths["retriever"], extra_fields=["hit_at_k", "mrr_at_k"]),
        "",
        "## Answer Golden Eval",
        *_summary_lines(
            reports.get("answer"),
            report_paths["answer"],
            extra_fields=[
                "citation_rate",
                "source_section_hit_rate",
                "term_match_rate",
                "answer_status_accuracy",
                "insufficient_handling_rate",
            ],
        ),
        "",
        "## RAGAS Eval",
        *_ragas_lines(reports.get("ragas"), report_paths["ragas"]),
        "",
        "## Known limitations",
        "- Table extraction is not production-grade.",
        "- Follow-up query rewriting is heuristic-first and needs a larger benchmark set.",
        "- Benchmark coverage is intentionally small for MVP v0.1.",
        "- Memory is in-memory only and not suitable for production persistence.",
        "",
    ]
    return "\n".join(lines)


def _summary_lines(report: dict[str, Any] | None, path: Path, extra_fields: list[str] | None = None) -> list[str]:
    if not report:
        return [f"- Report not found: `{path}`"]

    lines = [
        f"- Total: {report.get('total_cases', 0)}",
        f"- Passed: {report.get('passed_cases', 0)}",
        f"- Failed: {report.get('failed_cases', 0)}",
        f"- Skipped: {report.get('skipped_cases', 0)}",
    ]
    for field in extra_fields or []:
        if field in report:
            lines.append(f"- {_label(field)}: {_format_value(report.get(field))}")
    return lines


def _ragas_lines(report: dict[str, Any] | None, path: Path) -> list[str]:
    if not report:
        return [f"- Report not found: `{path}`"]

    lines = [
        f"- Total: {report.get('total_cases', 0)}",
        f"- Completed: {report.get('completed_cases', 0)}",
        f"- Failed: {report.get('failed_cases', 0)}",
        f"- Skipped: {report.get('skipped_cases', 0)}",
    ]
    metrics = report.get("metrics", {}) or {}
    for field in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]:
        if field in metrics:
            lines.append(f"- {_label(field)}: {_format_value(metrics.get(field))}")
    if not metrics and report.get("issues"):
        lines.append(f"- Issue: {report['issues'][0]}")
    return lines


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _label(field: str) -> str:
    return field.replace("_", " ").title()


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.1%}" if 0.0 <= value <= 1.0 else f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FinDocAI benchmark summary markdown.")
    parser.add_argument("--output", default="data/eval/benchmark_summary.md", help="Output markdown path.")
    args = parser.parse_args()
    markdown = build_benchmark_summary(args.output)
    print(markdown)


if __name__ == "__main__":
    main()
