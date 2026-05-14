import json

from app.rag.eval.benchmark_summary import build_benchmark_summary


def test_benchmark_summary_writes_markdown(tmp_path):
    chunking = tmp_path / "chunking.json"
    retriever = tmp_path / "retriever.json"
    answer = tmp_path / "answer.json"
    ragas = tmp_path / "ragas.json"
    output = tmp_path / "summary.md"

    chunking.write_text(
        json.dumps({"total_cases": 1, "passed_cases": 1, "failed_cases": 0, "skipped_cases": 0}),
        encoding="utf-8",
    )
    retriever.write_text(
        json.dumps(
            {
                "total_cases": 1,
                "passed_cases": 1,
                "failed_cases": 0,
                "skipped_cases": 0,
                "hit_at_k": 1.0,
                "mrr_at_k": 1.0,
            }
        ),
        encoding="utf-8",
    )
    answer.write_text(
        json.dumps(
            {
                "total_cases": 1,
                "passed_cases": 1,
                "failed_cases": 0,
                "skipped_cases": 0,
                "citation_rate": 1.0,
                "answer_status_accuracy": 1.0,
            }
        ),
        encoding="utf-8",
    )
    ragas.write_text(
        json.dumps({"total_cases": 1, "completed_cases": 1, "failed_cases": 0, "skipped_cases": 0, "metrics": {"faithfulness": 0.9}}),
        encoding="utf-8",
    )

    markdown = build_benchmark_summary(
        output,
        report_paths={
            "chunking": chunking,
            "retriever": retriever,
            "answer": answer,
            "ragas": ragas,
        },
    )

    assert output.exists()
    assert "# FinDocAI MVP Benchmark Summary" in markdown
    assert "Faithfulness: 90.0%" in markdown
