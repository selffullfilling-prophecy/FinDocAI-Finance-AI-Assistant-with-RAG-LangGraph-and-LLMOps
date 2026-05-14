import json

from app.rag.eval.chunking_eval import (
    evaluate_chunk_records_against_case,
    run_chunking_golden_eval,
)


def _chunk(chunk_id="chunk-1", section_item="7", chunk_type="section_text", content="net sales gross margin"):
    return {
        "page_content": content,
        "metadata": {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "section_item": section_item,
            "section_title": "Section",
            "page_start": 1,
        },
    }


def test_chunking_eval_missing_document_is_skipped(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "missing",
                    "document_path": str(tmp_path / "missing.pdf"),
                    "expected_sections": ["7"],
                }
            ]
        ),
        encoding="utf-8",
    )

    report = run_chunking_golden_eval(cases_path)

    assert report["skipped_cases"] == 1
    assert report["case_results"][0]["status"] == "skipped"


def test_chunking_eval_missing_expected_section_fails():
    case = {"case_id": "missing_section", "expected_sections": ["7", "8"]}

    result = evaluate_chunk_records_against_case(case, [_chunk(section_item="7")], "mock")

    assert result["status"] == "failed"
    assert any("Missing expected sections" in issue for issue in result["issues"])


def test_chunking_eval_missing_required_metadata_fails():
    case = {"case_id": "missing_meta", "required_metadata_fields": ["chunk_id", "section_title"]}
    chunks = [{"page_content": "text", "metadata": {"chunk_id": "chunk-1"}}]

    result = evaluate_chunk_records_against_case(case, chunks, "mock")

    assert result["status"] == "failed"
    assert any("Missing required metadata" in issue for issue in result["issues"])


def test_chunking_eval_duplicate_chunk_id_fails():
    case = {"case_id": "duplicates"}

    result = evaluate_chunk_records_against_case(
        case,
        [_chunk(chunk_id="dup"), _chunk(chunk_id="dup", section_item="8")],
        "mock",
    )

    assert result["status"] == "failed"
    assert any("Duplicate chunk_id" in issue for issue in result["issues"])


def test_chunking_eval_valid_small_mock_passes():
    case = {
        "case_id": "valid",
        "expected_sections": ["7", "8"],
        "required_metadata_fields": ["chunk_id", "chunk_type", "section_item", "section_title"],
        "min_table_chunks": 1,
        "must_contain": [{"section_item": "7", "terms": ["net sales"]}],
    }
    chunks = [
        _chunk(chunk_id="text-7", section_item="7", content="net sales increased"),
        _chunk(chunk_id="table-8", section_item="8", chunk_type="table", content="cash flows"),
    ]

    result = evaluate_chunk_records_against_case(case, chunks, "mock")

    assert result["status"] == "passed"
