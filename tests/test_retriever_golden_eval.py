from app.rag.eval import retriever_eval


def _result(chunk_id="chunk-1", section_item="7", chunk_type="section_text", content="net sales growth"):
    return (
        {
            "page_content": content,
            "metadata": {
                "chunk_id": chunk_id,
                "section_item": section_item,
                "chunk_type": chunk_type,
                "page_start": 1,
            },
        },
        0.1,
    )


def test_retriever_eval_blank_question_fails():
    result = retriever_eval.evaluate_retriever_case(
        {"case_id": "blank", "question": "   ", "retriever_type": "keyword_jsonl"}
    )

    assert result["status"] == "failed"
    assert "Question is required." in result["issues"]


def test_retriever_eval_expected_section_passes(monkeypatch):
    monkeypatch.setattr(
        retriever_eval,
        "_run_vector_retrieval",
        lambda case, question, top_k: [_result(section_item="7", content="net sales growth")],
    )
    case = {
        "case_id": "hit",
        "question": "What drove revenue growth?",
        "collection_name": "test",
        "expected_section": "7",
        "expected_terms_any": ["net sales"],
    }

    result = retriever_eval.evaluate_retriever_case(case)

    assert result["status"] == "passed"
    assert result["hit"] is True
    assert result["first_match_rank"] == 1
    assert result["mrr"] == 1.0


def test_retriever_eval_missing_expected_terms_fails(monkeypatch):
    monkeypatch.setattr(
        retriever_eval,
        "_run_vector_retrieval",
        lambda case, question, top_k: [_result(section_item="7", content="unrelated content")],
    )
    case = {
        "case_id": "miss_terms",
        "question": "What drove revenue growth?",
        "collection_name": "test",
        "expected_section": "7",
        "expected_terms_any": ["net sales"],
    }

    result = retriever_eval.evaluate_retriever_case(case)

    assert result["status"] == "failed"
    assert result["hit"] is False


def test_retriever_eval_mrr_uses_first_matching_rank(monkeypatch):
    monkeypatch.setattr(
        retriever_eval,
        "_run_vector_retrieval",
        lambda case, question, top_k: [
            _result(chunk_id="bad", section_item="1A", content="risk factors"),
            _result(chunk_id="good", section_item="7", content="net sales growth"),
        ],
    )
    case = {
        "case_id": "rank_two",
        "question": "What drove revenue growth?",
        "collection_name": "test",
        "expected_section": "7",
        "expected_terms_any": ["net sales"],
    }

    result = retriever_eval.evaluate_retriever_case(case)

    assert result["status"] == "passed"
    assert result["first_match_rank"] == 2
    assert result["mrr"] == 0.5


def test_retriever_report_metrics():
    report = retriever_eval._build_report(
        [
            {"status": "passed", "hit": True, "mrr": 1.0},
            {"status": "failed", "hit": False, "mrr": 0.0},
            {"status": "skipped", "hit": False, "mrr": 0.0},
        ]
    )

    assert report["total_cases"] == 3
    assert report["hit_at_k"] == 0.5
    assert report["mrr_at_k"] == 0.5
