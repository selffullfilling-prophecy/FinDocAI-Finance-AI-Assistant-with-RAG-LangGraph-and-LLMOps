from app.rag.eval import answer_eval


def _case(**overrides):
    base = {
        "case_id": "case",
        "question": "What drove net sales?",
        "collection_name": "collection",
        "expected_terms_any": ["net sales", "iPhone"],
        "expected_terms_all": [],
        "expected_source_sections": ["7"],
        "require_citation": True,
        "allow_insufficient_context": False,
        "forbidden_terms": ["probably"],
    }
    base.update(overrides)
    return base


def _response(answer="Net sales grew because of iPhone. [Source 1]", answer_status="answered", sources=None):
    return {
        "answer": answer,
        "answer_status": answer_status,
        "sources": sources if sources is not None else [{"chunk_id": "c1", "section_item": "7", "chunk_type": "section_text"}],
        "retrieved_context": [{"chunk_id": "related", "section_item": "1A"}],
    }


def test_answer_eval_expected_term_match_passes():
    result = answer_eval.evaluate_answer_result(_case(), _response())

    assert result["status"] == "passed"
    assert result["citation_present"] is True
    assert result["source_section_hit"] is True


def test_answer_eval_missing_citation_fails():
    result = answer_eval.evaluate_answer_result(_case(), _response("Net sales grew because of iPhone.", sources=[]))

    assert result["status"] == "failed"
    assert "Citation is required" in result["issues"][0]


def test_answer_eval_forbidden_term_fails():
    result = answer_eval.evaluate_answer_result(
        _case(),
        _response("Net sales probably grew because of iPhone. [Source 1]"),
    )

    assert result["status"] == "failed"
    assert "forbidden" in result["issues"][0].lower()


def test_answer_eval_insufficient_context_requires_empty_sources():
    result = answer_eval.evaluate_answer_result(
        _case(
            allow_insufficient_context=True,
            require_citation=False,
            expected_answer_status="insufficient_context",
            expected_terms_any=[],
            expected_source_sections=[],
        ),
        _response(
            "The provided documents do not contain enough information to answer this question.",
            answer_status="insufficient_context",
            sources=[],
        ),
    )

    assert result["status"] == "passed"


def test_answer_eval_insufficient_context_with_sources_fails():
    result = answer_eval.evaluate_answer_result(
        _case(allow_insufficient_context=True, require_citation=False),
        _response(
            "The provided documents do not contain enough information to answer this question.",
            answer_status="insufficient_context",
            sources=[{"chunk_id": "bad", "section_item": "7"}],
        ),
    )

    assert result["status"] == "failed"
    assert "must not return sources" in " ".join(result["issues"])


def test_answer_eval_expected_source_sections_checks_sources_not_retrieved_context():
    result = answer_eval.evaluate_answer_result(
        _case(expected_source_sections=["7"]),
        _response(
            "Net sales grew because of iPhone. [Source 1]",
            sources=[],
        ),
    )

    assert result["status"] == "failed"
    assert "No source section matched" in " ".join(result["issues"])


def test_answer_eval_skips_when_run_llm_eval_not_set(monkeypatch):
    monkeypatch.delenv("RUN_LLM_EVAL", raising=False)

    result = answer_eval.evaluate_answer_case(_case())

    assert result["status"] == "skipped"


def test_answer_eval_report_metrics():
    report = answer_eval._build_report(
        [
            {"status": "passed", "citation_present": True, "source_section_hit": True, "term_match_rate": 1.0},
            {"status": "failed", "citation_present": False, "source_section_hit": True, "term_match_rate": 0.5},
            {"status": "skipped", "citation_present": False, "source_section_hit": False, "term_match_rate": 0.0},
        ]
    )

    assert report["total_cases"] == 3
    assert report["citation_rate"] == 0.5
    assert report["source_section_hit_rate"] == 1.0
    assert report["term_match_rate"] == 0.75
