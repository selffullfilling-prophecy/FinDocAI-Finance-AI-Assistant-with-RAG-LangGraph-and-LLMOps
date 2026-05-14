import json

from app.rag.eval import ragas_eval


def test_ragas_eval_skips_when_env_not_set(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"case_id": "case", "question": "Q", "collection_name": "c"}]), encoding="utf-8")
    monkeypatch.delenv("RUN_RAGAS_EVAL", raising=False)

    report = ragas_eval.run_ragas_eval(cases_path)

    assert report["total_cases"] == 1
    assert report["skipped_cases"] == 1
    assert "RUN_RAGAS_EVAL" in report["issues"][0]


def test_ragas_eval_skips_when_dependencies_missing(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"case_id": "case", "question": "Q", "collection_name": "c"}]), encoding="utf-8")
    monkeypatch.setenv("RUN_RAGAS_EVAL", "1")
    monkeypatch.setattr(
        ragas_eval,
        "get_settings",
        lambda: type("Settings", (), {"nvidia_api_key": "key"})(),
    )
    monkeypatch.setattr(ragas_eval.importlib.util, "find_spec", lambda name: None)

    report = ragas_eval.run_ragas_eval(cases_path)

    assert report["skipped_cases"] == 1
    assert "requirements-eval.txt" in report["issues"][0]
