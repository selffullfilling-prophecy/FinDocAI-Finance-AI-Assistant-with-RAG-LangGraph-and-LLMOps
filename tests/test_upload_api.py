from pathlib import Path

from fastapi.testclient import TestClient

from app.api import routes_upload
from app.main import app


class DummySettings:
    def __init__(self, root: Path) -> None:
        self.raw_data_dir = str(root / "raw")
        self.processed_data_dir = str(root / "processed")


def test_upload_document_chunks_txt_and_writes_processed_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_upload, "get_settings", lambda: DummySettings(tmp_path))
    client = TestClient(app)

    content = (
        "Item 1. Business\n"
        "The company sells software and cloud services. " * 40
        + "\n\nItem 7. Management Discussion and Analysis\n"
        + "Revenue increased because of higher demand. " * 40
    )

    response = client.post(
        "/upload",
        files={"file": ("sample-10k.txt", content.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["file_name"] == "sample-10k.txt"
    assert body["total_chunks"] > 0
    assert body["indexed"] is False
    assert body["collection_name"] == "findoc_sample-10k"

    processed_path = Path(body["processed_path"])
    latest_processed_path = Path(body["latest_processed_path"])
    eval_report_path = Path(body["eval_report_path"])
    assert processed_path.exists()
    assert latest_processed_path.exists()
    assert eval_report_path.exists()
    assert '"chunk_id"' in processed_path.read_text(encoding="utf-8")
