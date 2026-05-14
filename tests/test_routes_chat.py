from fastapi.testclient import TestClient

from app.api import routes_chat
from app.main import app


def test_chat_route_blank_question_returns_400():
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"question": "   ", "collection_name": "test_collection"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question is required."


def test_chat_route_valid_request_returns_chat_response(monkeypatch):
    def fake_answer_question(question, collection_name, top_k, metadata_filter):
        assert question == "What drove net sales?"
        assert collection_name == "test_collection"
        assert top_k == 5
        assert metadata_filter == {"section_item": "7"}
        return {
            "answer": "Net sales were driven by Services. [Source 1]",
            "collection_name": collection_name,
            "top_k": top_k,
            "sources": [
                {
                    "chunk_id": "item-7-text-001",
                    "section_item": "7",
                    "section_title": "Management's Discussion and Analysis",
                    "chunk_type": "section_text",
                    "page_start": 12,
                    "page_end": 13,
                    "score": 0.2,
                    "preview": "Net sales increased due to Services.",
                }
            ],
        }

    monkeypatch.setattr(routes_chat, "answer_question", fake_answer_question)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "question": "What drove net sales?",
            "collection_name": "test_collection",
            "metadata_filter": {"section_item": "7"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Net sales were driven by Services. [Source 1]"
    assert body["collection_name"] == "test_collection"
    assert body["top_k"] == 5
    assert body["sources"][0]["chunk_id"] == "item-7-text-001"


def test_chat_route_value_error_returns_400(monkeypatch):
    def fake_answer_question(question, collection_name, top_k, metadata_filter):
        raise ValueError("NVIDIA_API_KEY is not configured.")

    monkeypatch.setattr(routes_chat, "answer_question", fake_answer_question)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"question": "What drove net sales?", "collection_name": "test_collection"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "NVIDIA_API_KEY is not configured."
