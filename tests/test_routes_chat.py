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


def test_chat_route_blank_collection_returns_400():
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"question": "What drove net sales?", "collection_name": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Collection name is required."


def test_chat_route_valid_request_returns_chat_response(monkeypatch):
    def fake_answer_question(
        question,
        collection_name,
        top_k,
        candidate_k,
        metadata_filter,
        retrieval_mode,
        rerank,
        session_id,
        use_memory,
    ):
        assert question == "What drove net sales?"
        assert collection_name == "test_collection"
        assert top_k == 5
        assert candidate_k == 20
        assert metadata_filter == {"section_item": "7"}
        assert retrieval_mode == "hybrid"
        assert rerank is True
        assert session_id == "session-1"
        assert use_memory is True
        return {
            "answer": "Net sales were driven by Services. [Source 1]",
            "collection_name": collection_name,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "retrieval_mode": retrieval_mode,
            "rerank": rerank,
            "session_id": session_id,
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
            "debug": {"candidate_count": 1},
        }

    monkeypatch.setattr(routes_chat, "answer_question", fake_answer_question)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "question": "What drove net sales?",
            "collection_name": "test_collection",
            "metadata_filter": {"section_item": "7"},
            "session_id": "session-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Net sales were driven by Services. [Source 1]"
    assert body["collection_name"] == "test_collection"
    assert body["top_k"] == 5
    assert body["candidate_k"] == 20
    assert body["retrieval_mode"] == "hybrid"
    assert body["sources"][0]["chunk_id"] == "item-7-text-001"


def test_chat_route_value_error_returns_400(monkeypatch):
    def fake_answer_question(**kwargs):
        raise ValueError("NVIDIA_API_KEY is not configured.")

    monkeypatch.setattr(routes_chat, "answer_question", fake_answer_question)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"question": "What drove net sales?", "collection_name": "test_collection"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "NVIDIA_API_KEY is not configured."


def test_chat_stream_route_returns_sse_events(monkeypatch):
    def fake_stream_answer_question(**kwargs):
        yield {"type": "token", "content": "Answer"}
        yield {"type": "sources", "sources": []}
        yield {"type": "done"}

    monkeypatch.setattr(routes_chat, "stream_answer_question", fake_stream_answer_question)
    client = TestClient(app)

    response = client.post(
        "/chat/stream",
        json={"question": "What drove net sales?", "collection_name": "test_collection"},
    )

    assert response.status_code == 200
    assert 'data: {"type": "token", "content": "Answer"}' in response.text
    assert 'data: {"type": "done"}' in response.text


def test_clear_memory_endpoint():
    client = TestClient(app)

    response = client.delete("/chat/sessions/unit-test-clear")

    assert response.status_code == 200
    assert response.json() == {"session_id": "unit-test-clear", "status": "cleared"}
