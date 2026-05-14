from langchain_core.documents import Document

from app.rag import hybrid_retriever


def _doc(chunk_id: str, content: str, section_item: str = "7") -> Document:
    return Document(
        page_content=content,
        metadata={"chunk_id": chunk_id, "section_item": section_item, "chunk_type": "section_text"},
    )


def test_retrieve_candidates_blank_query_returns_empty():
    assert hybrid_retriever.retrieve_candidates("   ", "collection") == []


def test_vector_mode_returns_vector_candidates(monkeypatch):
    monkeypatch.setattr(
        hybrid_retriever,
        "similarity_search_with_score",
        lambda query, collection_name, k, metadata_filter: [(_doc("c1", "net sales growth"), 0.1)],
    )

    candidates = hybrid_retriever.retrieve_candidates(
        "net sales",
        "collection",
        candidate_k=5,
        metadata_filter={"section_item": "7"},
        retrieval_mode="vector",
    )

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "c1"
    assert candidates[0].vector_score == 0.1
    assert candidates[0].hybrid_score == 1.0


def test_keyword_mode_scores_collection_documents(monkeypatch):
    calls = {}

    def fake_get_collection_documents(collection_name, metadata_filter=None, limit=None):
        calls["metadata_filter"] = metadata_filter
        return [_doc("c1", "net sales growth"), _doc("c2", "unrelated")]

    monkeypatch.setattr(hybrid_retriever, "get_collection_documents", fake_get_collection_documents)

    candidates = hybrid_retriever.retrieve_candidates(
        "net sales",
        "collection",
        metadata_filter={"section_item": "7"},
        retrieval_mode="keyword",
    )

    assert calls["metadata_filter"] == {"section_item": "7"}
    assert [candidate.chunk_id for candidate in candidates] == ["c1"]
    assert candidates[0].keyword_score is not None


def test_hybrid_merges_duplicate_chunk_id(monkeypatch):
    monkeypatch.setattr(
        hybrid_retriever,
        "similarity_search_with_score",
        lambda query, collection_name, k, metadata_filter: [(_doc("same", "net sales growth"), 0.1)],
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "get_collection_documents",
        lambda collection_name, metadata_filter=None, limit=None: [_doc("same", "net sales growth")],
    )

    candidates = hybrid_retriever.retrieve_candidates("net sales", "collection", retrieval_mode="hybrid")

    assert len(candidates) == 1
    assert candidates[0].vector_score == 0.1
    assert candidates[0].keyword_score is not None
    assert candidates[0].hybrid_score is not None


def test_metadata_filter_applies_to_vector(monkeypatch):
    calls = {}

    def fake_similarity_search_with_score(query, collection_name, k, metadata_filter):
        calls["metadata_filter"] = metadata_filter
        return [(_doc("c1", "net sales growth"), 0.1)]

    monkeypatch.setattr(hybrid_retriever, "similarity_search_with_score", fake_similarity_search_with_score)

    hybrid_retriever.retrieve_candidates(
        "net sales",
        "collection",
        metadata_filter={"section_item": "7"},
        retrieval_mode="vector",
    )

    assert calls["metadata_filter"] == {"section_item": "7"}
