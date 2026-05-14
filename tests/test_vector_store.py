from langchain_core.documents import Document

from app.rag import vector_store


class FakeVectorStore:
    def __init__(self) -> None:
        self.added_documents = None
        self.added_ids = None
        self.search_calls = []
        self.score_search_calls = []

    def add_documents(self, documents, ids):
        self.added_documents = documents
        self.added_ids = ids

    def similarity_search(self, query, k=5, filter=None):
        self.search_calls.append({"query": query, "k": k, "filter": filter})
        return [Document(page_content="match", metadata={"chunk_id": "chunk-1"})]

    def similarity_search_with_score(self, query, k=5, filter=None):
        self.score_search_calls.append({"query": query, "k": k, "filter": filter})
        return [(Document(page_content="match", metadata={"chunk_id": "chunk-1"}), 0.12)]


def test_similarity_search_returns_empty_for_blank_query(monkeypatch):
    def fail_get_vector_store(collection_name):
        raise AssertionError("blank query should not initialize vector store")

    monkeypatch.setattr(vector_store, "get_vector_store", fail_get_vector_store)

    assert vector_store.similarity_search("   ", "test_collection") == []


def test_index_documents_requires_chunk_id():
    documents = [Document(page_content="text", metadata={"section_item": "7"})]

    try:
        vector_store.index_documents(documents, "test_collection")
    except ValueError as exc:
        assert "index 0" in str(exc)
        assert "chunk_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing chunk_id")


def test_similarity_search_with_score_strips_query_and_passes_filter(monkeypatch):
    fake_store = FakeVectorStore()
    monkeypatch.setattr(vector_store, "get_vector_store", lambda collection_name: fake_store)
    monkeypatch.setattr(vector_store, "assert_embedding_config_matches", lambda collection_name: None)

    results = vector_store.similarity_search_with_score(
        "  revenue growth  ",
        "test_collection",
        k=3,
        metadata_filter={"section_item": "7"},
    )

    assert len(results) == 1
    assert fake_store.score_search_calls == [
        {"query": "revenue growth", "k": 3, "filter": {"section_item": "7"}}
    ]


def test_index_documents_cleans_metadata_and_uses_chunk_ids(monkeypatch):
    fake_store = FakeVectorStore()
    monkeypatch.setattr(vector_store, "get_vector_store", lambda collection_name: fake_store)
    monkeypatch.setattr(vector_store, "assert_embedding_config_matches", lambda collection_name: None)
    monkeypatch.setattr(
        vector_store,
        "write_vector_store_manifest",
        lambda collection_name, vector_count: {"collection_name": collection_name, "vector_count": vector_count},
    )
    documents = [
        Document(
            page_content="text",
            metadata={"chunk_id": "chunk-1", "page_start": None, "tags": ["risk", "revenue"]},
        )
    ]

    result = vector_store.index_documents(documents, "test_collection")

    assert result == {"collection_name": "test_collection", "vector_count": 1}
    assert fake_store.added_ids == ["chunk-1"]
    assert fake_store.added_documents[0].metadata == {
        "chunk_id": "chunk-1",
        "tags": "['risk', 'revenue']",
    }


def test_rebuild_collection_resets_then_indexes(monkeypatch):
    calls = []

    monkeypatch.setattr(
        vector_store,
        "reset_collection",
        lambda collection_name: calls.append(("reset", collection_name)) or {"collection_name": collection_name, "status": "deleted"},
    )
    monkeypatch.setattr(
        vector_store,
        "index_documents",
        lambda documents, collection_name: calls.append(("index", collection_name, len(documents)))
        or {"collection_name": collection_name, "vector_count": len(documents)},
    )

    result = vector_store.rebuild_collection(
        [Document(page_content="text", metadata={"chunk_id": "chunk-1"})],
        "test_collection",
    )

    assert result == {"collection_name": "test_collection", "vector_count": 1}
    assert calls == [("reset", "test_collection"), ("index", "test_collection", 1)]
