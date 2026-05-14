from app.rag.keyword_retriever import retrieve_from_chunk_records, tokenize_query


def test_tokenize_query_removes_stopwords():
    assert tokenize_query("What is net cash from operating activities?") == [
        "is",
        "net",
        "cash",
        "operating",
        "activities",
    ]


def test_retrieve_from_chunk_records_scores_matching_chunks():
    chunks = [
        {
            "page_content": "Net cash provided by operating activities was reported in the cash flow table.",
            "metadata": {"chunk_id": "table-1", "chunk_type": "table", "section_item": "7"},
        },
        {
            "page_content": "Risk factors include competition.",
            "metadata": {"chunk_id": "text-1", "chunk_type": "section_text", "section_item": "1A"},
        },
    ]

    results = retrieve_from_chunk_records(chunks, "net cash operating activities", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk"]["metadata"]["chunk_id"] == "table-1"
    assert results[0]["score"] > 0


def test_retrieve_from_chunk_records_blank_query_returns_empty():
    assert retrieve_from_chunk_records([], "   ", top_k=5) == []
