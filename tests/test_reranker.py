from langchain_core.documents import Document

from app.rag.hybrid_retriever import RetrievedCandidate
from app.rag.reranker import rerank_candidates


def _candidate(chunk_id: str, content: str, section_item: str = "1", chunk_type: str = "section_text") -> RetrievedCandidate:
    return RetrievedCandidate(
        document=Document(
            page_content=content,
            metadata={"chunk_id": chunk_id, "section_item": section_item, "chunk_type": chunk_type},
        ),
        chunk_id=chunk_id,
        hybrid_score=0.5,
    )


def test_exact_term_boost_ranks_expected_chunk_higher():
    candidates = [
        _candidate("bad", "generic business overview"),
        _candidate("good", "net sales revenue growth and margin details"),
    ]

    results = rerank_candidates("net sales growth", candidates, top_k=2)

    assert results[0][0].metadata["chunk_id"] == "good"


def test_section_filter_boost_works():
    candidates = [
        _candidate("item1", "net sales", section_item="1"),
        _candidate("item7", "net sales", section_item="7"),
    ]

    results = rerank_candidates(
        "net sales",
        candidates,
        top_k=2,
        metadata_filter={"section_item": "7"},
    )

    assert results[0][0].metadata["chunk_id"] == "item7"


def test_table_boost_and_top_k_respected():
    candidates = [
        _candidate("text", "cash flow statement", section_item="8", chunk_type="section_text"),
        _candidate("table", "cash flow statement", section_item="8", chunk_type="table"),
    ]

    results = rerank_candidates("cash flow table", candidates, top_k=1)

    assert len(results) == 1
    assert results[0][0].metadata["chunk_id"] == "table"
