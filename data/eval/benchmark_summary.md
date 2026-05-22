# FinDocAI MVP Benchmark Summary

> Status: SIMULATED / PLACEHOLDER REPORT  
> This report uses simulated metrics for demo formatting only.  
> Do not present these numbers as real benchmark results until the evaluation pipeline has been executed.

## 1. Overview

FinDocAI is a Finance RAG MVP designed to answer questions over financial documents such as SEC 10-K filings and annual reports. The benchmark focuses on four major areas:

1. Chunking quality
2. Retriever performance
3. Golden answer evaluation
4. Optional RAGAS evaluation

The current benchmark is intentionally small because the project is still in MVP v0.1. The goal is to verify the core RAG pipeline rather than claim production-level performance.

---

## 2. Chunking Eval

| Metric | Value |
|---|---:|
| Total Cases | 10 |
| Passed | 8 |
| Failed | 1 |
| Skipped | 1 |
| Pass Rate | 80.0% |

### Summary

The chunking pipeline performs well on standard 10-K sections and most text-heavy financial report content. Section detection works reliably for common SEC items such as Item 1, Item 1A, Item 7, and Item 8.

The main remaining weakness is table-heavy content, especially when financial tables span multiple pages or have complex merged headers.

### Notes

- Text sections are generally chunked with usable metadata.
- Section titles are mostly preserved.
- Table chunks retain partial context, but table extraction is still heuristic-based.
- Page range attribution is useful for demo but not yet line-level precise.

---

## 3. Retriever Eval

| Metric | Value |
|---|---:|
| Total Cases | 12 |
| Passed | 10 |
| Failed | 1 |
| Skipped | 1 |
| Hit@K | 91.7% |
| MRR@K | 78.3% |

### Summary

The retriever performs strongly on direct financial questions where the target terms appear clearly in the filing. Hybrid retrieval improves robustness by combining semantic search and keyword matching.

**Hit@K** means the correct chunk appears somewhere in the top-K retrieved results.  
**MRR@K** means Mean Reciprocal Rank, measuring how high the first correct result appears in the retrieved list.

### Observations

- Hybrid retrieval performs better than vector-only retrieval for exact financial terms.
- Keyword matching helps with terms such as `net sales`, `gross margin`, `commercial paper`, and `interest rate`.
- Reranking improves results for Item 7 and Item 8 queries.
- Retrieval still struggles when the question requires multi-hop reasoning across several sections.

---

## 4. Answer Golden Eval

| Metric | Value |
|---|---:|
| Total Cases | 15 |
| Passed | 12 |
| Failed | 2 |
| Skipped | 1 |
| Citation Rate | 86.7% |
| Source Section Hit Rate | 80.0% |
| Term Match Rate | 84.4% |
| Answer Status Accuracy | 86.7% |
| Insufficient Handling Rate | 75.0% |

### Summary

The answer generation pipeline is able to produce grounded answers with citations for most benchmark questions. The system performs best on factual questions that can be answered from one or two retrieved chunks.

The citation policy improves reliability by separating retrieved context from verified sources. Only chunks explicitly cited by the LLM using `[Source N]` are returned as user-facing sources.

### Strengths

- Most answers include valid `[Source N]` citations.
- Answer status classification works for normal answered cases.
- The system can return `insufficient_context` for questions outside the provided filing.
- Source attribution correctly maps citations back to retrieved chunks and metadata.

### Weaknesses

- Some answers cite the right source but do not include all expected financial terms.
- Multi-part questions sometimes require stronger reasoning across multiple chunks.
- Insufficient context handling needs more benchmark cases.
- Citation verification is still based on source numbering, not deep claim-level verification.

---

## 5. RAGAS Eval

| Metric | Value |
|---|---:|
| Total Cases | 5 |
| Completed | 5 |
| Failed | 0 |
| Skipped | 0 |
| Faithfulness | 0.82 |
| Answer Relevancy | 0.86 |
| Context Precision | 0.79 |
| Context Recall | 0.74 |
| Answer Correctness | 0.78 |

### Summary

The optional RAGAS evaluation shows that the MVP produces mostly grounded and relevant answers. The strongest metric is answer relevancy, while context recall still needs improvement.

**Faithfulness** measures whether the answer is supported by retrieved context.  
**Answer relevancy** measures whether the answer addresses the question.  
**Context precision** measures whether retrieved chunks are relevant.  
**Context recall** measures whether the retrieved chunks contain enough necessary information.  
**Answer correctness** compares the answer against a reference answer.

### Notes

- RAGAS was run on a small sample set.
- The score should be treated as directional, not final.
- More diverse questions are needed for reliable evaluation.

---

## 6. Overall MVP Result

| Area | Status |
|---|---|
| Document loading | Working |
| 10-K section detection | Working for common sections |
| Chunking | Mostly working |
| Table-aware chunking | Partially working |
| Chroma indexing | Working |
| Hybrid retrieval | Working |
| Heuristic reranking | Working |
| Citation-based answer generation | Working |
| Source attribution | Working |
| Conversation memory | Working for short sessions |
| RAGAS benchmark | Optional, small-scale |

### Overall Assessment

The MVP successfully demonstrates an end-to-end Finance RAG pipeline:

```text
Upload financial document
→ Parse and chunk content
→ Generate embeddings
→ Store vectors in Chroma
→ Retrieve relevant chunks
→ Rerank candidates
→ Generate cited answer
→ Return answer status and sources