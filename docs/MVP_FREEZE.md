# FinDocAI MVP Freeze

## Version

- MVP version: v0.1
- Freeze date: 2026-05-15

## Included features

- Upload PDF/TXT financial documents.
- 10-K-oriented chunking with section, recursive, and table-aware handling.
- Chunk artifact writing and chunk quality evaluation.
- Chroma indexing and rebuild/reset utilities.
- Vector, keyword, and hybrid retrieval.
- Heuristic reranking.
- NVIDIA LLM answer generation.
- `/chat` and `/chat/stream` APIs.
- Streamlit demo UI with streaming/typewriter response rendering.
- Source attribution policy with `sources` separated from `retrieved_context`.
- `answer_status` values for answered, insufficient context, and unverified sources.
- In-memory conversation memory.
- Query rewriting for follow-up retrieval.
- Golden chunking, retriever, and answer evaluation runners.

## Known limitations

- Table extraction is useful for MVP demos but not production-grade.
- Table header/context recovery is heuristic and should be benchmarked on more filings.
- Query rewriting is rule-based first with optional LLM fallback; it is not a complete conversational search planner.
- Conversation memory is in-memory only and is cleared when the process restarts.
- No production database-backed memory.
- No MLOps, MLflow tracking, deployment, auth, or user management.
- No formal answer-quality benchmark beyond the golden and optional RAGAS evals.

## Benchmark plan

FinDocAI v0.1 uses two benchmark layers:

1. FinDocAI Golden Benchmark
   - Domain-specific cases for 10-K QA.
   - Checks answer status, citations, cited source sections, table QA, negative cases, and follow-up query rewriting.

2. Optional RAGAS Benchmark
   - Computes RAGAS metrics when `ragas`, `datasets`, and required evaluator credentials are available.
   - Skips clearly when dependencies or environment variables are missing.

The benchmark reports are intended to be written under `data/eval/` and summarized by:

```powershell
python -m app.rag.eval.benchmark_summary --output data/eval/benchmark_summary.md
```
