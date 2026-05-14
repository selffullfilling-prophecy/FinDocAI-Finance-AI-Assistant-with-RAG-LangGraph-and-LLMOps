# FinDocAI MVP Benchmark Summary

## Chunking Eval
- Total: 3
- Passed: 1
- Failed: 1
- Skipped: 1

## Retriever Eval
- Total: 3
- Passed: 2
- Failed: 0
- Skipped: 1
- Hit At K: 100.0%
- Mrr At K: 75.0%

## Answer Golden Eval
- Total: 15
- Passed: 0
- Failed: 0
- Skipped: 15
- Citation Rate: 0.0%
- Source Section Hit Rate: 0.0%
- Term Match Rate: 0.0%
- Answer Status Accuracy: 0.0%
- Insufficient Handling Rate: 0.0%

## RAGAS Eval
- Total: 3
- Completed: 0
- Failed: 0
- Skipped: 3
- Issue: RUN_RAGAS_EVAL is not set to 1; skipping optional RAGAS eval.

## Known limitations
- Table extraction is not production-grade.
- Follow-up query rewriting is heuristic-first and needs a larger benchmark set.
- Benchmark coverage is intentionally small for MVP v0.1.
- Memory is in-memory only and not suitable for production persistence.
