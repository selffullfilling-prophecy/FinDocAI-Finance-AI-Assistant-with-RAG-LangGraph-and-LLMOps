# 10-K Chunking Practice Skeleton

This folder follows the ingestion part of `data/images/General-Flow.png`:

```text
User Upload
-> FastAPI /upload
-> loader.py
-> chunk_pipeline.py
-> splitter.py
-> embeddings.py
-> Chroma
```

## Learning Goal

Build a 10-K chunking pipeline that keeps the fixed SEC report structure:

1. Document-based chunking: load PDF/TXT into page-level `Document` objects.
2. Section-aware chunking: detect `Item 1`, `Item 1A`, `Item 7`, `Item 8`, etc.
3. Table-aware chunking: keep financial tables as standalone chunks.
4. Recursive chunking: split long narrative sections into retrieval-friendly chunks.

## Files To Practice

| File | Responsibility |
| --- | --- |
| `loader.py` | Read PDF/TXT into page-level LangChain documents. |
| `chunk_models.py` | Shared config and metadata models for chunks. |
| `section_detector.py` | Detect 10-K Item sections from extracted text. |
| `table_detector.py` | Detect financial table-like blocks. |
| `splitter.py` | Combine table-aware and recursive chunking for each section. |
| `chunk_pipeline.py` | Orchestrate loading, section detection, splitting, and output documents. |

## Suggested TODO Order

1. Improve `section_detector.detect_10k_sections`.
2. Improve `table_detector.detect_tables`.
3. Tune `ChunkConfig` in `chunk_models.py`.
4. Add tests with a small fake 10-K text sample.
5. Connect `chunk_10k_file` into `/upload`.
