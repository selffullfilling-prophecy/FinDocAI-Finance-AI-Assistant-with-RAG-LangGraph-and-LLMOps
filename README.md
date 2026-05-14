# FinDocAI — 10-K RAG Assistant

FinDocAI is a learning-focused finance RAG project for ingesting, chunking, indexing, retrieving, and evaluating SEC 10-K reports.

The current implementation focuses on the ingestion and retrieval foundation:

- PDF/TXT 10-K upload
- 10-K section-aware chunking
- table-aware chunking
- JSONL chunk artifacts
- deterministic chunk quality eval
- keyword retrieval over chunk JSONL
- Chroma vector store indexing
- vector retrieval with debug scores
- golden chunking eval
- golden retriever eval
- Streamlit QA interface

> This project is for research and education only. It does not provide investment advice, stock recommendations, or buy/sell/hold decisions.

---

## Current Status

The project currently supports this flow:

```text
10-K PDF/TXT
-> loader.py
-> section_detector.py
-> table_detector.py
-> splitter.py
-> chunk_pipeline.py
-> chunk JSONL + chunk eval report
-> optional embeddings
-> Chroma vector store
-> keyword/vector retrieval
-> golden eval runners
```

Answer generation with an LLM is not the active focus yet. The current goal is to make chunking and retrieval reliable before connecting `/chat` to full RAG answer generation.

---

## Main Components

| Area | Main Files | Status |
| --- | --- | --- |
| Document loading | `app/rag/loader.py` | Reads PDF/TXT page content. No OCR yet. |
| Section detection | `app/rag/section_detector.py` | Detects 10-K Items and filters TOC/index/cross-reference false positives. |
| Table detection | `app/rag/table_detector.py` | Heuristic table block detection. Good enough for current AAPL/AMZN eval, still improvable. |
| Splitting | `app/rag/splitter.py` | Section-aware, table-aware, recursive chunks. Splits oversized table/list blocks. |
| Chunk pipeline | `app/rag/chunk_pipeline.py` | Orchestrates load -> section detect -> split -> LangChain `Document`s. |
| Chunk artifacts | `app/rag/chunk_artifacts.py` | Writes versioned/latest JSONL and deterministic eval reports. |
| Keyword retrieval | `app/rag/keyword_retriever.py` | Reusable JSONL keyword retriever for manual/golden eval. |
| Embeddings | `app/rag/embeddings.py` | Sentence-transformers embedding model. |
| Vector store | `app/rag/vector_store.py` | Chroma indexing/search/reset/rebuild plus embedding manifest checks. |
| Retriever | `app/rag/retriever.py` | Thin wrapper around vector search. |
| Upload API | `app/api/routes_upload.py` | Upload, chunk, eval, optional Chroma indexing. |
| Retrieval API | `app/api/routes_retrieval.py` | Vector retrieval with optional scores/filtering. |
| Streamlit QA | `frontend/streamlit_app.py` | Upload/chunk QA, eval display, keyword query, vector query. |
| Golden eval | `app/rag/eval/*.py` | Chunking and retriever golden evaluation runners. |

---

## Project Structure

```text
findocAI/
├── app/
│   ├── api/
│   │   ├── routes_health.py
│   │   ├── routes_retrieval.py
│   │   └── routes_upload.py
│   ├── core/
│   │   └── config.py
│   ├── rag/
│   │   ├── eval/
│   │   │   ├── chunking_eval.py
│   │   │   └── retriever_eval.py
│   │   ├── chunk_artifacts.py
│   │   ├── chunk_models.py
│   │   ├── chunk_pipeline.py
│   │   ├── embeddings.py
│   │   ├── keyword_retriever.py
│   │   ├── loader.py
│   │   ├── retriever.py
│   │   ├── section_detector.py
│   │   ├── splitter.py
│   │   ├── table_detector.py
│   │   └── vector_store.py
│   ├── schemas/
│   └── main.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chroma/
│   └── eval/
├── frontend/
│   └── streamlit_app.py
├── tests/
│   ├── golden/
│   │   ├── chunking_cases.json
│   │   └── retriever_cases.json
│   ├── test_chunking_golden_eval.py
│   ├── test_chunking_pipeline.py
│   ├── test_keyword_retriever.py
│   ├── test_retriever_golden_eval.py
│   ├── test_upload_api.py
│   └── test_vector_store.py
└── requirements.txt
```

---

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` if needed. For chunking and local eval, LLM keys are not required.

Important environment variables:

```env
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=data/chroma
```

The first vector indexing run may download the embedding model if it is not cached.

---

## Run The App

FastAPI:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Streamlit QA UI:

```powershell
streamlit run frontend\streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
```

Recommended local QA workflow:

1. Select `Local chunking`.
2. Upload a 10-K PDF/TXT.
3. Inspect chunk score and section page ranges.
4. Query `Query JSONL Chunks` for deterministic keyword retrieval.
5. Optionally enable `Index chunks in Chroma`.
6. Query `Vector Retrieval`.

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | API health check |
| `POST` | `/upload` | Upload, chunk, eval, and optionally index a document |
| `POST` | `/retrieve` | Retrieve chunks from a Chroma collection |

Upload supports an optional form field:

```text
index_to_chroma=true
```

Example upload response fields:

```json
{
  "file_name": "NASDAQ_AMZN_2022.pdf",
  "total_chunks": 401,
  "processed_path": "data/processed/NASDAQ_AMZN_2022.20260514-101010.chunks.jsonl",
  "latest_processed_path": "data/processed/NASDAQ_AMZN_2022.chunks.latest.jsonl",
  "eval_report_path": "data/processed/NASDAQ_AMZN_2022.20260514-101010.chunk_eval.json",
  "latest_eval_report_path": "data/processed/NASDAQ_AMZN_2022.chunk_eval.latest.json",
  "chunk_quality_score": 98,
  "indexed": false,
  "collection_name": "findoc_nasdaq_amzn_2022"
}
```

Retrieve request:

```json
{
  "question": "Where is net cash provided by operating activities reported?",
  "collection_name": "findoc_nasdaq_amzn_2022",
  "top_k": 5,
  "metadata_filter": {
    "section_item": "7"
  },
  "with_score": true
}
```

---

## Chunk Artifacts

Each upload writes versioned and latest artifacts:

```text
data/processed/<document>.<timestamp>.chunks.jsonl
data/processed/<document>.chunks.latest.jsonl
data/processed/<document>.<timestamp>.chunk_eval.json
data/processed/<document>.chunk_eval.latest.json
```

Chunk metadata includes:

- `chunk_id`
- `chunk_type`
- `section_item`
- `section_title`
- `page_number`
- `page_start`
- `page_end`
- table metadata when applicable

---

## Chunk Quality Eval

`chunk_artifacts.evaluate_chunk_records()` performs deterministic checks:

- missing core sections
- unknown section ratio
- duplicate `chunk_id`
- table of contents leakage
- non-monotonic section page ranges
- very short chunk ratio/count
- very long chunk ratio/count

Scores are penalty-based. For example, 7 short chunks and 200 short chunks are no longer penalized equally.

---

## Golden Chunking Eval

Golden chunking cases live in:

```text
tests/golden/chunking_cases.json
```

Run:

```powershell
python -m app.rag.eval.chunking_eval --cases tests\golden\chunking_cases.json --output data\eval\chunking_golden_report.json
```

Current sample output:

```text
Chunking Golden Eval
- Total: 3
- Passed: 2
- Failed: 0
- Skipped: 1
- Pass rate: 100.0%
```

The skipped sample case is intentional and verifies that missing documents are skipped clearly instead of crashing.

---

## Golden Retriever Eval

Golden retriever cases live in:

```text
tests/golden/retriever_cases.json
```

Supported retriever types:

- `keyword_jsonl`: deterministic keyword retrieval over chunk JSONL
- `vector`: Chroma similarity search with scores

Run:

```powershell
python -m app.rag.eval.retriever_eval --cases tests\golden\retriever_cases.json --output data\eval\retriever_golden_report.json
```

Current sample output:

```text
Retriever Golden Eval
- Total: 3
- Hit@k: 100.0%
- MRR@k: 0.750
- Passed: 2
- Failed: 0
- Skipped: 1
```

The skipped vector case means the target Chroma collection has not been indexed yet. This is expected until you upload with `Index chunks in Chroma` enabled or rebuild the collection manually.

---

## Vector Store

`app/rag/vector_store.py` provides:

- `index_documents()`
- `similarity_search()`
- `similarity_search_with_score()`
- `reset_collection()`
- `rebuild_collection()`
- embedding config manifest read/write

Example:

```python
from app.rag.vector_store import similarity_search_with_score

results = similarity_search_with_score(
    query="net cash provided by operating activities",
    collection_name="findoc_nasdaq_amzn_2022",
    k=5,
    metadata_filter={"section_item": "7"},
)
```

The vector store writes:

```text
data/chroma/vector_store_manifest.json
```

If `EMBEDDING_MODEL` changes after indexing, the vector store raises a clear error asking you to rebuild the collection.

---

## Tests

Run all tests:

```powershell
python -m pytest tests -q
```

Current status:

```text
28 passed
```

Run only golden eval tests:

```powershell
python -m pytest tests\test_chunking_golden_eval.py tests\test_retriever_golden_eval.py tests\test_keyword_retriever.py -q
```

---

## Current Quality Assessment

Strong parts:

- 10-K section detection for current AAPL/AMZN samples
- chunk JSONL artifacts and versioning
- deterministic chunk quality scoring
- golden chunking eval foundation
- golden retriever eval foundation
- Chroma safety checks and embedding manifest

Known limitations:

- No OCR for scanned PDFs
- Table detection is still heuristic
- Vector retrieval quality is not yet benchmarked deeply
- No hybrid retrieval yet
- No reranking yet
- `/chat` is not yet connected to full RAG answer generation

---

## Suggested Next Steps

1. Add more golden chunking cases for AAPL/AMZN across years.
2. Add more golden retriever cases for finance-specific questions.
3. Index real Chroma collections and benchmark `keyword_jsonl` vs `vector`.
4. Implement hybrid retrieval.
5. Add reranking for retrieved chunks.
6. Connect `/chat` to retrieval + LLM answer generation with citations.

---

## Disclaimer

FinDocAI is for educational and research purposes only. It does not provide investment advice, investment recommendations, or predictions of future stock prices.
