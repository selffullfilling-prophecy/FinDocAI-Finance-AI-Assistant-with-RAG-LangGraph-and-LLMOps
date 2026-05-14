# FinDocAI - 10-K RAG Assistant

FinDocAI is a learning-focused finance RAG project for SEC 10-K reports. It supports upload, section/table-aware chunking, Chroma indexing, hybrid retrieval, reranking, answer generation with citations, Streamlit demo UI, and golden evals.

> Educational and research use only. This project does not provide investment advice, recommendations, or buy/sell/hold decisions.

## Current Flow

```text
10-K PDF/TXT
-> load pages
-> detect 10-K sections
-> split section/table chunks
-> write chunk JSONL + chunk eval
-> index chunks into Chroma
-> retrieve candidates by vector / keyword / hybrid
-> optional heuristic reranking
-> build grounded RAG prompt
-> NVIDIA LLM answer
-> answer + source citations
```

## Setup

```powershell
cd C:\Users\Admin\Documents\findocAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create or update `.env`:

```env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=deepseek-ai/deepseek-v4-flash
NVIDIA_TEMPERATURE=0.2
NVIDIA_TOP_P=0.95
NVIDIA_MAX_TOKENS=4096
NVIDIA_REASONING_ENABLED=false
NVIDIA_REASONING_EFFORT=high

EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=data/chroma

RAG_MAX_CHARS_PER_CHUNK=1800
RAG_MAX_TOTAL_CONTEXT_CHARS=12000
RERANK_ENABLED=true
RERANK_TOP_K=5
```

The first Chroma indexing run may download the embedding model if it is not cached.

## Run

Start FastAPI:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Start Streamlit:

```powershell
streamlit run frontend/streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
```

Demo workflow:

1. Open `Upload & Index`.
2. Upload a 10-K PDF/TXT.
3. Keep `Index into Chroma` enabled.
4. Copy or use the returned `collection_name`.
5. Open `Retriever Debug` to compare `vector`, `keyword`, and `hybrid`.
6. Open `Chat with Document`.
7. Ask questions with streaming on and inspect sources.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload, chunk, eval, optionally index into Chroma |
| `POST` | `/retrieve` | Debug retrieval with vector/keyword/hybrid and optional rerank |
| `POST` | `/chat` | Non-streaming RAG answer generation |
| `POST` | `/chat/stream` | Streaming RAG answer generation as SSE |
| `GET` | `/chat/sessions/{session_id}` | Read recent in-memory chat history |
| `DELETE` | `/chat/sessions/{session_id}` | Clear in-memory chat history |

Upload and index:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" `
  -F "file=@C:\Users\Admin\Documents\findocAI\data\images\raw\reports\NASDAQ_AAPL_2023.pdf" `
  -F "index_to_chroma=true"
```

Retriever debug:

```powershell
$body = @{
  question = "What were the drivers of net sales?"
  collection_name = "findoc_nasdaq_aapl_2023"
  retrieval_mode = "hybrid"
  rerank = $true
  top_k = 5
  candidate_k = 20
  with_score = $true
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/retrieve" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Chat:

```powershell
$body = @{
  question = "What were the drivers of net sales?"
  collection_name = "findoc_nasdaq_aapl_2023"
  retrieval_mode = "hybrid"
  rerank = $true
  top_k = 5
  candidate_k = 20
  session_id = "demo-session"
  use_memory = $true
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Retrieval Modes

`vector` uses Chroma similarity search with embedding scores.

`keyword` reads documents from the Chroma collection and scores query terms deterministically.

`hybrid` merges vector and keyword candidates by `chunk_id`, normalizes scores, and computes:

```text
hybrid_score = 0.65 * vector_norm + 0.35 * keyword_norm
```

## Reranking

The default reranker is deterministic and does not load an external model. It boosts:

- exact query term matches
- metadata section filters
- table chunks for cash flow / statement / balance sheet questions
- Item 7 for revenue, net sales, growth, margin, MD&A questions
- Item 8 for cash flow, assets, liabilities, consolidated statement questions

## Conversation Memory

Chat memory is in-memory per `session_id`. It stores recent user/assistant turns and sources. It is useful for the demo, but it is not a production database-backed memory layer.

Clear memory:

```powershell
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8000/chat/sessions/demo-session"
```

## Source Attribution Policy

FinDocAI separates retrieval context from cited sources:

- `retrieved_context` is the top-k context sent to the LLM. These passages are useful for Developer Mode and debugging.
- `sources` contains only the chunks explicitly cited by the answer with valid `[Source N]` citations.
- If the answer is insufficient, `answer_status = "insufficient_context"` and `sources = []`.
- If the answer makes claims without valid citations, `answer_status = "unverified_sources"` and `sources = []`.
- User Mode displays only supporting `sources`.
- Developer Mode may show `retrieved_context` as related retrieved passages, but they are not called sources.

Example:

```json
{
  "question": "What was Apple's weighted average interest rate in 2024?",
  "answer_status": "insufficient_context",
  "sources": []
}
```

Developer Mode may still show related retrieved passages from another period, such as 2023, for debugging. Those passages are not treated as sources unless the answer cites them directly and they support the specific claim.

## Golden Evals

Chunking eval:

```powershell
python -m app.rag.eval.chunking_eval --cases tests/golden/chunking_cases.json --output data/eval/chunking_golden_report.json
```

Retriever eval:

```powershell
python -m app.rag.eval.retriever_eval --cases tests/golden/retriever_cases.json --output data/eval/retriever_golden_report.json
```

Answer eval is skipped by default because it may call the live NVIDIA LLM. Run it explicitly:

```powershell
$env:RUN_LLM_EVAL="1"
python -m app.rag.eval.answer_eval --cases tests/golden/answer_cases.json --output data/eval/answer_golden_report.json
```

Without `RUN_LLM_EVAL=1`, answer eval reports live cases as skipped.

## Tests

Run targeted RAG demo tests:

```powershell
pytest tests/test_llm_client.py tests/test_hybrid_retriever.py tests/test_reranker.py tests/test_conversation_memory.py tests/test_answer_service.py tests/test_routes_chat.py tests/test_answer_golden_eval.py
```

Run all tests:

```powershell
pytest
```

Unit tests do not call NVIDIA, Chroma, or embedding models unless explicitly mocked for that test.

## Main Files

| Area | Files |
| --- | --- |
| Chunking | `app/rag/loader.py`, `section_detector.py`, `table_detector.py`, `splitter.py`, `chunk_pipeline.py` |
| Artifacts | `app/rag/chunk_artifacts.py` |
| Vector store | `app/rag/vector_store.py` |
| Keyword retrieval | `app/rag/keyword_retriever.py` |
| Hybrid retrieval | `app/rag/hybrid_retriever.py` |
| Reranking | `app/rag/reranker.py` |
| Memory | `app/rag/conversation_memory.py` |
| Prompting | `app/rag/prompt_builder.py` |
| LLM client | `app/rag/llm_client.py` |
| Answer service | `app/rag/answer_service.py` |
| APIs | `app/api/routes_upload.py`, `routes_retrieval.py`, `routes_chat.py` |
| Streamlit | `frontend/streamlit_app.py` |
| Golden evals | `app/rag/eval/*.py`, `tests/golden/*.json` |

## Known Non-Goals For This Phase

- MLOps
- MLflow integration
- cloud deployment
- auth and user management
- production database memory
- React frontend
