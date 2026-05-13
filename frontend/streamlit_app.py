from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from app.rag.chunk_artifacts import (
    build_chunk_artifact_paths,
    evaluate_chunk_records,
    load_chunks_jsonl,
    write_chunks_jsonl,
    write_eval_report,
)
from app.rag.chunk_pipeline import chunk_10k_file
from app.rag.loader import SUPPORTED_EXTENSIONS


DEFAULT_API_URL = os.getenv("FINDOC_API_URL", "http://127.0.0.1:8000")
RAW_DIR = Path(os.getenv("FINDOC_RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("FINDOC_PROCESSED_DIR", "data/processed"))


st.set_page_config(
    page_title="FinDocGPT Chunking QA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def init_upload_state() -> None:
    st.session_state.setdefault("upload_widget_version", 0)
    st.session_state.setdefault("active_upload_signature", None)
    st.session_state.setdefault("clear_uploader_on_next_run", False)


def rotate_upload_widget() -> None:
    st.session_state["upload_widget_version"] = st.session_state.get("upload_widget_version", 0) + 1


def reset_upload_state(clear_file: bool = True) -> None:
    keys_to_clear = [
        "upload_response",
        "processed_path",
        "chunks",
        "eval_report",
        "manual_chunk_query",
        "special_case_example",
        "inspect_section_filter",
        "inspect_type_filter",
        "inspect_search_query",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.session_state["active_upload_signature"] = None
    if clear_file:
        rotate_upload_widget()


def uploaded_file_signature(uploaded_file: Any | None) -> str | None:
    if uploaded_file is None:
        return None
    return f"{uploaded_file.name}:{uploaded_file.size}:{uploaded_file.type}"


def check_health(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.ok:
            payload = response.json()
            return True, f"{payload.get('app', 'API')} / {payload.get('env', 'unknown')}"
        return False, f"HTTP {response.status_code}: {response.text[:200]}"
    except requests.RequestException:
        return (
            False,
            "FastAPI backend is not running at this URL. Start it with: "
            "uvicorn app.main:app --reload --host 127.0.0.1 --port 8000",
        )


def upload_to_api(base_url: str, uploaded_file: Any) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    response = requests.post(f"{base_url}/upload", files=files, timeout=180)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def chunk_locally(uploaded_file: Any) -> dict[str, Any]:
    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RuntimeError(f"Unsupported file type. Allowed: {allowed}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / original_name
    raw_path.write_bytes(uploaded_file.getvalue())

    chunks = chunk_10k_file(raw_path)
    if not chunks:
        raise RuntimeError("No extractable chunks were created.")

    artifacts = build_chunk_artifact_paths(PROCESSED_DIR, raw_path)
    write_chunks_jsonl(artifacts["versioned_chunks"], chunks)
    write_chunks_jsonl(artifacts["latest_chunks"], chunks)

    eval_report = evaluate_chunk_records(load_chunks_jsonl(artifacts["versioned_chunks"]))
    write_eval_report(artifacts["versioned_eval"], eval_report)
    write_eval_report(artifacts["latest_eval"], eval_report)

    return {
        "file_name": original_name,
        "status": 200,
        "total_chunks": len(chunks),
        "message": f"Locally chunked successfully. Debug chunks: {artifacts['versioned_chunks']}",
        "processed_path": str(artifacts["versioned_chunks"]),
        "latest_processed_path": str(artifacts["latest_chunks"]),
        "eval_report_path": str(artifacts["versioned_eval"]),
        "latest_eval_report_path": str(artifacts["latest_eval"]),
        "chunk_quality_score": eval_report["score"],
    }


def resolve_processed_path(upload_response: dict[str, Any]) -> Path:
    processed_path = upload_response.get("processed_path")
    if processed_path:
        path = Path(processed_path)
        return path if path.is_absolute() else Path.cwd() / path

    file_name = upload_response.get("file_name", "")
    return Path.cwd() / "data" / "processed" / f"{Path(file_name).stem}.chunks.jsonl"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return load_chunks_jsonl(path)


def load_eval_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_metadata_value(chunk: dict[str, Any], key: str, default: str = "UNKNOWN") -> str:
    value = chunk.get("metadata", {}).get(key)
    if value is None:
        return default
    return str(value)


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_types = Counter(get_metadata_value(chunk, "chunk_type") for chunk in chunks)
    sections = Counter(get_metadata_value(chunk, "section_item") for chunk in chunks)
    lengths = [len(chunk.get("page_content", "")) for chunk in chunks]

    return {
        "total": len(chunks),
        "text_chunks": chunk_types.get("section_text", 0),
        "table_chunks": chunk_types.get("table", 0),
        "sections": len(sections),
        "avg_chars": int(sum(lengths) / len(lengths)) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "chunk_types": chunk_types,
        "section_counts": sections,
    }


def filtered_chunks(
    chunks: list[dict[str, Any]],
    section_filter: list[str],
    type_filter: list[str],
    query: str,
) -> list[dict[str, Any]]:
    query_lower = query.lower().strip()
    results: list[dict[str, Any]] = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        section = str(metadata.get("section_item", "UNKNOWN"))
        chunk_type = str(metadata.get("chunk_type", "UNKNOWN"))
        content = chunk.get("page_content", "")

        if section_filter and section not in section_filter:
            continue
        if type_filter and chunk_type not in type_filter:
            continue
        if query_lower and query_lower not in content.lower() and query_lower not in json.dumps(metadata).lower():
            continue

        results.append(chunk)

    return results


def tokenize_query(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{1,}", text.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "where",
        "when",
        "which",
        "were",
        "was",
        "are",
        "how",
        "did",
        "does",
        "cua",
        "la",
        "gi",
        "trong",
        "nam",
    }
    return [token for token in tokens if token not in stopwords]


def score_chunk_for_query(chunk: dict[str, Any], query: str) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    content = chunk.get("page_content", "")
    content_lower = content.lower()
    metadata_text = json.dumps(metadata, ensure_ascii=False).lower()
    tokens = tokenize_query(query)

    matched_terms = sorted(
        {
            token
            for token in tokens
            if token in content_lower or token in metadata_text
        }
    )
    phrase_bonus = 8 if query.strip().lower() in content_lower else 0
    table_bonus = 2 if metadata.get("chunk_type") == "table" else 0
    section_bonus = 2 if str(metadata.get("section_item", "")).lower() in tokens else 0
    score = len(matched_terms) * 3 + phrase_bonus + table_bonus + section_bonus

    return {
        "score": score,
        "matched_terms": matched_terms,
        "chunk": chunk,
    }


def retrieve_from_chunks(chunks: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
    scored = [score_chunk_for_query(chunk, query) for chunk in chunks]
    scored = [item for item in scored if item["score"] > 0]
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def highlight_terms(text: str, terms: list[str]) -> str:
    escaped = text
    for term in sorted(terms, key=len, reverse=True):
        if not term:
            continue
        escaped = re.sub(
            re.escape(term),
            lambda match: f"**{match.group(0)}**",
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


def render_metrics(summary: dict[str, Any]) -> None:
    cols = st.columns(6)
    cols[0].metric("Chunks", summary["total"])
    cols[1].metric("Text", summary["text_chunks"])
    cols[2].metric("Tables", summary["table_chunks"])
    cols[3].metric("Sections", summary["sections"])
    cols[4].metric("Avg chars", summary["avg_chars"])
    cols[5].metric("Max chars", summary["max_chars"])


def render_eval_report(report: dict[str, Any] | None) -> None:
    if not report:
        st.warning("No chunk eval report found for this upload.")
        return

    score = report.get("score", 0)
    if score >= 85:
        st.success(f"Chunk quality score: {score}/100")
    elif score >= 65:
        st.warning(f"Chunk quality score: {score}/100")
    else:
        st.error(f"Chunk quality score: {score}/100")

    issues = report.get("issues", [])
    if not issues:
        st.caption("No deterministic chunking issues detected.")
    else:
        for issue in issues:
            st.markdown(f"- `{issue.get('severity')}` `{issue.get('code')}`: {issue.get('message')}")
            if issue.get("examples"):
                st.json(issue["examples"], expanded=False)

    with st.expander("Section Page Ranges", expanded=False):
        st.dataframe(report.get("section_page_ranges", []), use_container_width=True)


def render_chunk(chunk: dict[str, Any], index: int) -> None:
    metadata = chunk.get("metadata", {})
    chunk_id = metadata.get("chunk_id", f"chunk-{index}")
    chunk_type = metadata.get("chunk_type", "UNKNOWN")
    section_item = metadata.get("section_item", "UNKNOWN")
    section_title = metadata.get("section_title", "")
    page_start = metadata.get("page_start") or metadata.get("page_number")
    page_end = metadata.get("page_end") or page_start
    content = chunk.get("page_content", "")

    title = f"{chunk_id} | {chunk_type} | Item {section_item}"
    with st.expander(title, expanded=index < 3):
        top = st.columns([2, 2, 1, 1])
        top[0].caption(f"Section: {section_title}")
        top[1].caption(f"Pages: {page_start}-{page_end}")
        top[2].caption(f"Chars: {len(content)}")
        top[3].caption(f"Type: {chunk_type}")

        if chunk_type == "table":
            st.info(metadata.get("table_summary", "Table chunk"))

        st.text_area(
            "Content",
            value=content,
            height=220 if chunk_type == "table" else 180,
            key=f"content_{index}_{chunk_id}",
        )
        st.json(metadata, expanded=False)


def render_retrieval_result(result: dict[str, Any], index: int) -> None:
    chunk = result["chunk"]
    metadata = chunk.get("metadata", {})
    content = chunk.get("page_content", "")
    chunk_id = metadata.get("chunk_id", f"chunk-{index}")
    section_item = metadata.get("section_item", "UNKNOWN")
    section_title = metadata.get("section_title", "")
    chunk_type = metadata.get("chunk_type", "UNKNOWN")
    page_start = metadata.get("page_start") or metadata.get("page_number")
    page_end = metadata.get("page_end") or page_start
    matched_terms = result["matched_terms"]

    with st.expander(
        f"#{index + 1} score={result['score']} | {chunk_id} | Item {section_item} | {chunk_type}",
        expanded=index < 3,
    ):
        cols = st.columns([2, 2, 1])
        cols[0].caption(f"Section: {section_title}")
        cols[1].caption(f"Pages: {page_start}-{page_end}")
        cols[2].caption(f"Matched: {', '.join(matched_terms) if matched_terms else '-'}")

        st.markdown(highlight_terms(content, matched_terms))
        st.json(metadata, expanded=False)


init_upload_state()
if st.session_state.get("clear_uploader_on_next_run"):
    rotate_upload_widget()
    st.session_state["clear_uploader_on_next_run"] = False

st.title("10-K Chunking QA")

with st.sidebar:
    st.subheader("Mode")
    mode = st.radio(
        "Chunking mode",
        ["Local chunking", "API /upload"],
        index=0,
        key="chunking_mode_v2",
        horizontal=False,
        help="Use Local chunking for quality review. Use API /upload only when FastAPI is running.",
    )

    healthy = False
    if mode == "API /upload":
        st.subheader("Backend")
        st.session_state["api_url"] = st.text_input("API URL", value=api_url())

        healthy, health_message = check_health(api_url())
        if healthy:
            st.success(f"Connected: {health_message}")
        else:
            st.warning("API unavailable")
            st.code("uvicorn app.main:app --reload --host 127.0.0.1 --port 8000", language="powershell")
            st.caption(health_message)
    else:
        st.info("Local chunking does not require FastAPI.")

    st.divider()
    st.subheader("Upload")
    st.caption(f"Upload slot #{st.session_state['upload_widget_version']}")
    uploader_key = f"tenk_file_uploader_{st.session_state['upload_widget_version']}"
    uploaded_file = st.file_uploader("10-K PDF/TXT", type=["pdf", "txt"], key=uploader_key)
    current_signature = uploaded_file_signature(uploaded_file)

    active_signature = st.session_state.get("active_upload_signature")
    if current_signature is not None and active_signature is not None and current_signature != active_signature:
        reset_upload_state(clear_file=False)
        st.rerun()

    action_cols = st.columns(2)
    upload_clicked = action_cols[0].button(
        "Upload and chunk",
        type="primary",
        disabled=uploaded_file is None or (mode == "API /upload" and not healthy),
    )
    clear_clicked = action_cols[1].button("Clear", disabled=uploaded_file is None and not st.session_state.get("chunks"))

    if clear_clicked:
        reset_upload_state(clear_file=True)
        st.rerun()

if upload_clicked and uploaded_file is not None:
    with st.spinner("Uploading and chunking document..."):
        try:
            if mode == "API /upload":
                response_payload = upload_to_api(api_url(), uploaded_file)
            else:
                response_payload = chunk_locally(uploaded_file)
            processed_path = resolve_processed_path(response_payload)
            chunks = load_chunks(processed_path)
            eval_report_path = response_payload.get("eval_report_path")
            eval_report = load_eval_report(
                Path(eval_report_path) if eval_report_path else None
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["upload_response"] = response_payload
            st.session_state["processed_path"] = str(processed_path)
            st.session_state["chunks"] = chunks
            st.session_state["eval_report"] = eval_report
            st.session_state["active_upload_signature"] = uploaded_file_signature(uploaded_file)
            st.session_state["clear_uploader_on_next_run"] = True
            st.rerun()

chunks = st.session_state.get("chunks", [])
upload_response = st.session_state.get("upload_response")

if not chunks:
    st.info("Upload a real 10-K PDF or TXT file to inspect generated chunks.")
    st.stop()

summary = summarize_chunks(chunks)

st.subheader("Upload Result")
result_cols = st.columns([2, 1, 3])
result_cols[0].write(upload_response.get("file_name") if upload_response else "")
result_cols[1].write(f"{summary['total']} chunks")
result_cols[2].code(st.session_state.get("processed_path", ""), language="text")

render_metrics(summary)

st.subheader("Chunk Quality Eval")
render_eval_report(st.session_state.get("eval_report"))

left, right = st.columns([1, 1])
with left:
    st.subheader("Chunks by Section")
    st.bar_chart(dict(summary["section_counts"]))

with right:
    st.subheader("Chunks by Type")
    st.bar_chart(dict(summary["chunk_types"]))

st.subheader("Manual Chunk QA")
retrieval_tab, inspect_tab = st.tabs(["Query JSONL Chunks", "Inspect Chunks"])

sections = sorted(summary["section_counts"].keys())
types = sorted(summary["chunk_types"].keys())

with retrieval_tab:
    st.caption(
        "This searches the chunk JSONL produced by the latest upload. "
        "It is a deterministic keyword retriever for testing chunk quality before embeddings/Chroma."
    )
    query_cols = st.columns([5, 1])
    manual_query = query_cols[0].text_input(
        "Manual test query",
        placeholder="Example: Net cash provided by operating activities",
        key="manual_chunk_query",
    )
    top_k = query_cols[1].number_input("Top k", min_value=1, max_value=20, value=5, step=1)

    examples = [
        "Net cash provided by operating activities",
        "free cash flow less principal repayments",
        "risk factors competition regulation",
        "management discussion revenue growth",
        "financial statements supplementary data cash flows",
    ]
    selected_example = st.selectbox("Special-case examples", [""] + examples, key="special_case_example")
    effective_query = manual_query.strip() or selected_example.strip()

    if effective_query:
        results = retrieve_from_chunks(chunks, effective_query, int(top_k))
        st.caption(f"Retrieved {len(results)} chunks from {len(chunks)} JSONL chunks")
        if not results:
            st.warning("No matching chunks found. Try fewer words or a phrase from the source document.")
        for index, result in enumerate(results):
            render_retrieval_result(result, index)
    else:
        st.info("Enter a query or choose an example to test whether the chunk JSONL contains the expected context.")

with inspect_tab:
    filters = st.columns([2, 2, 3])
    section_filter = filters[0].multiselect("Section", sections, key="inspect_section_filter")
    type_filter = filters[1].multiselect("Chunk type", types, key="inspect_type_filter")
    query = filters[2].text_input("Search content or metadata", key="inspect_search_query")

    visible_chunks = filtered_chunks(chunks, section_filter, type_filter, query)
    st.caption(f"Showing {len(visible_chunks)} of {len(chunks)} chunks")

    for index, chunk in enumerate(visible_chunks):
        render_chunk(chunk, index)
