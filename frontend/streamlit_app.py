from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("FINDOC_API_URL", "http://127.0.0.1:8000")
DEFAULT_CHAT_SETTINGS = {
    "retrieval_mode": "hybrid",
    "rerank": True,
    "stream": True,
    "use_memory": True,
    "top_k": 5,
    "candidate_k": 20,
    "section_filter": "",
}
SUGGESTED_QUESTIONS = [
    "What were the main drivers of revenue growth?",
    "What are the key risk factors?",
    "Summarize the cash flow performance.",
    "What changed in gross margin?",
]
EVAL_REPORTS = {
    "Chunking": Path("data/eval/chunking_golden_report.json"),
    "Retriever": Path("data/eval/retriever_golden_report.json"),
    "Answer": Path("data/eval/answer_golden_report.json"),
}


st.set_page_config(
    page_title="FinDocAI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("active_collection_name", "")
    st.session_state.setdefault("active_document_name", "")
    st.session_state.setdefault("document_ready", False)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("session_id", "default")
    st.session_state.setdefault("upload_widget_version", 0)
    st.session_state.setdefault("last_raw_response", None)


def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def check_health(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.ok:
            return True, "Backend is ready."
        return False, f"Backend returned HTTP {response.status_code}."
    except requests.RequestException:
        return (
            False,
            "Cannot connect to backend. Please run: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000",
        )


def upload_document(base_url: str, uploaded_file: Any, index_to_chroma: bool = True) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    response = requests.post(
        f"{base_url}/upload",
        files=files,
        data={"index_to_chroma": str(index_to_chroma).lower()},
        timeout=300,
    )
    _raise_for_api_error(response)
    return response.json()


def retrieve_chunks(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url}/retrieve", json=payload, timeout=120)
    _raise_for_api_error(response)
    return response.json()


def chat_once(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url}/chat", json=payload, timeout=180)
    _raise_for_api_error(response)
    return response.json()


def stream_chat(base_url: str, payload: dict[str, Any]):
    with requests.post(f"{base_url}/chat/stream", json=payload, stream=True, timeout=180) as response:
        _raise_for_api_error(response)
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            yield json.loads(line.removeprefix("data:").strip())


def clear_memory(base_url: str, session_id: str) -> None:
    response = requests.delete(f"{base_url}/chat/sessions/{session_id}", timeout=20)
    _raise_for_api_error(response)


def _raise_for_api_error(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise RuntimeError(str(detail))


def friendly_error(error: Exception, developer_mode: bool) -> str:
    message = str(error)
    if "NVIDIA_API_KEY" in message:
        return "LLM API key is not configured. Please set NVIDIA_API_KEY in your environment."
    if "Connection" in message or "connect" in message.lower():
        return "Cannot connect to backend. Please run: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
    if "Collection name is required" in message or "does not exist" in message.lower():
        return "Please upload and process a document first."
    if "no results" in message.lower() or "no chunks" in message.lower():
        return "I could not find relevant information in the uploaded document."
    if developer_mode:
        return message
    return "Something went wrong while processing your request."


def build_metadata_filter(section_item: str) -> dict[str, Any] | None:
    section_item = section_item.strip()
    if not section_item:
        return None
    return {"section_item": section_item}


def make_chat_payload(question: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": question,
        "collection_name": settings["collection_name"],
        "top_k": settings["top_k"],
        "candidate_k": settings["candidate_k"],
        "retrieval_mode": settings["retrieval_mode"],
        "rerank": settings["rerank"],
        "session_id": settings["session_id"],
        "use_memory": settings["use_memory"],
        "metadata_filter": build_metadata_filter(settings["section_filter"]),
    }


def render_sidebar() -> tuple[bool, bool, dict[str, Any]]:
    with st.sidebar:
        st.subheader("Status")
        developer_mode = st.checkbox("Developer Mode", value=False)

        if developer_mode:
            with st.expander("Advanced settings", expanded=True):
                st.session_state["api_url"] = st.text_input("API Base URL", value=api_url())
                collection_name = st.text_input(
                    "collection_name",
                    value=st.session_state.get("active_collection_name", ""),
                )
                session_id = st.text_input("session_id", value=st.session_state.get("session_id", "default"))
                retrieval_mode = st.selectbox("retrieval_mode", ["hybrid", "vector", "keyword"])
                rerank = st.checkbox("rerank", value=True)
                stream = st.checkbox("stream", value=True)
                use_memory = st.checkbox("memory", value=True)
                top_k = st.slider("top_k", 1, 20, DEFAULT_CHAT_SETTINGS["top_k"])
                candidate_k = st.slider("candidate_k", 1, 50, DEFAULT_CHAT_SETTINGS["candidate_k"])
                section_filter = st.text_input("section filter", value="")
        else:
            collection_name = st.session_state.get("active_collection_name", "")
            session_id = st.session_state.get("session_id", "default")
            retrieval_mode = DEFAULT_CHAT_SETTINGS["retrieval_mode"]
            rerank = DEFAULT_CHAT_SETTINGS["rerank"]
            stream = DEFAULT_CHAT_SETTINGS["stream"]
            use_memory = DEFAULT_CHAT_SETTINGS["use_memory"]
            top_k = DEFAULT_CHAT_SETTINGS["top_k"]
            candidate_k = DEFAULT_CHAT_SETTINGS["candidate_k"]
            section_filter = DEFAULT_CHAT_SETTINGS["section_filter"]

        healthy, health_message = check_health(api_url())
        if healthy:
            st.success("Ready")
        else:
            st.warning("Backend unavailable")
            st.caption(health_message)

        st.divider()
        st.subheader("Upload document")
        uploaded_file = st.file_uploader(
            "PDF or TXT",
            type=["pdf", "txt"],
            key=f"financial_doc_{st.session_state['upload_widget_version']}",
        )
        index_to_chroma = True
        if developer_mode:
            index_to_chroma = st.checkbox("Index into Chroma", value=True)

        process_clicked = st.button(
            "Process document",
            type="primary",
            disabled=uploaded_file is None or not healthy,
            use_container_width=True,
        )

        if process_clicked and uploaded_file is not None:
            process_uploaded_document(uploaded_file, index_to_chroma, developer_mode)

    settings = {
        "collection_name": collection_name,
        "session_id": session_id,
        "retrieval_mode": retrieval_mode,
        "rerank": rerank,
        "stream": stream,
        "use_memory": use_memory,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "section_filter": section_filter,
    }
    return healthy, developer_mode, settings


def process_uploaded_document(uploaded_file: Any, index_to_chroma: bool, developer_mode: bool) -> None:
    with st.spinner("Processing your document..."):
        try:
            response = upload_document(api_url(), uploaded_file, index_to_chroma=index_to_chroma)
        except Exception as exc:
            st.error(friendly_error(exc, developer_mode))
            if developer_mode:
                st.exception(exc)
            return

    st.session_state["upload_response"] = response
    st.session_state["active_document_name"] = response.get("file_name", uploaded_file.name)
    st.session_state["active_collection_name"] = response.get("collection_name", "")
    st.session_state["document_ready"] = bool(response.get("indexed") and response.get("collection_name"))
    st.session_state["chat_messages"] = []
    st.session_state["last_raw_response"] = response

    if st.session_state["document_ready"]:
        st.success("Your document is ready. You can now ask questions.")
    elif response.get("indexing_error"):
        st.warning(friendly_error(RuntimeError(response["indexing_error"]), developer_mode))
    else:
        st.warning("Document was processed, but it was not indexed for chat.")


def render_user_mode(healthy: bool, settings: dict[str, Any]) -> None:
    st.title("FinDocAI")
    st.caption("Ask questions about your financial documents.")

    render_document_card(developer_mode=False)
    render_suggested_questions(healthy, settings, developer_mode=False)
    render_chat_area(healthy, settings, developer_mode=False)


def render_developer_mode(healthy: bool, settings: dict[str, Any]) -> None:
    st.title("FinDocAI")
    st.caption("Ask questions about your financial documents.")

    tab_chat, tab_retriever, tab_eval = st.tabs(["Chat", "Retriever Debug", "Golden Evals"])
    with tab_chat:
        render_document_card(developer_mode=True)
        render_suggested_questions(healthy, settings, developer_mode=True)
        render_chat_area(healthy, settings, developer_mode=True)
        if st.session_state.get("last_raw_response") is not None:
            with st.expander("Raw JSON response", expanded=False):
                st.json(st.session_state["last_raw_response"])

    with tab_retriever:
        render_retriever_debug(healthy, settings)

    with tab_eval:
        render_golden_evals()


def render_document_card(developer_mode: bool) -> None:
    with st.container(border=True):
        document_name = st.session_state.get("active_document_name") or "No document processed"
        if st.session_state.get("document_ready"):
            st.success("Your document is ready. You can now ask questions.")
            st.write(f"Document: **{document_name}**")
        else:
            st.info("Upload a financial document to start asking questions.")

        if developer_mode and st.session_state.get("upload_response"):
            response = st.session_state["upload_response"]
            cols = st.columns(4)
            cols[0].metric("Chunks", response.get("total_chunks", 0))
            cols[1].metric("Vectors", response.get("vector_count") or 0)
            cols[2].metric("Chunk score", response.get("chunk_quality_score") or 0)
            cols[3].write(f"Collection: `{response.get('collection_name')}`")


def render_suggested_questions(healthy: bool, settings: dict[str, Any], developer_mode: bool) -> None:
    st.subheader("Suggested questions")
    cols = st.columns(2)
    selected_question = None
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        if cols[index % 2].button(question, use_container_width=True, disabled=not can_chat(healthy)):
            selected_question = question

    if selected_question:
        submit_question(selected_question, settings, developer_mode)


def render_chat_area(healthy: bool, settings: dict[str, Any], developer_mode: bool) -> None:
    st.subheader("Chat")
    clear_cols = st.columns([1, 4])
    if clear_cols[0].button("Clear chat", use_container_width=True):
        clear_chat(settings["session_id"], developer_mode)

    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            if message.get("warning"):
                st.warning(message["content"])
            else:
                st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"], developer_mode=developer_mode)

    if not st.session_state.get("document_ready"):
        st.info("Upload a financial document to start asking questions.")

    user_input = st.chat_input(
        "Ask about this financial document...",
        disabled=not can_chat(healthy),
    )
    if user_input:
        submit_question(user_input, settings, developer_mode)


def submit_question(question: str, settings: dict[str, Any], developer_mode: bool) -> None:
    if not st.session_state.get("document_ready"):
        st.warning("Please upload and process a document first.")
        return

    payload = make_chat_payload(question, settings)
    st.session_state["chat_messages"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if settings["stream"]:
            answer, sources, raw_response = run_streaming_chat(payload, developer_mode)
        else:
            answer, sources, raw_response = run_non_streaming_chat(payload, developer_mode)

        warning = is_insufficient_context(answer)
        if warning:
            st.warning("I could not find enough information in the uploaded document.")
        elif not settings["stream"]:
            st.markdown(answer)
        render_sources(sources, developer_mode=developer_mode)

    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": answer, "sources": sources, "warning": warning}
    )
    st.session_state["last_raw_response"] = raw_response


def run_streaming_chat(payload: dict[str, Any], developer_mode: bool) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    answer_placeholder = st.empty()
    answer = ""
    sources: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []

    try:
        for event in stream_chat(api_url(), payload):
            raw_events.append(event)
            event_type = event.get("type")
            if event_type == "token":
                answer += event.get("content", "")
                answer_placeholder.markdown(answer)
            elif event_type == "sources":
                sources = event.get("sources", [])
            elif event_type == "error":
                raise RuntimeError(event.get("message", "Streaming failed."))
    except Exception as exc:
        message = friendly_error(exc, developer_mode)
        st.error(message)
        if developer_mode:
            st.exception(exc)
        return message, [], {"error": str(exc), "events": raw_events}

    return answer, sources, {"events": raw_events}


def run_non_streaming_chat(payload: dict[str, Any], developer_mode: bool) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    with st.spinner("Thinking..."):
        try:
            result = chat_once(api_url(), payload)
        except Exception as exc:
            message = friendly_error(exc, developer_mode)
            st.error(message)
            if developer_mode:
                st.exception(exc)
            return message, [], {"error": str(exc)}

    return result.get("answer", ""), result.get("sources", []), result


def clear_chat(session_id: str, developer_mode: bool) -> None:
    st.session_state["chat_messages"] = []
    try:
        clear_memory(api_url(), session_id)
    except Exception as exc:
        if developer_mode:
            st.warning(f"Could not clear backend memory: {exc}")
    st.success("Chat cleared.")


def can_chat(healthy: bool) -> bool:
    return bool(healthy and st.session_state.get("document_ready"))


def is_insufficient_context(answer: str) -> bool:
    return "provided documents do not contain enough information" in answer.lower()


def render_sources(sources: list[dict[str, Any]], developer_mode: bool) -> None:
    if not sources:
        return

    st.markdown("**Sources**")
    for index, source in enumerate(sources, start=1):
        section_item = source.get("section_item") or "UNKNOWN"
        section_title = source.get("section_title") or "Untitled section"
        label = f"Source {index}: Item {section_item} - {section_title}"
        with st.expander(label, expanded=index == 1):
            cols = st.columns(3)
            cols[0].write(f"Section: `Item {section_item}`")
            cols[1].write(f"Page: `{_page_range(source)}`")
            cols[2].write(f"Relevance score: `{_format_score(source.get('score') or source.get('final_score'))}`")
            st.write(source.get("preview") or "")
            if developer_mode:
                st.write(f"chunk_id: `{source.get('chunk_id')}`")
                st.json(source, expanded=False)


def render_retriever_debug(healthy: bool, settings: dict[str, Any]) -> None:
    st.subheader("Retriever Debug")
    collection_name = st.text_input("Collection", value=settings["collection_name"], key="debug_collection")
    question = st.text_input("Question", key="debug_question")

    cols = st.columns(5)
    retrieval_mode = cols[0].selectbox("Mode", ["hybrid", "vector", "keyword"], key="debug_mode")
    rerank = cols[1].checkbox("Rerank", value=settings["rerank"], key="debug_rerank")
    top_k = cols[2].slider("Top k", 1, 20, settings["top_k"], key="debug_top_k")
    candidate_k = cols[3].slider("Candidate k", 1, 50, settings["candidate_k"], key="debug_candidate_k")
    section_filter = cols[4].text_input("Section", value=settings["section_filter"], key="debug_section")

    if st.button("Retrieve", type="primary", disabled=not healthy):
        payload = {
            "question": question,
            "collection_name": collection_name,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "retrieval_mode": retrieval_mode,
            "rerank": rerank,
            "with_score": True,
            "metadata_filter": build_metadata_filter(section_filter),
        }
        with st.spinner("Retrieving..."):
            try:
                result = retrieve_chunks(api_url(), payload)
            except Exception as exc:
                st.error(friendly_error(exc, developer_mode=True))
                st.exception(exc)
                return
        st.session_state["retriever_result"] = result

    result = st.session_state.get("retriever_result")
    if result:
        render_retrieved_chunks(result.get("chunks", []))
        with st.expander("Raw JSON", expanded=False):
            st.json(result)


def render_retrieved_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        st.warning("No chunks returned.")
        return

    rows = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        rows.append(
            {
                "rank": index,
                "score": chunk.get("score") or chunk.get("final_score"),
                "vector_score": chunk.get("vector_score"),
                "keyword_score": chunk.get("keyword_score"),
                "chunk_id": metadata.get("chunk_id"),
                "section_item": metadata.get("section_item"),
                "chunk_type": metadata.get("chunk_type"),
                "page": _page_range(metadata),
                "preview": " ".join(chunk.get("page_content", "").split())[:180],
            }
        )

    st.dataframe(rows, use_container_width=True)
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        title = (
            f"#{index} {metadata.get('chunk_id') or 'unknown'} "
            f"| Item {metadata.get('section_item') or 'UNKNOWN'} "
            f"| {metadata.get('chunk_type') or 'UNKNOWN'}"
        )
        with st.expander(title, expanded=index <= 2):
            st.text_area("Content", chunk.get("page_content", ""), height=180, key=f"retrieved_{index}")
            st.json(chunk, expanded=False)


def render_golden_evals() -> None:
    st.subheader("Golden Evals")
    st.code(
        "\n".join(
            [
                "python -m app.rag.eval.chunking_eval --cases tests/golden/chunking_cases.json --output data/eval/chunking_golden_report.json",
                "python -m app.rag.eval.retriever_eval --cases tests/golden/retriever_cases.json --output data/eval/retriever_golden_report.json",
                '$env:RUN_LLM_EVAL="1"',
                "python -m app.rag.eval.answer_eval --cases tests/golden/answer_cases.json --output data/eval/answer_golden_report.json",
            ]
        ),
        language="powershell",
    )

    for name, path in EVAL_REPORTS.items():
        report = load_report(path)
        with st.expander(f"{name}: {path}", expanded=report is not None):
            if not report:
                st.warning("Report not found.")
                continue
            cols = st.columns(4)
            cols[0].metric("Total", report.get("total_cases", 0))
            cols[1].metric("Passed", report.get("passed_cases", 0))
            cols[2].metric("Failed", report.get("failed_cases", 0))
            cols[3].metric("Skipped", report.get("skipped_cases", 0))
            st.json(report, expanded=False)


def _page_range(value: dict[str, Any]) -> str:
    page_start = value.get("page_start") or value.get("page_number")
    page_end = value.get("page_end") or page_start
    if page_start is None:
        return "-"
    if page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _format_score(score: Any) -> str:
    if score is None:
        return "-"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


init_state()
healthy, developer_mode, settings = render_sidebar()

if developer_mode:
    render_developer_mode(healthy, settings)
else:
    render_user_mode(healthy, settings)
