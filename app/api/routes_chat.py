import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.rag.answer_service import answer_question, stream_answer_question
from app.rag.conversation_memory import memory_store
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if not request.collection_name.strip():
        raise HTTPException(status_code=400, detail="Collection name is required.")

    try:
        result = answer_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            metadata_filter=request.metadata_filter,
            retrieval_mode=request.retrieval_mode,
            rerank=request.rerank,
            session_id=request.session_id,
            use_memory=request.use_memory,
            use_memory_for_retrieval=request.use_memory_for_retrieval,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(**result)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if not request.collection_name.strip():
        raise HTTPException(status_code=400, detail="Collection name is required.")

    def event_stream():
        try:
            for event in stream_answer_question(
                question=request.question,
                collection_name=request.collection_name,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                metadata_filter=request.metadata_filter,
                retrieval_mode=request.retrieval_mode,
                rerank=request.rerank,
                session_id=request.session_id,
                use_memory=request.use_memory,
                use_memory_for_retrieval=request.use_memory_for_retrieval,
            ):
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/chat/sessions/{session_id}")
def clear_chat_session(session_id: str) -> dict[str, str]:
    return memory_store.clear_session(session_id)


@router.get("/chat/sessions/{session_id}")
def get_chat_session(session_id: str) -> dict:
    turns = memory_store.get_recent_history(session_id)
    return {
        "session_id": session_id,
        "turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "sources": turn.sources,
            }
            for turn in turns
        ],
    }


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
