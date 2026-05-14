from fastapi import APIRouter, HTTPException

from app.rag.answer_service import answer_question
from app.schemas.chat import ChatRequest, ChatResponse


router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        result = answer_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(**result)
