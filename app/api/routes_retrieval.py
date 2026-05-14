from fastapi import APIRouter, HTTPException

from app.rag.retriever import retrieve_relevant_chunks
from app.rag.vector_store import similarity_search_with_score
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunk


router = APIRouter(tags=["Retrieval"])


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve_chunks(request: RetrievalRequest) -> RetrievalResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        if request.with_score:
            scored_documents = similarity_search_with_score(
                query=request.question,
                collection_name=request.collection_name,
                k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
            chunks = [
                RetrievedChunk(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    score=score,
                )
                for document, score in scored_documents
            ]
        else:
            documents = retrieve_relevant_chunks(
                question=request.question,
                collection_name=request.collection_name,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
            chunks = [
                RetrievedChunk(page_content=document.page_content, metadata=document.metadata)
                for document in documents
            ]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RetrievalResponse(
        collection_name=request.collection_name,
        top_k=request.top_k,
        chunks=chunks,
    )
