from fastapi import APIRouter, HTTPException

from app.rag.hybrid_retriever import retrieve_candidates
from app.rag.reranker import candidates_to_ranked_documents, rerank_candidates
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.vector_store import similarity_search_with_score
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedChunk


router = APIRouter(tags=["Retrieval"])


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve_chunks(request: RetrievalRequest) -> RetrievalResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if not request.collection_name.strip():
        raise HTTPException(status_code=400, detail="Collection name is required.")

    try:
        if request.retrieval_mode != "vector" or request.rerank:
            candidates = retrieve_candidates(
                query=request.question,
                collection_name=request.collection_name,
                candidate_k=request.candidate_k,
                metadata_filter=request.metadata_filter,
                retrieval_mode=request.retrieval_mode,
            )
            scored_documents = (
                rerank_candidates(
                    request.question,
                    candidates,
                    top_k=request.top_k,
                    metadata_filter=request.metadata_filter,
                )
                if request.rerank
                else candidates_to_ranked_documents(candidates, top_k=request.top_k)
            )
            chunks = [
                RetrievedChunk(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    score=score,
                    vector_score=_as_float(document.metadata.get("vector_score")),
                    keyword_score=_as_float(document.metadata.get("keyword_score")),
                    hybrid_score=_as_float(document.metadata.get("hybrid_score")),
                    final_score=_as_float(document.metadata.get("final_score")),
                )
                for document, score in scored_documents
            ]
        elif request.with_score:
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
        candidate_k=request.candidate_k,
        retrieval_mode=request.retrieval_mode,
        rerank=request.rerank,
        chunks=chunks,
    )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
