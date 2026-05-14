from fastapi import FastAPI

from app.api.routes_health import router as health_router
from app.api.routes_retrieval import router as retrieval_router
from app.api.routes_upload import router as upload_router


app = FastAPI(
    title="FinDocGPT API",
    description="Finance document RAG API",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(retrieval_router)
