from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return the embedding model used by Chroma.

    The project currently defaults to sentence-transformers because it can run
    locally and does not require an external API key. The first run may download
    the model configured in `.env`.
    """

    settings = get_settings()

    if settings.embedding_provider != "sentence_transformers":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
