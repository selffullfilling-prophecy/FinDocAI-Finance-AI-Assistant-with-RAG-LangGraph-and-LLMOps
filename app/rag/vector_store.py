from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.embeddings import get_embedding_model


MANIFEST_FILE_NAME = "vector_store_manifest.json"


def collection_name_from_file(file_name: str) -> str:
    """Create a Chroma-safe collection name from an uploaded file name."""

    stem = Path(file_name).stem.lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "_", stem).strip("_")
    normalized = normalized or "document"
    return f"findoc_{normalized}"[:63]


def get_vector_store(collection_name: str | None = None) -> Chroma:
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name or "findoc_documents",
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_dir),
    )


def get_manifest_path() -> Path:
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir / MANIFEST_FILE_NAME


def read_vector_store_manifest(collection_name: str | None = None) -> dict[str, Any] | None:
    """Read vector-store manifest metadata.

    If collection_name is provided, returns only that collection's manifest
    entry, or None when it has not been written yet.
    """

    manifest_path = get_manifest_path()
    if not manifest_path.exists():
        return None if collection_name else {"collections": {}}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if collection_name is None:
        return manifest

    return manifest.get("collections", {}).get(collection_name)


def write_vector_store_manifest(
    collection_name: str,
    vector_count: int,
) -> dict[str, Any]:
    """Write current embedding config for a collection."""

    settings = get_settings()
    manifest = read_vector_store_manifest() or {"collections": {}}
    entry = {
        "collection_name": collection_name,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "vector_count": vector_count,
    }
    manifest.setdefault("collections", {})[collection_name] = entry

    manifest_path = get_manifest_path()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def _remove_collection_from_manifest(collection_name: str) -> None:
    manifest = read_vector_store_manifest() or {"collections": {}}
    manifest.setdefault("collections", {}).pop(collection_name, None)
    get_manifest_path().write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_embedding_config_matches(collection_name: str) -> None:
    """Raise when collection was indexed with a different embedding config."""

    entry = read_vector_store_manifest(collection_name)
    if not entry:
        return

    settings = get_settings()
    current_provider = settings.embedding_provider
    current_model = settings.embedding_model
    indexed_provider = entry.get("embedding_provider")
    indexed_model = entry.get("embedding_model")

    if indexed_provider != current_provider or indexed_model != current_model:
        raise ValueError(
            "Embedding model changed. Please rebuild the vector store. "
            f"Collection '{collection_name}' was indexed with "
            f"{indexed_provider}/{indexed_model}, current config is "
            f"{current_provider}/{current_model}."
        )


def index_documents(
    documents: list[Document],
    collection_name: str,
) -> dict[str, Any]:
    """Add chunk documents to a Chroma collection."""

    if not documents:
        return {"collection_name": collection_name, "vector_count": 0}

    cleaned_documents = [_clean_document_metadata(document) for document in documents]
    _validate_chunk_ids(cleaned_documents)
    assert_embedding_config_matches(collection_name)
    vector_store = get_vector_store(collection_name)
    ids = [str(document.metadata["chunk_id"]) for document in cleaned_documents]
    vector_store.add_documents(cleaned_documents, ids=ids)
    write_vector_store_manifest(collection_name, len(cleaned_documents))

    return {
        "collection_name": collection_name,
        "vector_count": len(cleaned_documents),
    }


def similarity_search(
    query: str,
    collection_name: str,
    k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Document]:
    query = query.strip()
    if not query:
        return []

    assert_embedding_config_matches(collection_name)
    vector_store = get_vector_store(collection_name)
    return vector_store.similarity_search(query, k=k, filter=metadata_filter)


def similarity_search_with_score(
    query: str,
    collection_name: str,
    k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Search a collection and return Chroma scores for retrieval debugging."""

    query = query.strip()
    if not query:
        return []

    assert_embedding_config_matches(collection_name)
    vector_store = get_vector_store(collection_name)
    return vector_store.similarity_search_with_score(query, k=k, filter=metadata_filter)


def get_collection_documents(
    collection_name: str,
    metadata_filter: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Read stored documents from a Chroma collection without running embedding search."""

    collection_name = collection_name.strip()
    if not collection_name:
        raise ValueError("Collection name is required.")

    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if metadata_filter:
        get_kwargs["where"] = metadata_filter
    if limit is not None:
        get_kwargs["limit"] = limit
    result = collection.get(**get_kwargs)

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or [{} for _ in documents]
    return [
        Document(page_content=content or "", metadata=metadata or {})
        for content, metadata in zip(documents, metadatas, strict=False)
    ]


def reset_collection(collection_name: str) -> dict[str, str]:
    """Delete a Chroma collection and remove its manifest entry."""

    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        client.delete_collection(collection_name)
        status = "deleted"
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" in message or "not found" in message:
            status = "not_found"
        else:
            raise

    _remove_collection_from_manifest(collection_name)
    return {"collection_name": collection_name, "status": status}


def rebuild_collection(
    documents: list[Document],
    collection_name: str,
) -> dict[str, Any]:
    """Reset a collection and index documents with the current embedding model."""

    reset_collection(collection_name)
    return index_documents(documents, collection_name)


def _clean_document_metadata(document: Document) -> Document:
    metadata = {
        key: _clean_metadata_value(value)
        for key, value in document.metadata.items()
        if value is not None
    }
    return Document(page_content=document.page_content, metadata=metadata)


def _validate_chunk_ids(documents: list[Document]) -> None:
    for index, document in enumerate(documents):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is None or str(chunk_id).strip() == "":
            raise ValueError(f"Document at index {index} is missing required metadata['chunk_id'].")


def _clean_metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)