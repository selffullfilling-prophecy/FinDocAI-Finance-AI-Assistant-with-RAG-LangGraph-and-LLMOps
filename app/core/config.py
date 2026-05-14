from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FinDocGPT"
    env: str = "development"

    llm_provider: str = "nvidia"

    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "deepseek-ai/deepseek-v4-flash"
    nvidia_temperature: float = 0.2
    nvidia_top_p: float = 0.95
    nvidia_max_tokens: int = 4096
    nvidia_reasoning_enabled: bool = False
    nvidia_reasoning_effort: str = "high"

    rag_max_chars_per_chunk: int = 1800
    rag_max_total_context_chars: int = 12000
    rerank_enabled: bool = True
    rerank_top_k: int = 5

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_persist_dir: str = "data/chroma"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    api_key: str = "dev-secret-key"
    rate_limit_per_minute: int = 30

    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "findocgpt-dev"

    mlflow_tracking_uri: str = "http://localhost:5000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
