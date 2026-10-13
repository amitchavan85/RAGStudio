"""
Central configuration. Defaults come from environment / .env; API
requests can override chunker, vector store and LLM per call.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # .env is read from the project root, so settings apply no matter where you run from.
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Ingestion
    pdf_dir: str = "src/data/pdfs"  # relative paths are resolved from the project root
    ocr: str = "auto"  # auto: OCR scanned pages if rapidocr-onnxruntime is installed; off: never

    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"  # empty string disables the log file

    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embeddings
    embedding_provider: str = "sentence_transformers"  # | openai
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_api_key: SecretStr = SecretStr("")  # env: OPENAI_API_KEY; masked in repr/logs

    # Retrieval
    retrieval_mode: str = "hybrid"  # hybrid | vector | keyword
    reranker: str = "none"  # none | cross_encoder (local model, more precise, slower)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM
    llm_model: str = "gpt-4o-mini"  # any catalog model or "<provider>:<model>"; provider follows from it
    anthropic_api_key: SecretStr = SecretStr("")  # env: ANTHROPIC_API_KEY
    ollama_url: str = "http://localhost:11434"

    # Vector store
    vector_store: str = "chroma"  # chroma | faiss | qdrant | pgvector
    collection_name: str = "rag_docs"
    data_dir: str = "./.rag_data"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr = SecretStr("")
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/rag"

    # API (P4)
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Named indexes users can pick by name. JSON in .env, e.g.
    # INDEXES={"production": {"vector_store": "qdrant", "embedding_provider": "openai"}}
    # Empty -> one index called "default" built from VECTOR_STORE / EMBEDDING_PROVIDER / COLLECTION_NAME.
    indexes: Dict[str, Dict[str, str]] = {}
    allow_custom_index: bool = False  # True: requests may also pass {vector_store, embedding_provider, ...}
    allowed_chunkers: List[str] = []  # empty = all strategies; e.g. ["recursive","sentence","document"]
    max_upload_mb: int = 50
    max_k: int = 50
    # Auth + rate limits (P6). API_KEYS maps a client name to its key, e.g.
    # API_KEYS={"webapp": "long-random-key", "ci": "another-key"}. Empty = no auth (development only).
    api_keys: SecretStr = SecretStr("")  # JSON object; kept secret so keys never appear in logs/repr
    rate_limit_per_minute: int = 120  # all /v1 requests, per client; 0 = unlimited
    rate_limit_answers_per_minute: int = 30  # POST /v1/answer (calls a paid LLM), per client; 0 = unlimited

    @property
    def pdf_path(self) -> Path:
        p = Path(self.pdf_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
