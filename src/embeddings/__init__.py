"""Embedding providers. `get_embedder()` builds one by name."""
from src.embeddings.base import EmbeddingError, EmbeddingProvider


def get_embedder(provider: str | None = None, model: str | None = None) -> EmbeddingProvider:
    from src.config import get_settings

    s = get_settings()
    provider = provider or s.embedding_provider
    if provider == "sentence_transformers":
        from src.embeddings.sentence_transformers_provider import SentenceTransformerEmbeddings

        return SentenceTransformerEmbeddings(model or s.embedding_model)
    if provider == "openai":
        from src.embeddings.openai_provider import OpenAIEmbeddings

        return OpenAIEmbeddings(model or "text-embedding-3-small", api_key=s.openai_api_key.get_secret_value())
    raise ValueError(f"Unknown embedding provider: {provider!r}")


__all__ = ["EmbeddingError", "EmbeddingProvider", "get_embedder"]
