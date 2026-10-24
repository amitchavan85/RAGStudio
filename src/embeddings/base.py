"""Embedding provider contract used by chunkers, ingestion and retrieval."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingError(RuntimeError):
    """Turning text into vectors failed (e.g. the embedding API key is invalid). Raised by the
    pipeline so callers can tell an embedding problem apart from an LLM problem."""


class EmbeddingProvider(ABC):
    """Turns text into dense vectors. `dimension` is needed to create stores."""

    model_name: str = "unknown"  # identifies the vector space; part of the collection name

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    def fit(self, texts: List[str]) -> None:
        """Hook for corpus-relative providers (e.g. TF-IDF). No-op for dense models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
