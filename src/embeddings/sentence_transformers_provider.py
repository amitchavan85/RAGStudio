"""Local embeddings via sentence-transformers (free, no API key)."""
from __future__ import annotations

import logging
from typing import List

from src.embeddings.base import EmbeddingProvider
from src.logging_config import log_step

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._batch_size = batch_size
        self._dim = int(self._model.get_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with log_step(logger, "embed", provider="sentence_transformers", texts=len(texts)):
            vecs = self._model.encode(
                texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return vecs.tolist()
