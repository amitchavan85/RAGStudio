"""OpenAI embeddings (paid API). Batches requests and retries transient errors."""
from __future__ import annotations

import logging
from typing import List

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.embeddings.base import EmbeddingProvider
from src.logging_config import log_step

logger = logging.getLogger(__name__)

_KNOWN_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None, batch_size: int = 100):
        from openai import OpenAI

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set (add it to .env or the environment)")
        self._client = OpenAI(api_key=api_key)
        self._model = model_name
        self.model_name = model_name
        self._batch_size = batch_size
        self._dim = _KNOWN_DIMS.get(model_name) or len(self._embed(["dimension probe"])[0])

    @property
    def dimension(self) -> int:
        return self._dim

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True,
           before_sleep=before_sleep_log(logger, logging.WARNING))
    def _embed(self, batch: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=batch)
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        with log_step(logger, "embed", provider="openai", model=self._model, texts=len(texts)):
            for i in range(0, len(texts), self._batch_size):
                out.extend(self._embed(texts[i : i + self._batch_size]))
        return out
