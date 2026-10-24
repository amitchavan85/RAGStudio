"""
Vector store contract. Every backend (Chroma, FAISS, Qdrant, pgvector)
implements the same four operations, so ingestion/retrieval code never
knows which one is underneath. All backends use cosine similarity and
return `score` where HIGHER = more similar.
"""
from __future__ import annotations

import logging
import re
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from src.chunking.base import Chunk
from src.logging_config import log_step

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    scores: dict = field(default_factory=dict)  # per-stage scores: vector, keyword, rrf, rerank


Filters = Dict[str, object]


def normalize_filters(filters: Optional[Filters]) -> Dict[str, list]:
    """Validate metadata filters and turn every value into a list ("any of").

    {"source": "a.pdf"} or {"source": ["a.pdf", "b.pdf"], "page": 2}: every key must match.
    Keys are validated because some backends (pgvector) put them into the query text.
    """
    out: Dict[str, list] = {}
    for key, value in (filters or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ValueError(f"invalid filter key: {key!r}")
        values = list(value) if isinstance(value, (list, tuple, set)) else [value]
        if not values or not all(isinstance(v, (str, int, float, bool)) for v in values):
            raise ValueError(f"filter {key!r} needs a value or a non-empty list of plain values")
        out[key] = values
    return out


def matches(metadata: dict, filters: Dict[str, list]) -> bool:
    """Compare as strings so page=2 and page="2" behave the same."""
    return all(str(metadata.get(k)) in {str(v) for v in vals} for k, vals in filters.items())


def chunk_uuid(chunk_id: str) -> str:
    """Deterministic UUID from a chunk_id, so re-ingesting upserts instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore(ABC):
    def __init__(self, collection: str, dimension: int):
        self.collection = collection
        self.dimension = dimension

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """Upsert chunks with their embeddings (keyed on metadata['chunk_id'])."""
        self._check(chunks, vectors)
        with log_step(logger, "store.add", store=self._name, collection=self.collection, chunks=len(chunks)):
            self._add(chunks, vectors)

    def search(self, vector: List[float], k: int = 5, filters: Optional[Filters] = None) -> List[SearchResult]:
        """Return the k most similar chunks, best first, optionally restricted by metadata filters."""
        flt = normalize_filters(filters)
        with log_step(logger, "retrieve", store=self._name, collection=self.collection, k=k,
                      filters=sorted(flt) or None) as out:
            results = self._search(vector, k, flt)
            for r in results:
                r.scores.setdefault("vector", r.score)
            out["hits"] = len(results)
            if results:
                out["top_score"] = round(results[0].score, 3)
        return results

    def reset(self) -> None:
        """Delete all data in this collection."""
        with log_step(logger, "store.reset", store=self._name, collection=self.collection):
            self._reset()

    def delete_source(self, source: str) -> int:
        """Delete every chunk of one document (metadata['source']). Returns chunks removed."""
        with log_step(logger, "store.delete", store=self._name, collection=self.collection, source=source) as out:
            out["removed"] = removed = self._delete_source(source)
        return removed

    @contextmanager
    def bulk(self) -> Iterator[None]:
        """Wrap many add() calls (e.g. ingesting a folder). Stores whose indexes are cheaper to
        build once than to maintain per insert (pgvector) override this; others do nothing."""
        yield

    def list_sources(self) -> Dict[str, int]:
        """Documents in this collection: {source: chunk_count}."""
        return self._list_sources()

    def iter_records(self) -> Iterator[Tuple[str, dict]]:
        """Every stored chunk as (text, metadata); used to build the keyword (BM25) index."""
        return self._iter_records()

    @abstractmethod
    def _iter_records(self) -> Iterator[Tuple[str, dict]]: ...

    @abstractmethod
    def _delete_source(self, source: str) -> int: ...

    @abstractmethod
    def _list_sources(self) -> Dict[str, int]: ...

    @abstractmethod
    def _add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None: ...

    @abstractmethod
    def _search(self, vector: List[float], k: int, filters: Dict[str, list]) -> List[SearchResult]: ...

    @abstractmethod
    def _reset(self) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @property
    def _name(self) -> str:
        return type(self).__name__.removesuffix("Store").lower()

    @staticmethod
    def _check(chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
