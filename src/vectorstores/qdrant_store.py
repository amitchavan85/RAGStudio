"""Qdrant: production vector DB. Use a server URL, or ":memory:" for tests."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterator, List, Tuple

from src.chunking.base import Chunk
from src.vectorstores.base import SearchResult, VectorStore, chunk_uuid


class QdrantStore(VectorStore):
    def __init__(self, collection: str, dimension: int, url: str = "http://localhost:6333", api_key: str = ""):
        super().__init__(collection, dimension)
        from qdrant_client import QdrantClient

        self._in_memory = url == ":memory:"
        self._client = QdrantClient(location=url) if self._in_memory else QdrantClient(url=url, api_key=api_key or None)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection, vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )
            if not self._in_memory:  # payload indexes are not supported (and not needed) in :memory: mode
                from qdrant_client.models import PayloadSchemaType

                # Index "source" so delete/list by document stays fast on large collections.
                self._client.create_payload_index(self.collection, "source", PayloadSchemaType.KEYWORD)

    def _add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        self._check(chunks, vectors)
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=chunk_uuid(c.metadata["chunk_id"]), vector=v, payload={"text": c.text, **c.metadata})
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 256):
            self._client.upsert(self.collection, points=points[i : i + 256])

    @staticmethod
    def _filter(filters: Dict[str, list]):
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        conds = []
        for key, vals in filters.items():
            match = MatchValue(value=vals[0]) if len(vals) == 1 else MatchAny(any=list(vals))
            conds.append(FieldCondition(key=key, match=match))
        return Filter(must=conds)

    def _search(self, vector: List[float], k: int, filters: Dict[str, list]) -> List[SearchResult]:
        res = self._client.query_points(
            self.collection, query=vector, limit=k, with_payload=True, query_filter=self._filter(filters)
        )
        out = []
        for p in res.points:
            payload = dict(p.payload or {})
            out.append(SearchResult(text=payload.pop("text", ""), metadata=payload, score=float(p.score)))
        return out

    def _iter_records(self) -> Iterator[Tuple[str, dict]]:
        offset = None
        while True:
            points, offset = self._client.scroll(self.collection, limit=1000, offset=offset,
                                                 with_payload=True, with_vectors=False)
            for p in points:
                payload = dict(p.payload or {})
                yield payload.pop("text", ""), payload
            if offset is None:
                return

    def _source_filter(self, source: str):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])

    def _delete_source(self, source: str) -> int:
        from qdrant_client.models import FilterSelector

        flt = self._source_filter(source)
        n = self._client.count(self.collection, count_filter=flt, exact=True).count
        if n:
            self._client.delete(self.collection, points_selector=FilterSelector(filter=flt), wait=True)
        return n

    def _list_sources(self) -> Dict[str, int]:
        counts: Counter = Counter()
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=["source"], with_vectors=False
            )
            counts.update((p.payload or {}).get("source") for p in points)
            if offset is None:
                return dict(counts)

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count

    def _reset(self) -> None:
        self._client.delete_collection(self.collection)
        self._ensure_collection()
