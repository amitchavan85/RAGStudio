"""Chroma: embedded, persistent on local disk, zero external services."""
from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Iterator, List, Tuple

from src.chunking.base import Chunk
from src.vectorstores.base import SearchResult, VectorStore


class ChromaStore(VectorStore):
    def __init__(self, collection: str, dimension: int, data_dir: str = "./.rag_data"):
        super().__init__(collection, dimension)
        import chromadb

        self._client = chromadb.PersistentClient(path=os.path.join(data_dir, "chroma"))
        self._col = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def _add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        self._check(chunks, vectors)
        for i in range(0, len(chunks), 500):  # stay under Chroma's max batch size
            batch = slice(i, i + 500)
            self._col.upsert(
                ids=[c.metadata["chunk_id"] for c in chunks[batch]],
                embeddings=vectors[batch],
                documents=[c.text for c in chunks[batch]],
                metadatas=[c.metadata for c in chunks[batch]],
            )

    @staticmethod
    def _where(filters: Dict[str, list]):
        if not filters:
            return None
        clauses = [{k: {"$in": v}} for k, v in filters.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _search(self, vector: List[float], k: int, filters: Dict[str, list]) -> List[SearchResult]:
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[vector], n_results=min(k, n), where=self._where(filters))
        return [
            SearchResult(text=t, metadata=m, score=1.0 - d)
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def _iter_records(self) -> Iterator[Tuple[str, dict]]:
        offset = 0
        while True:
            page = self._col.get(include=["documents", "metadatas"], limit=5000, offset=offset)
            if not page["ids"]:
                return
            yield from zip(page["documents"], page["metadatas"])
            offset += len(page["ids"])

    def _delete_source(self, source: str) -> int:
        ids = self._col.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def _list_sources(self) -> Dict[str, int]:
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return dict(Counter(m.get("source") for m in metas))

    def count(self) -> int:
        return self._col.count()

    def _reset(self) -> None:
        self._client.delete_collection(self.collection)
        self._col = self._client.get_or_create_collection(self.collection, metadata={"hnsw:space": "cosine"})
