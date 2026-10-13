"""
FAISS: local exact (flat) index. FAISS holds only vectors, so chunk text/metadata
live in a sidecar. Vectors are L2-normalized so inner product == cosine similarity.

Storage is append-only so ingest cost does not grow with collection size:
  {collection}.f32    raw float32 vectors, appended per add()
  {collection}.jsonl  one record (text + metadata) per line, appended per add()
Only an upsert of existing chunk_ids rewrites the files (rare). The in-memory
index is rebuilt from the .f32 file on load, which is fast for a flat index.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from typing import Dict, Iterator, List, Tuple

import numpy as np

from src.chunking.base import Chunk
from src.vectorstores.base import SearchResult, VectorStore, matches


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class FaissStore(VectorStore):
    def __init__(self, collection: str, dimension: int, data_dir: str = "./.rag_data"):
        super().__init__(collection, dimension)
        import faiss

        self._faiss = faiss
        self._dir = os.path.join(data_dir, "faiss")
        os.makedirs(self._dir, exist_ok=True)
        self._vec_path = os.path.join(self._dir, f"{collection}.f32")
        self._meta_path = os.path.join(self._dir, f"{collection}.jsonl")
        self._records: List[dict] = []
        self._pos: Dict[str, int] = {}  # chunk_id -> row, for O(1) upsert checks
        self._index = faiss.IndexFlatIP(dimension)
        self._lock = threading.RLock()  # shared instance: requests may add/search concurrently
        self._load()

    def _load(self) -> None:
        if not (os.path.exists(self._vec_path) and os.path.exists(self._meta_path)):
            return
        with open(self._meta_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        vecs = np.fromfile(self._vec_path, dtype="float32").reshape(-1, self.dimension)
        n = min(len(records), len(vecs))  # tolerate a crash between the two appends
        self._records = records[:n]
        if n:
            self._index.add(np.ascontiguousarray(vecs[:n]))
        self._pos = {r["metadata"]["chunk_id"]: i for i, r in enumerate(self._records)}
        if n != len(records) or n != len(vecs):
            self._rewrite(vecs[:n])

    def _append(self, mat: np.ndarray, records: List[dict]) -> None:
        with open(self._vec_path, "ab") as f:
            mat.astype("float32").tofile(f)
        with open(self._meta_path, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)

    def _rewrite(self, all_vecs: np.ndarray) -> None:
        for p in (self._vec_path, self._meta_path):
            if os.path.exists(p):
                os.remove(p)
        self._append(all_vecs, self._records)

    def _add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        with self._lock:
            self._check(chunks, vectors)
            if not chunks:
                return
            mat = _normalize(np.asarray(vectors, dtype="float32"))
            records = [{"text": c.text, "metadata": c.metadata} for c in chunks]
            replaced = {self._pos[c.metadata["chunk_id"]] for c in chunks if c.metadata["chunk_id"] in self._pos}

            if not replaced:  # common path: pure append, cost independent of collection size
                start = len(self._records)
                self._index.add(mat)
                self._records.extend(records)
                self._pos.update({r["metadata"]["chunk_id"]: start + i for i, r in enumerate(records)})
                self._append(mat, records)
                return

            # Upsert: drop replaced rows, re-add everything in one vectorized rebuild.
            self._rebuild(drop=replaced, extra_vecs=mat, extra_records=records)

    def _rebuild(self, drop: set, extra_vecs: np.ndarray | None = None, extra_records: List[dict] | None = None) -> None:
        """Rebuild index + files without the `drop` rows, optionally appending new rows."""
        keep = np.array([i for i in range(len(self._records)) if i not in drop], dtype="int64")
        old = self._index.reconstruct_n(0, self._index.ntotal)[keep] if len(keep) else np.empty((0, self.dimension), "float32")
        all_vecs = old if extra_vecs is None else np.vstack([old, extra_vecs])
        self._records = [self._records[i] for i in keep] + (extra_records or [])
        self._pos = {r["metadata"]["chunk_id"]: i for i, r in enumerate(self._records)}
        self._index = self._faiss.IndexFlatIP(self.dimension)
        if len(all_vecs):
            self._index.add(np.ascontiguousarray(all_vecs, dtype="float32"))
        self._rewrite(all_vecs)

    def _iter_records(self) -> Iterator[Tuple[str, dict]]:
        with self._lock:  # snapshot, so a concurrent write can't change the list mid-iteration
            return iter([(r["text"], r["metadata"]) for r in self._records])

    def _delete_source(self, source: str) -> int:
        with self._lock:
            drop = {i for i, r in enumerate(self._records) if r["metadata"].get("source") == source}
            if drop:
                self._rebuild(drop)
            return len(drop)

    def _list_sources(self) -> Dict[str, int]:
        return dict(Counter(r["metadata"].get("source") for r in self._records))

    def _search(self, vector: List[float], k: int, filters: Dict[str, list]) -> List[SearchResult]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            q = _normalize(np.asarray([vector], dtype="float32"))
            if filters:  # restrict the exact search to matching rows (FAISS ID selector)
                allowed = np.array([i for i, r in enumerate(self._records) if matches(r["metadata"], filters)], dtype="int64")
                if not len(allowed):
                    return []
                params = self._faiss.SearchParameters()
                params.sel = self._faiss.IDSelectorBatch(allowed)
                scores, ids = self._index.search(q, min(k, len(allowed)), params=params)
            else:
                scores, ids = self._index.search(q, min(k, self._index.ntotal))
            return [
                SearchResult(text=self._records[i]["text"], metadata=self._records[i]["metadata"], score=float(s))
                for s, i in zip(scores[0], ids[0])
                if i != -1
            ]

    def count(self) -> int:
        return int(self._index.ntotal)

    def _reset(self) -> None:
        with self._lock:
            self._index = self._faiss.IndexFlatIP(self.dimension)
            self._records = []
            self._pos = {}
            for p in (self._vec_path, self._meta_path):
                if os.path.exists(p):
                    os.remove(p)
