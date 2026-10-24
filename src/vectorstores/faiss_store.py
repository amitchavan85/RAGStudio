"""
FAISS: local exact (flat) index. FAISS holds only vectors, so chunk text/metadata
live in a sidecar. Vectors are L2-normalized so inner product == cosine similarity.

Storage is append-only, so every change costs about the size of the change, not the size
of the collection:
  {collection}.f32           raw float32 vectors, appended per add()
  {collection}.jsonl         one record (text + metadata) per line, appended per add()
  {collection}.deleted.json  row numbers of deleted chunks ("tombstones")
Deleting a document, or replacing a chunk (re-ingest), only marks the old rows as deleted and
appends the new ones; searches skip deleted rows. When more than COMPACT_RATIO of the rows
are deleted, the files are compacted (rewritten without them) once.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from typing import Dict, Iterator, List, Set, Tuple

import numpy as np

from src.chunking.base import Chunk
from src.vectorstores.base import SearchResult, VectorStore, matches

COMPACT_RATIO = 0.25


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
        self._dead_path = os.path.join(self._dir, f"{collection}.deleted.json")
        self._records: List[dict] = []
        self._pos: Dict[str, int] = {}  # chunk_id -> live row
        self._deleted: Set[int] = set()  # rows still in the index/files but no longer live
        self._index = faiss.IndexFlatIP(dimension)
        self._lock = threading.RLock()  # shared instance: requests may add/search concurrently
        self._load()

    # ---------- persistence ----------

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
        if os.path.exists(self._dead_path):
            with open(self._dead_path, encoding="utf-8") as f:
                self._deleted = {i for i in json.load(f) if i < n}
        self._pos = {r["metadata"]["chunk_id"]: i for i, r in enumerate(self._records) if i not in self._deleted}
        if n != len(records) or n != len(vecs):
            self._compact()

    def _append(self, mat: np.ndarray, records: List[dict]) -> None:
        with open(self._vec_path, "ab") as f:
            mat.astype("float32").tofile(f)
        with open(self._meta_path, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)

    def _save_deleted(self) -> None:
        if self._deleted:
            with open(self._dead_path, "w", encoding="utf-8") as f:
                json.dump(sorted(self._deleted), f)
        elif os.path.exists(self._dead_path):
            os.remove(self._dead_path)

    def _mark_deleted(self, rows: Set[int]) -> None:
        """Tombstone rows; compact once too many rows are dead."""
        self._deleted |= rows
        if len(self._deleted) > COMPACT_RATIO * len(self._records):
            self._compact()
        else:
            self._save_deleted()

    def _compact(self) -> None:
        """Rewrite index and files with live rows only (amortized: runs rarely)."""
        keep = np.array([i for i in range(len(self._records)) if i not in self._deleted], dtype="int64")
        vecs = self._index.reconstruct_n(0, self._index.ntotal)[keep] if len(keep) else np.empty((0, self.dimension), "float32")
        self._records = [self._records[i] for i in keep]
        self._deleted = set()
        self._pos = {r["metadata"]["chunk_id"]: i for i, r in enumerate(self._records)}
        self._index = self._faiss.IndexFlatIP(self.dimension)
        if len(vecs):
            self._index.add(np.ascontiguousarray(vecs, dtype="float32"))
        for p in (self._vec_path, self._meta_path):
            if os.path.exists(p):
                os.remove(p)
        self._append(vecs, self._records)
        self._save_deleted()

    # ---------- operations ----------

    def _add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        with self._lock:
            self._check(chunks, vectors)
            if not chunks:
                return
            mat = _normalize(np.asarray(vectors, dtype="float32"))
            records = [{"text": c.text, "metadata": c.metadata} for c in chunks]
            replaced = {self._pos[c.metadata["chunk_id"]] for c in chunks if c.metadata["chunk_id"] in self._pos}
            start = len(self._records)
            self._index.add(mat)
            self._records.extend(records)
            self._append(mat, records)
            self._pos.update({r["metadata"]["chunk_id"]: start + i for i, r in enumerate(records)})
            if replaced:  # upsert: the old versions become tombstones
                self._mark_deleted(replaced)

    def _delete_source(self, source: str) -> int:
        with self._lock:
            rows = {i for cid, i in self._pos.items() if self._records[i]["metadata"].get("source") == source}
            for i in rows:
                self._pos.pop(self._records[i]["metadata"]["chunk_id"], None)
            if rows:
                self._mark_deleted(rows)
            return len(rows)

    def _live(self) -> List[int]:
        return [i for i in range(len(self._records)) if i not in self._deleted]

    def _iter_records(self) -> Iterator[Tuple[str, dict]]:
        with self._lock:  # snapshot, so a concurrent write can't change the list mid-iteration
            return iter([(self._records[i]["text"], self._records[i]["metadata"]) for i in self._live()])

    def _list_sources(self) -> Dict[str, int]:
        with self._lock:
            return dict(Counter(self._records[i]["metadata"].get("source") for i in self._live()))

    def _search(self, vector: List[float], k: int, filters: Dict[str, list]) -> List[SearchResult]:
        with self._lock:
            live = self.count()
            if live == 0:
                return []
            q = _normalize(np.asarray([vector], dtype="float32"))
            params = None
            limit = min(k, live)
            if filters:  # restrict the exact search to matching live rows (FAISS ID selector)
                allowed = np.array([i for i in self._live() if matches(self._records[i]["metadata"], filters)], dtype="int64")
                if not len(allowed):
                    return []
                params = self._faiss.SearchParameters()
                params.sel = self._faiss.IDSelectorBatch(allowed)
                limit = min(k, len(allowed))
            elif self._deleted:  # skip tombstoned rows
                params = self._faiss.SearchParameters()
                params.sel = self._faiss.IDSelectorNot(self._faiss.IDSelectorBatch(np.array(sorted(self._deleted), dtype="int64")))
            scores, ids = self._index.search(q, limit, params=params) if params else self._index.search(q, limit)
            return [
                SearchResult(text=self._records[i]["text"], metadata=self._records[i]["metadata"], score=float(s))
                for s, i in zip(scores[0], ids[0])
                if i != -1
            ]

    def count(self) -> int:
        return int(self._index.ntotal) - len(self._deleted)

    def _reset(self) -> None:
        with self._lock:
            self._index = self._faiss.IndexFlatIP(self.dimension)
            self._records, self._pos, self._deleted = [], {}, set()
            for p in (self._vec_path, self._meta_path, self._dead_path):
                if os.path.exists(p):
                    os.remove(p)
