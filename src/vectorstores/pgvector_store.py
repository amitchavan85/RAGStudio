"""pgvector: vectors in Postgres (needs the `vector` extension; see docker-compose.yml)."""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

from src.chunking.base import Chunk
from src.vectorstores.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class PgVectorStore(VectorStore):
    def __init__(self, collection: str, dimension: int, dsn: str = "postgresql://postgres:postgres@localhost:5432/rag",
                 ef_search: int = 100):
        """ef_search: HNSW candidate list size per query. pgvector's default (40) dropped
        recall@10 to ~0.81 at 50k chunks; 100 restores ~1.0 for a small latency cost."""
        super().__init__(collection, dimension)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", collection):
            raise ValueError("collection must be a valid SQL identifier for pgvector")
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("SET hnsw.ef_search = %s", (int(ef_search),))
            try:  # pgvector >= 0.8: keep scanning the index until k rows pass a WHERE filter
                cur.execute("SET hnsw.iterative_scan = relaxed_order")
            except Exception:
                pass  # older pgvector: filtered searches may return fewer than k rows
        register_vector(self._conn)
        self._create_table()

    def _create_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.collection} ("
                f"chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, "
                f"metadata JSONB NOT NULL, embedding vector({self.dimension}))"
            )
            self._create_vector_index(cur)
            cur.execute(  # fast delete/list by document
                f"CREATE INDEX IF NOT EXISTS {self.collection}_source ON {self.collection} ((metadata->>'source'))"
            )

    def _create_vector_index(self, cur) -> None:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {self.collection}_hnsw ON {self.collection} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )

    @contextmanager
    def bulk(self) -> Iterator[None]:
        """First load into an empty table: drop the HNSW index, insert everything, build it once.

        pgvector's own advice: building the index after loading is much faster than updating it
        on every insert. Only for empty tables, so searches on a live index are never affected.
        """
        if self.count() > 0:
            yield
            return
        with self._conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {self.collection}_hnsw")
        try:
            yield
        finally:
            self._rebuild_vector_index()

    def _rebuild_vector_index(self) -> None:
        """Build the HNSW index after a bulk load; never leave the table without it."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '512MB'")  # room to build the graph in memory (fast)
                self._create_vector_index(cur)
        except Exception as e:  # e.g. too little shared memory for a parallel build (Docker's 64 MB /dev/shm)
            logger.warning("fast HNSW build failed (%s: %s); retrying with default settings", type(e).__name__, e)
            with self._conn.cursor() as cur:
                cur.execute("RESET maintenance_work_mem")
                cur.execute("SET max_parallel_maintenance_workers = 0")  # no shared-memory workers
                self._create_vector_index(cur)
                cur.execute("RESET max_parallel_maintenance_workers")

    def _add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        self._check(chunks, vectors)
        import numpy as np
        from psycopg2.extras import Json, execute_values

        rows = [
            (c.metadata["chunk_id"], c.text, Json(c.metadata), np.asarray(v, dtype="float32"))
            for c, v in zip(chunks, vectors)
        ]
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                f"INSERT INTO {self.collection} (chunk_id, text, metadata, embedding) VALUES %s "
                f"ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, "
                f"metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                rows,
                page_size=1000,
            )

    def _search(self, vector: List[float], k: int, filters: Dict[str, list]) -> List[SearchResult]:
        import numpy as np

        q = np.asarray(vector, dtype="float32")
        # Keys were validated by normalize_filters; values are always passed as parameters.
        where = " AND ".join(f"metadata->>'{key}' = ANY(%s)" for key in filters)
        params = [[str(v) for v in vals] for vals in filters.values()]
        sql = (f"SELECT text, metadata, 1 - (embedding <=> %s) AS score FROM {self.collection} "
               f"{'WHERE ' + where if where else ''} ORDER BY embedding <=> %s LIMIT %s")
        with self._conn.cursor() as cur:
            cur.execute(sql, (q, *params, q, k))
            return [SearchResult(text=t, metadata=m, score=float(s)) for t, m, s in cur.fetchall()]

    def _iter_records(self) -> Iterator[Tuple[str, dict]]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT text, metadata FROM {self.collection}")
            while rows := cur.fetchmany(2000):
                yield from rows

    def _delete_source(self, source: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.collection} WHERE metadata->>'source' = %s", (source,))
            return cur.rowcount

    def _list_sources(self) -> Dict[str, int]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT metadata->>'source', count(*) FROM {self.collection} GROUP BY 1")
            return {s: int(n) for s, n in cur.fetchall()}

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.collection}")
            return int(cur.fetchone()[0])

    def _reset(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.collection}")
        self._create_table()
