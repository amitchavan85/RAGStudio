"""
Scale benchmark for the vector stores.

    python -m src.benchmark --n 10000
    python -m src.benchmark --n 50000 --stores faiss chroma

Real chunks from the PDFs are embedded once with the real model; the corpus is then
grown to N by adding small noise to those vectors (so N distinct vectors, realistic
geometry, no N-times embedding cost). Each store is measured on:
  - ingest:   add N chunks in batches of --batch (like ingesting many documents)
  - query:    p50 / p95 latency of k=10 searches
  - recall@10 vs exact brute-force cosine (HNSW indexes are approximate)
  - upsert:   re-adding 100 existing chunks into the full collection
Live stores (qdrant, pgvector) need `docker compose up -d`. Collections are dropped afterwards.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

from src.chunking import get_chunker
from src.config import get_settings
from src.chunking.base import Chunk
from src.embeddings import get_embedder
from src.ingestion.loader import load_pdf_directory
from src.logging_config import setup_logging
from src.vectorstores import get_vector_store


def build_corpus(n: int, seed: int = 0):
    docs = load_pdf_directory()
    base = get_chunker("recursive", 500, 50).chunk_documents(docs)
    emb = get_embedder()
    t = time.perf_counter()
    base_vecs = np.asarray(emb.embed_documents([c.text for c in base]), dtype="float32")
    embed_rate = len(base) / (time.perf_counter() - t)

    rng = np.random.default_rng(seed)
    idx = np.arange(n) % len(base)
    vecs = base_vecs[idx] + rng.normal(0, 0.03, size=(n, base_vecs.shape[1])).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [
        Chunk(base[i].text, {**base[i].metadata, "chunk_id": f"bench::{j}", "chunk_index": j})
        for j, i in enumerate(idx)
    ]
    queries = vecs[rng.choice(n, 50, replace=False)] + rng.normal(0, 0.05, size=(50, vecs.shape[1])).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return chunks, vecs, queries, len(base), embed_rate, emb.dimension


def run_store(name, chunks, vecs, queries, dim, batch):
    store = get_vector_store(name, dim, collection=f"bench_{name}")
    store.reset()
    try:
        t = time.perf_counter()
        for i in range(0, len(chunks), batch):
            store.add(chunks[i : i + batch], vecs[i : i + batch].tolist())
        ingest_s = time.perf_counter() - t
        assert store.count() == len(chunks), f"{name}: count {store.count()} != {len(chunks)}"

        exact = np.argsort(-(vecs @ queries.T), axis=0)[:10].T  # true top-10 per query
        lat, recall = [], []
        for q, truth in zip(queries, exact):
            t = time.perf_counter()
            hits = store.search(q.tolist(), k=10)
            lat.append((time.perf_counter() - t) * 1000)
            got = {h.metadata["chunk_id"] for h in hits}
            recall.append(len(got & {f"bench::{j}" for j in truth}) / 10)

        t = time.perf_counter()
        store.add(chunks[:100], vecs[:100].tolist())
        upsert_ms = (time.perf_counter() - t) * 1000
        assert store.count() == len(chunks), f"{name}: upsert duplicated rows"

        return {
            "store": name,
            "ingest_s": round(ingest_s, 1),
            "chunks_per_s": int(len(chunks) / ingest_s),
            "query_p50_ms": round(float(np.percentile(lat, 50)), 1),
            "query_p95_ms": round(float(np.percentile(lat, 95)), 1),
            "recall@10": round(float(np.mean(recall)), 3),
            "upsert100_ms": int(upsert_ms),
        }
    finally:
        store.reset()  # leave no benchmark data behind
        if name == "qdrant":
            store._client.delete_collection(store.collection)
        if name == "pgvector":
            with store._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {store.collection}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--stores", nargs="+", default=["chroma", "faiss", "qdrant", "pgvector"])
    args = ap.parse_args()

    setup_logging(level="WARNING")  # console stays quiet; warnings/errors still go to LOG_DIR/rag.log
    os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="rag_bench_"))

    chunks, vecs, queries, n_real, embed_rate, dim = build_corpus(args.n)
    print(f"corpus: {args.n} chunks grown from {n_real} real chunks | dim={dim} | "
          f"embedding speed {embed_rate:.0f} chunks/s ({get_settings().embedding_provider})")
    print(f"{'store':9} {'ingest_s':>8} {'chunks/s':>9} {'p50_ms':>7} {'p95_ms':>7} {'recall@10':>9} {'upsert100_ms':>12}")
    for name in args.stores:
        try:
            r = run_store(name, chunks, vecs, queries, dim, args.batch)
            print(f"{r['store']:9} {r['ingest_s']:>8} {r['chunks_per_s']:>9} {r['query_p50_ms']:>7} "
                  f"{r['query_p95_ms']:>7} {r['recall@10']:>9} {r['upsert100_ms']:>12}")
        except Exception as e:
            print(f"{name:9} FAILED: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
