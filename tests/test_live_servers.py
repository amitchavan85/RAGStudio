"""Live tests for Qdrant and pgvector against the docker-compose servers.

Each test uses a unique throwaway collection (dropped afterwards). Tests
skip automatically when the server is not reachable.
"""
import os
import uuid

import pytest
from helpers import HashEmbedder, PDF_DIR

from src.chunking import get_chunker
from src.chunking.base import Chunk
from src.config import Settings
from src.ingestion.loader import load_pdf_directory
from src.vectorstores.pgvector_store import PgVectorStore
from src.vectorstores.qdrant_store import QdrantStore

S = Settings()
DIM = HashEmbedder.dim


def _qdrant(name):
    return QdrantStore(name, DIM, url=S.qdrant_url, api_key=S.qdrant_api_key.get_secret_value())


def _pg(name):
    return PgVectorStore(name, DIM, dsn=S.pg_dsn)


def _drop_pg(store):
    with store._conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {store.collection}")
    store._conn.close()


def _drop_qdrant(store):
    store._client.delete_collection(store.collection)


BACKENDS = {"qdrant": (_qdrant, _drop_qdrant), "pgvector": (_pg, _drop_pg)}


@pytest.fixture(params=list(BACKENDS))
def live(request):
    make, drop = BACKENDS[request.param]
    name = f"live_{uuid.uuid4().hex[:10]}"
    try:
        store = make(name)
    except Exception as e:  # server down / not configured
        pytest.skip(f"{request.param} server not reachable: {type(e).__name__}")
    yield request.param, store, make
    drop(store)


@pytest.fixture(scope="module")
def corpus():
    emb = HashEmbedder()
    chunks = get_chunker("recursive", 400, 40).chunk_documents(load_pdf_directory(PDF_DIR))
    return chunks, emb.embed_documents([c.text for c in chunks]), emb


def test_starts_empty(live):
    _, store, _ = live
    assert store.count() == 0
    assert store.search([0.1] * DIM, k=3) == []


def test_add_search_roundtrip_preserves_text_and_metadata(live, corpus):
    _, store, _ = live
    chunks, vecs, emb = corpus
    store.add(chunks, vecs)
    assert store.count() == len(chunks)

    target = chunks[5]
    hits = store.search(emb.embed_query(target.text), k=3)
    assert len(hits) == 3
    assert hits[0].text == target.text
    assert hits[0].metadata["chunk_id"] == target.metadata["chunk_id"]
    assert hits[0].metadata["source"] == target.metadata["source"]
    assert hits[0].metadata["page"] == target.metadata["page"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)  # identical vector -> cosine 1
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_upsert_does_not_duplicate_and_updates_text(live, corpus):
    _, store, _ = live
    chunks, vecs, _ = corpus
    store.add(chunks[:5], vecs[:5])
    store.add(chunks[:5], vecs[:5])
    assert store.count() == 5

    changed = Chunk("replacement text", dict(chunks[0].metadata))
    store.add([changed], [vecs[0]])
    assert store.count() == 5
    assert store.search(vecs[0], k=1)[0].text == "replacement text"


def test_k_larger_than_collection(live, corpus):
    _, store, _ = live
    chunks, vecs, _ = corpus
    store.add(chunks[:3], vecs[:3])
    assert len(store.search(vecs[0], k=50)) == 3


def test_reset_empties_collection_and_it_stays_usable(live, corpus):
    _, store, _ = live
    chunks, vecs, _ = corpus
    store.add(chunks[:4], vecs[:4])
    store.reset()
    assert store.count() == 0
    store.add(chunks[:2], vecs[:2])
    assert store.count() == 2


def test_data_persists_across_new_connection(live, corpus):
    name, store, make = live
    chunks, vecs, _ = corpus
    store.add(chunks[:6], vecs[:6])
    reopened = make(store.collection)
    try:
        assert reopened.count() == 6
        assert reopened.search(vecs[2], k=1)[0].text == chunks[2].text
    finally:
        if name == "pgvector":
            reopened._conn.close()


def test_top_k_matches_brute_force_cosine(live, corpus):
    """The server's ranking must equal an exact cosine ranking computed locally."""
    import numpy as np

    _, store, _ = live
    chunks, vecs, emb = corpus
    store.add(chunks, vecs)
    q = np.asarray(emb.embed_query("India GDP growth rate"))
    m = np.asarray(vecs)
    sims = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-12)
    expected = np.argsort(-sims)[:5]

    hits = store.search(q.tolist(), k=5)
    assert [h.score for h in hits] == pytest.approx([sims[i] for i in expected], abs=1e-4)
    assert {h.metadata["chunk_id"] for h in hits[:2]} == {chunks[i].metadata["chunk_id"] for i in expected[:2]}


def test_list_and_delete_by_source(live, corpus):
    _, store, _ = live
    chunks, vecs, _ = corpus
    store.add(chunks, vecs)
    by_source = store.list_sources()
    assert sum(by_source.values()) == len(chunks)
    victim = "usa_gdp_report.pdf"
    assert store.delete_source(victim) == by_source[victim]
    assert victim not in store.list_sources()
    assert store.count() == len(chunks) - by_source[victim]
    assert store.delete_source(victim) == 0


def test_filtered_search_and_iter_records(live, corpus):
    _, store, _ = live
    chunks, vecs, emb = corpus
    store.add(chunks, vecs)
    q = emb.embed_query("GDP growth")
    hits = store.search(q, k=50, filters={"source": "ind_gdp_report.pdf"})
    assert hits and {h.metadata["source"] for h in hits} == {"ind_gdp_report.pdf"}
    hits = store.search(q, k=50, filters={"source": ["usa_gdp_report.pdf", "ind_gdp_report.pdf"], "page": 1})
    assert hits and all(h.metadata["page"] == 1 for h in hits)
    assert store.search(q, k=5, filters={"source": "nope.pdf"}) == []
    assert len(list(store.iter_records())) == len(chunks)


def test_bulk_load_builds_index_once_and_keeps_search_exact(live, corpus):
    """pgvector: bulk() on an empty table drops the HNSW index, then rebuilds it; other stores: no-op."""
    import numpy as np

    name, store, _ = live
    chunks, vecs, emb = corpus
    with store.bulk():
        if name == "pgvector":
            with store._conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname = %s", (f"{store.collection}_hnsw",))
                assert cur.fetchone()[0] == 0  # index dropped during the load
        for i in range(0, len(chunks), 200):
            store.add(chunks[i:i + 200], vecs[i:i + 200])
    if name == "pgvector":
        with store._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname = %s", (f"{store.collection}_hnsw",))
            assert cur.fetchone()[0] == 1  # rebuilt once at the end
    assert store.count() == len(chunks)
    q = np.asarray(emb.embed_query("India GDP growth rate"))
    m = np.asarray(vecs)
    sims = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-12)
    hits = store.search(q.tolist(), k=5)
    assert [h.score for h in hits] == pytest.approx(sorted(sims, reverse=True)[:5], abs=1e-4)
    with store.bulk():  # non-empty table: index must stay in place
        store.add(chunks[:3], vecs[:3])
    assert store.count() == len(chunks)
