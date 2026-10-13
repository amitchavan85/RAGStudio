"""Roadmap P2: keyword search (BM25), rank fusion, filters, reranking."""
import os

import pytest
from helpers import PDF_DIR, HashEmbedder

from src.chunking import get_chunker
from src.ingestion.loader import load_pdf
from src.pipeline import RAGPipeline
from src.retrieval import BM25Index, rrf_fuse, tokenize
from src.vectorstores.base import SearchResult, normalize_filters
from src.vectorstores.chroma_store import ChromaStore
from src.vectorstores.faiss_store import FaissStore
from src.vectorstores.qdrant_store import QdrantStore

USA = os.path.join(PDF_DIR, "usa_gdp_report.pdf")
IND = os.path.join(PDF_DIR, "ind_gdp_report.pdf")


# ---------- units ----------

def test_tokenize_keeps_numbers_and_codes_whole():
    assert tokenize("GDP was 3,170 bn (+7.2%) in 2021; error E27") == ["gdp", "3,170", "bn", "7.2", "2021", "error", "e27"]


def test_bm25_ranks_exact_term_first_and_respects_filters():
    idx = BM25Index([
        ("Inverter error E27 means low string voltage.", {"chunk_id": "a", "source": "manual.pdf"}),
        ("Clean the panels every six months.", {"chunk_id": "b", "source": "manual.pdf"}),
        ("Error E27 also appears in the other guide.", {"chunk_id": "c", "source": "guide.pdf"}),
    ])
    assert idx.search("E27 voltage", k=1)[0].metadata["chunk_id"] == "a"  # only "a" has both terms
    only_guide = idx.search("E27", k=5, filters={"source": "guide.pdf"})
    assert [r.metadata["chunk_id"] for r in only_guide] == ["c"]
    assert idx.search("completely unrelated words", k=5) == []


def test_rrf_rewards_agreement_between_lists():
    def r(cid):
        return SearchResult(text=cid, metadata={"chunk_id": cid})

    fused = rrf_fuse([[r("x"), r("y"), r("z")], [r("y"), r("x"), r("w")]], k=4)
    assert {fused[0].metadata["chunk_id"], fused[1].metadata["chunk_id"]} == {"x", "y"}  # in both lists
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert len(fused) == 4


@pytest.mark.parametrize("bad", [{"bad key": "x"}, {"source": []}, {"source": {"nested": 1}}, {"x; DROP": 1}])
def test_invalid_filters_rejected(bad):
    with pytest.raises(ValueError):
        normalize_filters(bad)


# ---------- filters + iter_records on every local store ----------

@pytest.fixture(params=["chroma", "faiss", "qdrant"])
def filled_store(request, tmp_path):
    emb = HashEmbedder()
    chunks = get_chunker("recursive", 300, 30).chunk_documents(load_pdf(USA) + load_pdf(IND))
    store = {
        "chroma": lambda: ChromaStore("filter_test", emb.dimension, data_dir=str(tmp_path)),
        "faiss": lambda: FaissStore("filter_test", emb.dimension, data_dir=str(tmp_path)),
        "qdrant": lambda: QdrantStore("filter_test", emb.dimension, url=":memory:"),
    }[request.param]()
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return store, chunks, emb


def test_store_filter_by_source_page_and_list(filled_store):
    store, chunks, emb = filled_store
    q = emb.embed_query("GDP growth")
    hits = store.search(q, k=50, filters={"source": "ind_gdp_report.pdf"})
    assert hits and {h.metadata["source"] for h in hits} == {"ind_gdp_report.pdf"}
    hits = store.search(q, k=50, filters={"source": "usa_gdp_report.pdf", "page": 2})
    assert hits and all(h.metadata["source"] == "usa_gdp_report.pdf" and h.metadata["page"] == 2 for h in hits)
    both = store.search(q, k=50, filters={"source": ["usa_gdp_report.pdf", "ind_gdp_report.pdf"]})
    assert len(both) == len(chunks)
    assert store.search(q, k=5, filters={"source": "nope.pdf"}) == []
    assert all("vector" in h.scores for h in hits)


def test_iter_records_returns_every_chunk(filled_store):
    store, chunks, _ = filled_store
    records = list(store.iter_records())
    assert len(records) == len(chunks) == store.count()
    assert {m["chunk_id"] for _, m in records} == {c.metadata["chunk_id"] for c in chunks}


# ---------- pipeline ----------

@pytest.fixture
def rag(tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    p = RAGPipeline(vector_store="faiss", embedding_provider="sentence_transformers",
                    collection_name="p2test", embedder=HashEmbedder())
    p.ingest([USA, IND])
    yield p
    get_settings.cache_clear()


def test_hybrid_is_default_and_reports_stage_scores(rag):
    assert rag.config.retrieval_mode == "hybrid"
    hits = rag.query("Indian Rupee New Delhi", k=3)
    assert hits[0].metadata["source"] == "ind_gdp_report.pdf"
    assert "rrf" in hits[0].scores and ("vector" in hits[0].scores or "keyword" in hits[0].scores)


def test_keyword_mode_does_not_embed_the_question(rag, monkeypatch):
    monkeypatch.setattr(rag.embedder, "embed_query", lambda q: (_ for _ in ()).throw(AssertionError("embedded")))
    hits = rag.query("27,360", k=1, mode="keyword")
    assert hits[0].metadata["source"] == "usa_gdp_report.pdf" and "27,360" in hits[0].text


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_filters_apply_in_every_mode(rag, mode):
    hits = rag.query("GDP growth", k=10, mode=mode, filters={"source": "usa_gdp_report.pdf"})
    assert hits and {h.metadata["source"] for h in hits} == {"usa_gdp_report.pdf"}


def test_keyword_index_follows_delete_and_reingest(rag):
    assert rag.query("27,360", k=1, mode="keyword")[0].metadata["source"] == "usa_gdp_report.pdf"
    rag.delete_document("usa_gdp_report.pdf")
    assert all(h.metadata["source"] != "usa_gdp_report.pdf" for h in rag.query("27,360", k=5, mode="keyword"))
    rag.ingest(USA)
    assert rag.query("27,360", k=1, mode="keyword")[0].metadata["source"] == "usa_gdp_report.pdf"


def test_pipelines_in_one_process_share_the_store(rag):
    other = RAGPipeline(rag.config, embedder=rag.embedder)  # e.g. another API request
    assert other.store is rag.store
    other.delete_document("ind_gdp_report.pdf")
    assert "ind_gdp_report.pdf" not in rag.list_documents()
    assert all(h.metadata["source"] != "ind_gdp_report.pdf" for h in rag.query("Rupee", k=5, mode="keyword"))


def test_keyword_index_notices_writes_from_another_process(tmp_path, monkeypatch):
    """A second store object on the same Chroma files stands in for another process."""
    from src.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        rag = RAGPipeline(vector_store="chroma", embedding_provider="sentence_transformers",
                          collection_name="xproc", embedder=HashEmbedder())
        rag.ingest([USA, IND])
        assert rag.query("Rupee", k=1, mode="keyword")[0].metadata["source"] == "ind_gdp_report.pdf"
        outsider = ChromaStore(rag.collection, rag.embedder.dimension, data_dir=str(tmp_path))
        outsider.delete_source("ind_gdp_report.pdf")  # bypasses rag entirely
        assert all(h.metadata["source"] != "ind_gdp_report.pdf" for h in rag.query("Rupee", k=5, mode="keyword"))
    finally:
        get_settings.cache_clear()


def test_invalid_mode_and_reranker_rejected(rag):
    with pytest.raises(ValueError):
        rag.query("GDP", mode="fuzzy")
    with pytest.raises(ValueError):
        rag.query("GDP", reranker="gpt")
