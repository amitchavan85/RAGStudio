"""Retrieval quality with the real local models (no API key needed).

Locks in what the P2 comparison showed on the GDP samples: plain vector search misses an
exact-figure lookup that hybrid search finds. Skipped if the models cannot be loaded.
"""
import os

import pytest
from helpers import PDF_DIR

from src.pipeline import RAGPipeline

FILES = [os.path.join(PDF_DIR, f) for f in sorted(os.listdir(PDF_DIR))
         if f.endswith("_gdp_report.pdf") or f == "global_gdp_comparison.pdf"]


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    from src.config import get_settings

    old = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(tmp_path_factory.mktemp("real"))
    get_settings.cache_clear()
    try:
        p = RAGPipeline(vector_store="faiss", embedding_provider="sentence_transformers", collection_name="realtest")
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {type(e).__name__}")
    p.ingest(FILES)
    yield p
    if old is None:
        os.environ.pop("DATA_DIR", None)
    else:
        os.environ["DATA_DIR"] = old
    get_settings.cache_clear()


def _sources(hits):
    return [h.metadata["source"] for h in hits]


def test_hybrid_finds_exact_figure_that_vector_search_misses(rag):
    # USA 2023 nominal GDP. Embeddings treat "27,360" as a fuzzy number; BM25 matches it exactly.
    assert "usa_gdp_report.pdf" not in _sources(rag.query("27,360", k=3, mode="vector"))
    assert "usa_gdp_report.pdf" in _sources(rag.query("27,360", k=3, mode="hybrid"))


def test_hybrid_keeps_semantic_questions_working(rag):
    for question, source in [
        ("Which economy relies on agricultural commodities and iron ore exports?", "bra_gdp_report.pdf"),
        ("Which country became the most populous in the world?", "ind_gdp_report.pdf"),
        ("What does Germany mainly export?", "deu_gdp_report.pdf"),
    ]:
        assert source in _sources(rag.query(question, k=3, mode="hybrid")), question


def test_cross_encoder_reranker_runs_and_scores(rag):
    try:
        hits = rag.query("Which country became the most populous in the world?", k=3, reranker="cross_encoder")
    except Exception as e:
        pytest.skip(f"reranker model unavailable: {type(e).__name__}")
    assert hits[0].metadata["source"] == "ind_gdp_report.pdf"
    assert [h.scores["rerank"] for h in hits] == sorted((h.scores["rerank"] for h in hits), reverse=True)
