"""Roadmap P1: pipeline orchestration (entry points, collection per model, document replace/delete)."""
import os
import shutil

import pytest
from helpers import PDF_DIR, HashEmbedder

from src.pipeline import PipelineConfig, RAGPipeline, collection_name

USA = os.path.join(PDF_DIR, "usa_gdp_report.pdf")
IND = os.path.join(PDF_DIR, "ind_gdp_report.pdf")


class OtherDimEmbedder(HashEmbedder):
    dim = 32
    model_name = "other-model"


@pytest.fixture(params=["chroma", "faiss", "qdrant"])
def rag(request, tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    yield RAGPipeline(vector_store=request.param, embedding_provider="sentence_transformers",
                      chunking_strategy="recursive", collection_name="p1test", embedder=HashEmbedder())
    get_settings.cache_clear()


# ---------- config: defaults + overrides ----------

def test_config_defaults_come_from_settings():
    from src.config import get_settings

    c = PipelineConfig.from_settings()
    assert c.vector_store == get_settings().vector_store
    assert c.chunking_strategy == get_settings().chunking_strategy


def test_overrides_apply_and_none_means_default():
    c = PipelineConfig.from_settings(vector_store="faiss", chunk_size=None)
    assert c.vector_store == "faiss"
    assert c.chunk_size == PipelineConfig.from_settings().chunk_size


@pytest.mark.parametrize("bad", [
    {"vector_store": "mongo"}, {"chunking_strategy": "magic"}, {"embedding_provider": "x"},
    {"chunk_size": 100, "chunk_overlap": 100}, {"collection_name": "bad name;"}, {"not_an_option": 1},
])
def test_invalid_options_are_rejected(bad):
    with pytest.raises(ValueError):
        PipelineConfig.from_settings(**bad)


# ---------- one collection per embedding model ----------

def test_collection_name_includes_model_and_dimension():
    assert collection_name("rag_docs", HashEmbedder()) == "rag_docs_hash_test_64"
    assert collection_name("rag_docs", OtherDimEmbedder()) == "rag_docs_other_model_32"


def test_collection_name_is_sql_safe_and_short():
    class Long(HashEmbedder):
        model_name = "org/" + "very-long-model-name-" * 5

    name = collection_name("rag_docs", Long())
    assert len(name) <= 50 and name.replace("_", "").isalnum() and name[0].isalpha()


def test_different_embedders_never_share_a_collection(tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        a = RAGPipeline(vector_store="faiss", embedding_provider="sentence_transformers", embedder=HashEmbedder())
        b = RAGPipeline(vector_store="faiss", embedding_provider="sentence_transformers", embedder=OtherDimEmbedder())
        a.ingest(USA)
        assert a.collection != b.collection
        assert b.list_documents() == {}  # b's collection is untouched by a's vectors
    finally:
        get_settings.cache_clear()


# ---------- ingest / query / list / delete ----------

def test_ingest_query_list(rag):
    report = rag.ingest([USA, IND])
    assert [f["source"] for f in report.files] == ["usa_gdp_report.pdf", "ind_gdp_report.pdf"]
    assert all(f["replaced"] == 0 for f in report.files)
    docs = rag.list_documents()
    assert set(docs) == {"usa_gdp_report.pdf", "ind_gdp_report.pdf"}
    assert sum(docs.values()) == report.total_chunks == rag.store.count()
    hits = rag.query("India Indian Rupee New Delhi", k=3)
    assert hits and hits[0].metadata["source"] == "ind_gdp_report.pdf"


def test_reingest_with_different_chunker_leaves_no_stale_chunks(rag):
    first = rag.ingest(USA)
    smaller = RAGPipeline(rag.config, embedder=rag.embedder, chunk_size=150, chunk_overlap=20)
    smaller._store = rag.store  # same collection, same store instance
    second = smaller.ingest(USA)
    assert second.files[0]["replaced"] == first.total_chunks
    assert second.total_chunks > first.total_chunks  # smaller chunks -> more of them
    assert rag.list_documents() == {"usa_gdp_report.pdf": second.total_chunks}
    assert rag.store.count() == second.total_chunks  # nothing from the first ingest survives


def test_delete_document_removes_only_that_document(rag):
    rag.ingest([USA, IND])
    usa_chunks = rag.list_documents()["usa_gdp_report.pdf"]
    assert rag.delete_document("usa_gdp_report.pdf") == usa_chunks
    assert set(rag.list_documents()) == {"ind_gdp_report.pdf"}
    assert all(h.metadata["source"] == "ind_gdp_report.pdf" for h in rag.query("GDP", k=10))
    assert rag.delete_document("usa_gdp_report.pdf") == 0  # deleting again is a no-op


def test_ingest_folder_and_skip_unreadable(rag, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    shutil.copy(USA, folder / "a.pdf")
    (folder / "broken.pdf").write_bytes(b"not a pdf")
    report = rag.ingest(str(folder))
    assert [f["source"] for f in report.files] == ["a.pdf"]
    assert [s["source"] for s in report.skipped] == ["broken.pdf"]


def test_duplicate_file_names_are_rejected(rag, tmp_path):
    other = tmp_path / "x"
    other.mkdir()
    shutil.copy(USA, other / "usa_gdp_report.pdf")
    with pytest.raises(ValueError, match="more than once"):
        rag.ingest([USA, str(other / "usa_gdp_report.pdf")])


def test_failed_ingest_keeps_previous_version(rag, monkeypatch):
    rag.ingest(USA)
    before = rag.list_documents()

    def boom(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(rag.embedder, "embed_documents", boom)
    with pytest.raises(RuntimeError):
        rag.ingest(USA)
    assert rag.list_documents() == before  # old chunks were not deleted


def test_empty_question_rejected(rag):
    with pytest.raises(ValueError):
        rag.query("   ")


class AuthenticationError(Exception):
    """Stands in for openai.AuthenticationError (matched by class name)."""


def test_embedding_failure_is_reported_as_embedding_error_with_a_fix(rag, monkeypatch):
    from src.embeddings import EmbeddingError

    rag.ingest(USA)

    def revoked(*a, **k):
        raise AuthenticationError("Error code: 401 - Your API key has been invalidated.")

    monkeypatch.setattr(rag.embedder, "embed_query", revoked)
    monkeypatch.setattr(rag.embedder, "embed_documents", revoked)
    with pytest.raises(EmbeddingError, match="API key is invalid"):
        rag.query("GDP", mode="hybrid")
    with pytest.raises(EmbeddingError, match="Embedding with .* failed"):
        rag.ingest(IND)
    assert rag.query("27,360", mode="keyword")  # keyword search needs no embeddings and still works
