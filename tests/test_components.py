import os

import pytest

from src.chunking import SemanticChunker, get_chunker
from src.chunking.agentic import AgenticChunker
from src.chunking.base import Chunk
from src.config import Settings
from src.embeddings import get_embedder
from src.ingestion.loader import load_pdf, load_pdf_directory
from src.vectorstores import get_vector_store
from src.vectorstores.base import chunk_uuid
from src.vectorstores.pgvector_store import PgVectorStore
from helpers import HashEmbedder, PDF_DIR



def test_loader_single_pdf_has_page_metadata():
    docs = load_pdf(os.path.join(PDF_DIR, "usa_gdp_report.pdf"))
    assert docs
    assert docs[0].metadata["source"] == "usa_gdp_report.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["doc_id"] == "usa_gdp_report.pdf#p1"


def test_loader_directory_loads_all_eight_pdfs():
    sources = {d.metadata["source"] for d in load_pdf_directory(PDF_DIR)}
    assert len(sources) >= 8


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.vector_store == "chroma"
    assert s.chunking_strategy == "recursive"
    assert s.chunk_overlap < s.chunk_size


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    assert Settings(_env_file=None).vector_store == "qdrant"


def test_unknown_embedder_and_store_rejected():
    with pytest.raises(ValueError):
        get_embedder("nope")
    with pytest.raises(ValueError):
        get_vector_store("nope")


def test_vector_store_factory_builds_each_local_backend(tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    try:
        for name in ("chroma", "faiss", "qdrant"):
            store = get_vector_store(name, dimension=8, collection="factory_test")
            assert store.count() == 0
    finally:
        get_settings.cache_clear()


def test_chunker_requires_its_dependencies():
    with pytest.raises(ValueError):
        get_chunker("semantic")
    with pytest.raises(ValueError):
        get_chunker("agentic")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        get_chunker("fixed", chunk_size=100, overlap=100)


def test_chunk_size_is_respected_by_recursive_chunker():
    text = "Sentence number one is here. " * 200
    chunks = get_chunker("recursive", chunk_size=200, overlap=20).split(text)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_sliding_window_overlap():
    chunks = get_chunker("sliding_window", chunk_size=10, overlap=5).split(" ".join(str(i) for i in range(30)))
    assert chunks[0].split()[5:] == chunks[1].split()[:5]


def test_semantic_chunker_runs_with_fake_embedder():
    text = "Cats purr softly. Cats chase mice. Stocks fell today. Markets dropped sharply."
    parts = SemanticChunker(embedder=HashEmbedder(), breakpoint_percentile=50).split(text)
    assert parts and "".join(parts).replace(" ", "")


def test_agentic_chunker_uses_llm_output():
    chunker = AgenticChunker(llm_fn=lambda p: '```json\n["one", "two"]\n```', chunk_size=10, overlap=1)
    assert chunker.split("anything") == ["one", "two"]


def test_agentic_chunker_falls_back_on_bad_llm_output():
    chunker = AgenticChunker(llm_fn=lambda p: "not json")
    assert chunker.split("Some real text to split.") == ["Some real text to split."]


def test_chunk_ids_are_stable_and_unique():
    docs = load_pdf_directory(PDF_DIR)
    chunker = get_chunker("recursive", 300, 30)
    ids1 = [c.metadata["chunk_id"] for c in chunker.chunk_documents(docs)]
    ids2 = [c.metadata["chunk_id"] for c in chunker.chunk_documents(docs)]
    assert ids1 == ids2
    assert len(ids1) == len(set(ids1))


def test_chunk_uuid_is_deterministic():
    assert chunk_uuid("a::chunk0") == chunk_uuid("a::chunk0")
    assert chunk_uuid("a::chunk0") != chunk_uuid("a::chunk1")


def test_store_rejects_mismatched_vectors(tmp_path):
    from src.vectorstores.faiss_store import FaissStore

    store = FaissStore("mismatch", 4, data_dir=str(tmp_path))
    with pytest.raises(ValueError):
        store.add([Chunk("x", {"chunk_id": "x"})], [])


def test_faiss_persists_across_instances(tmp_path):
    from src.vectorstores.faiss_store import FaissStore

    c = Chunk("hello world", {"chunk_id": "h::0"})
    vec = HashEmbedder().embed_documents(["hello world"])
    FaissStore("persist", HashEmbedder.dim, data_dir=str(tmp_path)).add([c], vec)
    reopened = FaissStore("persist", HashEmbedder.dim, data_dir=str(tmp_path))
    assert reopened.count() == 1
    assert reopened.search(vec[0], k=1)[0].text == "hello world"


def test_pgvector_rejects_unsafe_collection_name():
    with pytest.raises(ValueError):
        PgVectorStore("bad name; DROP TABLE x", 4)


def test_end_to_end_ingest_and_retrieve(tmp_path):
    from src.vectorstores.chroma_store import ChromaStore

    emb = HashEmbedder()
    chunks = get_chunker("sentence", chunk_size=300).chunk_documents(load_pdf_directory(PDF_DIR))
    store = ChromaStore("e2e_test", emb.dimension, data_dir=str(tmp_path))
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    hits = store.search(emb.embed_query("India GDP"), k=3)
    assert len(hits) == 3
    assert all(h.metadata["source"].endswith(".pdf") for h in hits)


def test_openai_key_is_masked_and_loaded_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    s = Settings(_env_file=None)
    assert s.openai_api_key.get_secret_value() == "sk-test-123"
    assert "sk-test-123" not in repr(s) and "sk-test-123" not in str(s)


def test_openai_embedder_requires_key(monkeypatch):
    from src.embeddings.openai_provider import OpenAIEmbeddings

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddings(api_key="")


def test_agentic_rejects_output_that_drops_numbers():
    source = "India GDP was 3,170 billion in 2023, up 7.2%."
    lossy = AgenticChunker(llm_fn=lambda p: '["India GDP was large in 2023."]')
    # falls back to the recursive chunker, which keeps the source text intact
    assert lossy.split(source) == [source]


def test_agentic_accepts_faithful_output():
    source = "India GDP was 3,170 billion in 2023, up 7.2%."
    ok = AgenticChunker(llm_fn=lambda p: '["India GDP was 3,170 billion in 2023.", "It grew 7.2% in 2023."]', chunk_size=40, overlap=5)
    assert len(ok.split(source)) == 2


def test_agentic_parses_json_wrapped_in_commentary():
    chunker = AgenticChunker(llm_fn=lambda p: 'Sure! Here you go:\n["alpha beta", "gamma"]\nHope that helps.', chunk_size=10, overlap=1)
    assert chunker.split("alpha beta gamma") == ["alpha beta", "gamma"]


def test_agentic_merges_tiny_chunks_up_to_chunk_size():
    chunker = AgenticChunker(llm_fn=lambda p: '["a", "b", "c", "d"]', chunk_size=30, overlap=5)
    assert chunker.split("a b c d") == ["a\nb\nc\nd"]


def test_recursive_chunks_never_exceed_chunk_size_on_real_pages():
    docs = load_pdf_directory(PDF_DIR)
    for size, overlap in ((500, 50), (300, 60), (120, 40)):
        chunks = get_chunker("recursive", size, overlap).chunk_documents(docs)
        assert max(len(c.text) for c in chunks) <= size, (size, overlap)


def test_split_units_preserves_text_and_lines():
    from src.chunking.semantic import split_units

    text = "Title line\nYear 2020 2,670 -5.8%\nIt grew fast. Then it slowed down."
    units = split_units(text)
    assert "".join(units) == text
    assert len(units) == 4  # 2 lines + 2 sentences


def test_semantic_chunker_keeps_line_structure_and_numbers():
    text = "Heading A\nCats purr. Cats nap.\nHeading B\nStocks fell 3.5%. Markets dropped 4,200 points."
    parts = SemanticChunker(embedder=HashEmbedder(), breakpoint_percentile=50, chunk_size=200).split(text)
    assert "\n" in "".join(parts) or len(parts) > 1
    assert not AgenticChunker._missing_numbers(text, parts)


def test_pdf_dir_default_resolves_from_project_root_and_loads(monkeypatch):
    from src.config import get_settings

    s = Settings(_env_file=None)
    assert s.pdf_path.is_absolute() and s.pdf_path.as_posix().endswith("src/data/pdfs")
    monkeypatch.setenv("PDF_DIR", PDF_DIR)  # no argument -> the PDF_DIR setting is used
    get_settings.cache_clear()
    try:
        assert len({d.metadata["source"] for d in load_pdf_directory()}) == 8
    finally:
        get_settings.cache_clear()


def test_pdf_dir_env_override(monkeypatch, tmp_path):
    from src.config import get_settings

    monkeypatch.setenv("PDF_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert load_pdf_directory() == []  # empty folder -> nothing loaded
    finally:
        get_settings.cache_clear()


def test_loader_skips_unreadable_pdfs_and_logs_warning(tmp_path, caplog):
    import logging
    import shutil

    shutil.copy(os.path.join(PDF_DIR, "usa_gdp_report.pdf"), tmp_path / "good.pdf")
    (tmp_path / "empty.pdf").write_bytes(b"")
    (tmp_path / "garbage.pdf").write_bytes(b"this is not a pdf")
    with caplog.at_level(logging.WARNING):
        docs = load_pdf_directory(str(tmp_path))
    assert {d.metadata["source"] for d in docs} == {"good.pdf"}
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "empty.pdf" in warned and "garbage.pdf" in warned


def test_document_chunker_never_drops_consecutive_headings_or_number_lines():
    text = "CHAPTER 1\nGLOBAL OUTLOOK\nGrowth slowed this year.\n102682\nRegional View\nAsia grew 5.1%."
    chunks = get_chunker("document", chunk_size=500, overlap=50).split(text)
    joined = "\n".join(chunks)
    for line in text.split("\n"):
        assert line in joined, line
    assert chunks[0].startswith("CHAPTER 1\nGLOBAL OUTLOOK")  # both headings kept as context
