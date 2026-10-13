"""
Document-agnostic pipeline checks on ONE PDF.

Point it at any PDF:
    PowerShell:  $env:RAG_TEST_PDF = "C:\\path\\to\\your.pdf"; python -m pytest tests/test_single_document.py -v
    bash:        RAG_TEST_PDF=/path/to/your.pdf python -m pytest tests/test_single_document.py -v

Without RAG_TEST_PDF it runs on one of the bundled sample PDFs. Nothing here assumes
what the document says: it checks load -> chunk -> embed -> store -> retrieve using the
document's own text (e.g. a chunk's opening words must retrieve that chunk).
"""
import json
import os
import re

import pytest
from helpers import PDF_DIR

from src.chunking import AVAILABLE_STRATEGIES, get_chunker
from src.chunking.agentic import AgenticChunker
from src.embeddings import get_embedder
from src.ingestion.loader import load_pdf
from src.vectorstores.chroma_store import ChromaStore
from src.vectorstores.faiss_store import FaissStore
from src.vectorstores.qdrant_store import QdrantStore

PDF = os.environ.get("RAG_TEST_PDF") or os.path.join(PDF_DIR, "usa_gdp_report.pdf")


@pytest.fixture(scope="module")
def docs():
    assert os.path.exists(PDF), f"RAG_TEST_PDF not found: {PDF}"
    pages = load_pdf(PDF)
    assert pages, f"no extractable text in {PDF} (scanned/image-only PDFs need OCR)"
    return pages


@pytest.fixture(scope="module")
def embedder():
    try:
        return get_embedder("sentence_transformers")
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {type(e).__name__}")


def _echo_llm(prompt: str) -> str:
    """Stand-in LLM for the agentic chunker: returns the page text cut into paragraphs."""
    text = prompt.split('"""')[1].strip("\n")
    return json.dumps([p for p in re.split(r"\n{2,}|(?<=\.)\n", text) if p.strip()] or [text])


def _numbers(text: str) -> set:
    return {n.rstrip(".,") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def test_document_loads_with_page_metadata(docs):
    assert all(d.text.strip() for d in docs)
    assert all(d.metadata["source"] == os.path.basename(PDF) for d in docs)
    assert [d.metadata["page"] for d in docs] == sorted(d.metadata["page"] for d in docs)


@pytest.mark.parametrize("strategy", AVAILABLE_STRATEGIES)
def test_every_strategy_chunks_this_document(strategy, docs, embedder):
    chunker = get_chunker(strategy, chunk_size=500, overlap=50, embedder=embedder, llm_fn=_echo_llm)
    chunks = chunker.chunk_documents(docs)
    assert chunks, f"{strategy} produced no chunks"
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert all(c.text.strip() for c in chunks)
    # No fact may be lost: every number in the source appears in some chunk.
    source_numbers = _numbers(" ".join(d.text for d in docs))
    chunk_numbers = _numbers(" ".join(c.text for c in chunks))
    assert not (source_numbers - chunk_numbers), sorted(source_numbers - chunk_numbers)[:5]


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "document", "semantic"])
def test_character_strategies_respect_chunk_size(strategy, docs, embedder):
    chunks = get_chunker(strategy, chunk_size=500, overlap=50, embedder=embedder).chunk_documents(docs)
    assert max(len(c.text) for c in chunks) <= 500


@pytest.fixture(params=["chroma", "faiss", "qdrant-memory"])
def store(request, tmp_path, embedder):
    d = embedder.dimension
    if request.param == "chroma":
        return ChromaStore("single_doc", d, data_dir=str(tmp_path))
    if request.param == "faiss":
        return FaissStore("single_doc", d, data_dir=str(tmp_path))
    return QdrantStore("single_doc", d, url=":memory:")


def test_retrieval_finds_chunks_from_their_own_opening_words(store, docs, embedder):
    chunks = get_chunker("recursive", 500, 50).chunk_documents(docs)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    assert store.count() == len(chunks)

    step = max(1, len(chunks) // 10)
    probes = chunks[::step][:10]
    hits_at_3 = 0
    for c in probes:
        query = " ".join(c.text.split()[:25])  # the chunk's own opening words
        top = store.search(embedder.embed_query(query), k=3)
        hits_at_3 += any(h.metadata["chunk_id"] == c.metadata["chunk_id"] for h in top)
    assert hits_at_3 / len(probes) >= 0.8, f"only {hits_at_3}/{len(probes)} chunks retrievable in top 3"


def test_reingesting_same_document_does_not_duplicate(store, docs, embedder):
    chunks = get_chunker("recursive", 500, 50).chunk_documents(docs)
    vecs = embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, vecs)
    store.add(chunks, vecs)
    assert store.count() == len(chunks)
