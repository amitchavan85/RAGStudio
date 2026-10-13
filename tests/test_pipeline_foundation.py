import json
import os

import pytest

from helpers import HashEmbedder, PDF_DIR
from src.chunking import AVAILABLE_STRATEGIES, get_chunker
from src.ingestion.loader import load_pdf_directory
from src.vectorstores.chroma_store import ChromaStore
from src.vectorstores.faiss_store import FaissStore
from src.vectorstores.qdrant_store import QdrantStore



@pytest.fixture(scope="module")
def docs():
    return load_pdf_directory(PDF_DIR)


@pytest.mark.parametrize("strategy", AVAILABLE_STRATEGIES)
def test_every_strategy_chunks(strategy, docs):
    chunker = get_chunker(
        strategy, chunk_size=300, overlap=30, embedder=HashEmbedder(),
        llm_fn=lambda p: json.dumps(["alpha beta", "gamma delta"]),
    )
    chunks = chunker.chunk_documents(docs[:2])
    assert chunks and all(c.text.strip() for c in chunks)
    assert all("chunk_id" in c.metadata for c in chunks)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        get_chunker("nope")


@pytest.fixture(params=["chroma", "faiss", "qdrant"])
def store(request, tmp_path):
    d = HashEmbedder.dim
    if request.param == "chroma":
        return ChromaStore("test_col", d, data_dir=str(tmp_path))
    if request.param == "faiss":
        return FaissStore("test_col", d, data_dir=str(tmp_path))
    return QdrantStore("test_col", d, url=":memory:")


def test_store_roundtrip_and_upsert(store, docs):
    emb = HashEmbedder()
    chunks = get_chunker("recursive", 400, 40).chunk_documents(docs[:3])
    vecs = emb.embed_documents([c.text for c in chunks])
    store.add(chunks, vecs)
    assert store.count() == len(chunks)
    store.add(chunks, vecs)  # upsert must not duplicate
    assert store.count() == len(chunks)

    hits = store.search(emb.embed_query(chunks[0].text), k=3)
    assert hits[0].text == chunks[0].text
    assert hits[0].score >= hits[-1].score

    store.reset()
    assert store.count() == 0
