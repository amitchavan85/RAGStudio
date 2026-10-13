"""Semantic chunker with the real local embedding model (no API key needed).

Skipped if the sentence-transformers model cannot be loaded (e.g. offline, first run).
"""
import os

import pytest
from helpers import PDF_DIR

from src.chunking import SemanticChunker
from src.chunking.agentic import AgenticChunker
from src.embeddings import get_embedder
from src.ingestion.loader import load_pdf_directory



@pytest.fixture(scope="module")
def embedder():
    try:
        return get_embedder("sentence_transformers")
    except Exception as e:
        pytest.skip(f"embedding model unavailable: {type(e).__name__}")


def test_cuts_between_two_clearly_different_topics(embedder):
    cats = "\n".join([
        "My cat loves to sleep on the warm sofa all afternoon.",
        "She purrs softly whenever I stroke her fur.",
        "Kittens chase string and pounce on tiny toy mice.",
        "Our tabby cat drinks milk from a small saucer.",
    ])
    finance = "\n".join([
        "The central bank raised interest rates to curb inflation.",
        "Stock markets fell sharply after the quarterly earnings report.",
        "Bond yields climbed as investors priced in tighter monetary policy.",
        "Gross domestic product growth slowed in the second quarter.",
    ])
    chunker = SemanticChunker(embedder, breakpoint_percentile=80, chunk_size=2000, min_chunk_size=0)
    parts = chunker.split(cats + "\n" + finance)
    assert len(parts) >= 2
    # No chunk may contain both topics.
    assert not any("cat" in p.lower() and "interest rates" in p.lower() for p in parts)


def test_real_pages_lose_no_numbers_and_respect_size(embedder):
    pages = load_pdf_directory(PDF_DIR)[:8]
    chunker = SemanticChunker(embedder, breakpoint_percentile=85, chunk_size=500, overlap=50)
    for page in pages:
        parts = chunker.split(page.text)
        assert parts
        assert max(len(p) for p in parts) <= 500
        assert not AgenticChunker._missing_numbers(page.text, parts), page.metadata["doc_id"]


def test_real_pages_keep_table_rows_on_their_own_lines(embedder):
    page = next(p for p in load_pdf_directory(PDF_DIR) if p.metadata["doc_id"] == "ind_gdp_report.pdf#p1")
    parts = SemanticChunker(embedder, chunk_size=500).split(page.text)
    assert any("2021 3,170 +9.7%" in p for p in parts)  # a table row survives intact
    assert sum(p.count("\n") for p in parts) > 10      # line structure not flattened
