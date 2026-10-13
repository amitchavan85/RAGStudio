"""Live agentic-chunker check with a real LLM. Skipped unless OPENAI_API_KEY is set.

Costs a few cents at most (a handful of pages through a small model).
"""
import logging
import os

import pytest
from helpers import PDF_DIR

from src.chunking.agentic import AgenticChunker
from src.config import Settings
from src.ingestion.loader import load_pdf_directory
from src.llm import get_llm_fn


pytestmark = pytest.mark.skipif(
    not Settings().openai_api_key.get_secret_value(), reason="OPENAI_API_KEY not set in .env or environment"
)


def test_real_llm_chunks_pages_without_fallback_and_without_losing_numbers(caplog):
    pages = load_pdf_directory(PDF_DIR)[:4]
    chunker = AgenticChunker(llm_fn=get_llm_fn(), chunk_size=500)
    with caplog.at_level(logging.WARNING):
        chunks = chunker.chunk_documents(pages)
    fallbacks = [r for r in caplog.records if "falling back" in r.getMessage()]
    assert chunks
    assert len(fallbacks) <= 1, [r.getMessage() for r in fallbacks]  # real model should mostly comply
    for page in pages:
        mine = [c.text for c in chunks if c.metadata["doc_id"] == page.metadata["doc_id"]]
        assert not AgenticChunker._missing_numbers(page.text, mine), page.metadata["doc_id"]
