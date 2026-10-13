"""
Step 3: Chunking — shared base.

Why chunk at all?
  - Embedding models produce a single vector per input -- if you embed
    a whole 5-page document as one vector, that vector is a blurry
    average of everything in it and retrieval precision collapses.
  - LLM context windows and cost also push you toward retrieving only
    the few hundred words that are actually relevant to a question,
    not entire documents.

`BaseChunker` is the contract every strategy (FixedSizeChunker,
RecursiveChunker, ...) implements: same constructor shape
(chunk_size, overlap), same `.split(text) -> List[str]` method. That's
what lets `chunk_documents()` -- and everything downstream of it --
treat every strategy interchangeably; swapping one for another is a
one-line change, never a rewrite.

Overlap (a sliding window of shared text between consecutive chunks)
exists so a fact sitting right at a chunk boundary isn't lost to
either chunk.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from src.ingestion.loader import Document
from src.logging_config import log_step

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.metadata.get('chunk_id')!r}, text={preview!r}...)"


class BaseChunker(ABC):
    """Common interface every chunking strategy implements."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split a single string of text into a list of chunk strings."""

    def chunk_documents(self, docs: List[Document]) -> List[Chunk]:
        """Apply this chunker to a list of page-level Documents, carrying
        source/page metadata forward and adding a stable chunk_id.
        """
        all_chunks: List[Chunk] = []
        with log_step(logger, "chunk", strategy=type(self).__name__, chunk_size=self.chunk_size,
                      overlap=self.overlap, docs=len(docs)) as out:
            for doc in docs:
                for i, piece in enumerate(self.split(doc.text)):
                    meta = dict(doc.metadata)
                    meta["chunk_index"] = i
                    meta["chunk_id"] = f"{meta.get('doc_id', meta.get('source'))}::chunk{i}"
                    all_chunks.append(Chunk(text=piece, metadata=meta))
            lengths = [len(c.text) for c in all_chunks]
            out["chunks"] = len(all_chunks)
            if lengths:
                out["len_min_avg_max"] = f"{min(lengths)}/{sum(lengths) // len(lengths)}/{max(lengths)}"
        return all_chunks
