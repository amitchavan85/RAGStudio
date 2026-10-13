"""
SemanticChunker — splits text where MEANING shifts, not where a
character count runs out. This is the technique popularized by
Greg Kamradt / LlamaIndex's "semantic chunking":

  1. Split the text into small units (lines and sentences).
  2. Embed each unit (combined with a small window of neighboring
     units, so short units get a more stable embedding).
  3. Compute the cosine distance between every pair of *consecutive*
     unit embeddings.
  4. Wherever that distance spikes above a percentile threshold, it
     means the topic just changed -- cut the chunk there.
  5. Glue tiny chunks onto a neighbour and re-split oversized ones.

Units are lines AND sentences because extracted PDF text is mostly
line-structured (headings, table rows) with few full stops; splitting
on sentences alone finds almost no boundaries. Original whitespace is
preserved, so table rows stay on their own lines inside the chunk.

The result: chunks that track topical/semantic boundaries instead of
an arbitrary character count, at the cost of one embedding per unit
(more expensive than every other strategy here except AgenticChunker)
and a "breakpoint_percentile" hyperparameter that needs tuning per corpus.

Needs an EmbeddingProvider -- pass any of src/embeddings' providers.
Sparse providers (TfidfEmbeddings) are fit fresh per-document here
(TF-IDF is corpus-relative), which is exactly what we want at this granularity.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from src.chunking.base import BaseChunker
from src.chunking.recursive import RecursiveChunker
from src.embeddings.base import EmbeddingProvider

# A unit ends at a line break, or after sentence punctuation followed by a new sentence.
_BOUNDARY_RE = re.compile(r"\n+|(?<=[.!?])[ \t]+(?=[A-Z0-9\[])")


def split_units(text: str) -> List[str]:
    """Split into lines/sentences, keeping the separators so "".join(units) == text."""
    units, start = [], 0
    for m in _BOUNDARY_RE.finditer(text):
        units.append(text[start : m.end()])
        start = m.end()
    if start < len(text):
        units.append(text[start:])
    return [u for u in units if u.strip()]


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedder: EmbeddingProvider,
        breakpoint_percentile: float = 85.0,
        buffer_size: int = 1,
        chunk_size: int = 1500,
        overlap: int = 50,
        min_chunk_size: int | None = None,
    ):
        """
        breakpoint_percentile: how large a semantic "jump" between
            consecutive units has to be (relative to all jumps seen
            in this document) before we treat it as a topic boundary.
            Lower = more, smaller chunks; higher = fewer, larger chunks.
        buffer_size: number of neighboring units on each side
            included when embedding a given unit's "position", to
            smooth out noise from very short units (table rows).
        chunk_size: hard ceiling -- any semantic chunk that still comes
            out larger than this gets split further by RecursiveChunker
            as a safety net (semantic boundaries can be sparse in dense
            single-topic sections).
        min_chunk_size: chunks shorter than this are merged into a
            neighbour when the result still fits chunk_size.
            Defaults to chunk_size // 4.
        """
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_chunk_size = chunk_size // 4 if min_chunk_size is None else min_chunk_size
        self._fallback = RecursiveChunker(chunk_size, overlap)

    def split(self, text: str) -> List[str]:
        units = split_units(text)
        if len(units) < 2:
            return [text.strip()] if text.strip() else []

        # Combine each unit with its neighbors for a more stable embedding.
        windowed = []
        for i in range(len(units)):
            lo = max(0, i - self.buffer_size)
            hi = min(len(units), i + self.buffer_size + 1)
            windowed.append(" ".join(u.strip() for u in units[lo:hi]))

        self.embedder.fit(windowed)
        vectors = np.array(self.embedder.embed_documents(windowed))

        distances = []
        for i in range(len(vectors) - 1):
            a, b = vectors[i], vectors[i + 1]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            sim = float(np.dot(a, b) / denom) if denom > 0 else 1.0
            distances.append(1.0 - sim)

        if not distances or max(distances) == 0:
            groups = [units]
        else:
            threshold = float(np.percentile(distances, self.breakpoint_percentile))
            groups, current = [], [units[0]]
            for i, dist in enumerate(distances):
                if dist > threshold:
                    groups.append(current)
                    current = []
                current.append(units[i + 1])
            if current:
                groups.append(current)

        merged = self._merge_small(["".join(g).strip() for g in groups])

        # Safety net: enforce the hard character ceiling.
        final: List[str] = []
        for chunk in merged:
            if len(chunk) > self.chunk_size:
                final.extend(self._fallback.split(chunk))
            else:
                final.append(chunk)
        return final

    def _merge_small(self, chunks: List[str]) -> List[str]:
        merged: List[str] = []
        for c in chunks:
            if merged and (len(merged[-1]) < self.min_chunk_size or len(c) < self.min_chunk_size) \
                    and len(merged[-1]) + len(c) + 1 <= self.chunk_size:
                merged[-1] = merged[-1] + "\n" + c
            else:
                merged.append(c)
        return merged
