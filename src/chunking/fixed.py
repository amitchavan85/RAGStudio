"""FixedSizeChunker — naive fixed-width chunking with overlap."""
from __future__ import annotations

from typing import List

from src.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Splits on a raw character count with overlap. O(n), no awareness
    of sentence/paragraph boundaries -- kept mainly so you can compare
    its output against RecursiveChunker and see why it's usually worse.
    """

    def split(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - self.overlap
        return chunks
