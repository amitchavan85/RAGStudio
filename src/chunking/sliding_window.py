"""
SlidingWindowChunker — a fixed-size window that slides over WORDS
(not raw characters) with a configurable step. This is the classic
"sliding window" formulation you'll see in NLP literature: window_size
and step are independent, so overlap = window_size - step can be
tuned as a fraction of the window rather than a raw character count.

How this differs from FixedSizeChunker: FixedSizeChunker slices by
character count, so it can (and does) cut words in half. Operating on
whitespace-delimited words instead means every chunk boundary falls
between words -- a small but real quality improvement for roughly the
same cost, and a natural stepping stone toward SentenceChunker.
"""
from __future__ import annotations

from typing import List

from src.chunking.base import BaseChunker


class SlidingWindowChunker(BaseChunker):
    def __init__(self, window_size_words: int = 100, step_words: int | None = None):
        """
        window_size_words: how many words each chunk contains.
        step_words: how many words to advance between windows. Defaults
            to window_size_words // 2 (50% overlap) if not given.
        """
        step_words = step_words or max(1, window_size_words // 2)
        if step_words > window_size_words:
            raise ValueError("step_words cannot exceed window_size_words (that would skip content)")
        # We don't use BaseChunker's char-based chunk_size/overlap here,
        # but we still call super().__init__ with placeholder values so
        # every chunker in this package exposes the same attributes.
        super().__init__(chunk_size=window_size_words, overlap=window_size_words - step_words)
        self.window_size_words = window_size_words
        self.step_words = step_words

    def split(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks: List[str] = []
        start = 0
        n = len(words)
        while start < n:
            end = min(start + self.window_size_words, n)
            chunks.append(" ".join(words[start:end]))
            if end == n:
                break
            start += self.step_words
        return chunks
