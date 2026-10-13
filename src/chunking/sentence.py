"""
SentenceChunker — groups whole sentences together up to a character
budget, and NEVER splits a sentence in half (unlike FixedSizeChunker,
and unlike RecursiveChunker in the rare case a single sentence is
itself larger than chunk_size and no finer separator is available).

Sentence splitting here is a lightweight regex heuristic, not a full
NLP sentence tokenizer (spaCy/nltk's punkt) -- good enough for clean
prose like this tutorial's PDFs, but it will mis-split on things like
"Dr. Smith" or "3.14". Swap `split_sentences` for spaCy's sentencizer
or nltk.sent_tokenize in a production system handling messier text.
"""
from __future__ import annotations

import re
from typing import List, Optional

from src.chunking.base import BaseChunker

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class SentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, sentence_overlap: int = 1):
        """
        chunk_size: soft character budget per chunk (a chunk may exceed
            this if a single sentence alone is longer than chunk_size --
            we choose not to mutilate sentences to enforce a hard cap).
        sentence_overlap: number of trailing sentences carried over from
            the end of one chunk to the start of the next (this replaces
            character-based overlap, which doesn't make sense once
            chunk boundaries are pinned to sentence boundaries).
        """
        super().__init__(chunk_size=chunk_size, overlap=0)
        self.sentence_overlap = sentence_overlap

    def split(self, text: str) -> List[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for sent in sentences:
            if current and current_len + len(sent) + 1 > self.chunk_size:
                chunks.append(" ".join(current))
                current = current[-self.sentence_overlap :] if self.sentence_overlap else []
                current_len = sum(len(s) + 1 for s in current)
            current.append(sent)
            current_len += len(sent) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks
