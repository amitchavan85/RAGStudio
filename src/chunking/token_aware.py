"""
TokenAwareChunker — sizes chunks by LLM TOKEN count instead of raw
characters. This matters because "chunk_size=500" in characters is
only a rough proxy (~4 characters per token for English, but it varies
a lot with punctuation, numbers, and non-English text) for what
actually determines whether a chunk fits an embedding model's input
limit or an LLM's context window. Two 500-character chunks can be 90
tokens or 180 tokens depending on content -- token-aware chunking
makes your size budget accurate regardless.

Uses tiktoken when available (downloads a small vocab/merge file from
OpenAI on first use -- needs normal internet access to
openaipublic.blob.core.windows.net). Falls back to a rough
word-count-based approximation when tiktoken or the network is
unavailable (e.g. this sandbox), so the class still runs everywhere --
but the fallback is an approximation, not the real tokenization your
LLM will see. On your own machine, `pip install tiktoken` and it will
be used automatically.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from src.chunking.base import BaseChunker
from src.chunking.recursive import DEFAULT_SEPARATORS


def _get_tiktoken_counter() -> Optional[Callable[[str], int]]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        enc.encode("connectivity check")  # forces the network call now, not on first real use
        return lambda text: len(enc.encode(text))
    except Exception:
        return None


def _fallback_word_counter(text: str) -> int:
    # Crude approximation: English averages roughly 1.3 tokens per
    # whitespace-delimited word under typical BPE tokenizers.
    words = len(text.split())
    return max(1, round(words * 1.3))


class TokenAwareChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 20,
        separators: Optional[List[str]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        """chunk_size / overlap are measured in TOKENS here, not characters."""
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.separators = separators or DEFAULT_SEPARATORS
        counter = token_counter or _get_tiktoken_counter()
        self.using_fallback_counter = counter is None
        self._count_tokens = counter or _fallback_word_counter

    def _recursive_split(self, piece: str, seps: List[str]) -> List[str]:
        if self._count_tokens(piece) <= self.chunk_size:
            return [piece]
        if not seps:
            # Last resort: chop proportionally by characters using the
            # observed chars-per-token ratio for this specific piece.
            ratio = len(piece) / max(1, self._count_tokens(piece))
            approx_chars = max(1, int(self.chunk_size * ratio))
            return [piece[i : i + approx_chars] for i in range(0, len(piece), approx_chars)]

        sep = seps[0]
        if sep == "":
            return list(piece)
        raw_parts = piece.split(sep)
        parts = [p + sep if i < len(raw_parts) - 1 else p for i, p in enumerate(raw_parts)]
        if len(parts) == 1:
            return self._recursive_split(piece, seps[1:])

        results: List[str] = []
        for part in parts:
            if self._count_tokens(part) > self.chunk_size:
                results.extend(self._recursive_split(part, seps[1:]))
            else:
                results.append(part)
        return results

    def _take_overlap(self, parts: List[str], max_tokens: int) -> Tuple[List[str], int]:
        acc: List[str] = []
        tokens = 0
        for p in reversed(parts):
            t = self._count_tokens(p)
            if tokens + t > max_tokens:
                break
            acc.insert(0, p)
            tokens += t
        return acc, tokens

    def split(self, text: str) -> List[str]:
        pieces = self._recursive_split(text, self.separators)

        packed: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0
        for piece in pieces:
            piece_tokens = self._count_tokens(piece)
            if current_parts and current_tokens + piece_tokens > self.chunk_size:
                packed.append("".join(current_parts))
                # Cap the carried overlap so carry + this piece never
                # exceeds chunk_size, mirroring the same fix in
                # RecursiveChunker.split().
                max_carry = max(0, self.chunk_size - piece_tokens)
                current_parts, current_tokens = self._take_overlap(
                    current_parts, min(self.overlap, max_carry)
                )
            current_parts.append(piece)
            current_tokens += piece_tokens
        if current_parts:
            packed.append("".join(current_parts))

        return [c.strip() for c in packed if c.strip()]
