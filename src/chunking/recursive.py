"""
RecursiveChunker — splits on "natural" boundaries first (paragraph,
then line, then sentence, then word), only falling back to a hard
character cut if a piece is still too big. This is the strategy most
production RAG systems use by default (it's what LangChain's
RecursiveCharacterTextSplitter does).
"""
from __future__ import annotations

from typing import List, Optional

from src.chunking.base import BaseChunker
from src.chunking.fixed import FixedSizeChunker

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, overlap)
        self.separators = separators or DEFAULT_SEPARATORS
        # Reused as the fallback for any piece that still has no natural
        # boundary to split on (e.g. one giant run-on word/table blob).
        self._fallback = FixedSizeChunker(chunk_size, overlap)

    @staticmethod
    def _split_on_separator(text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        sep = separators[0]
        if sep == "":
            return list(text)  # last resort: character-level
        parts = text.split(sep)
        # Re-attach the separator (except after the last part) so joined
        # length accounting stays correct and no text is silently dropped.
        return [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]

    def _recursive_split(self, piece: str, seps: List[str]) -> List[str]:
        if len(piece) <= self.chunk_size:
            return [piece]
        if not seps:
            return self._fallback.split(piece)

        parts = self._split_on_separator(piece, seps)
        # If splitting on this separator didn't actually break anything
        # up (e.g. no newlines in a single long paragraph), try the
        # next, finer-grained separator instead.
        if len(parts) == 1:
            return self._recursive_split(piece, seps[1:])

        results: List[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                results.extend(self._recursive_split(part, seps[1:]))
            else:
                results.append(part)
        return results

    def split(self, text: str) -> List[str]:
        raw_pieces = self._recursive_split(text, self.separators)

        # Greedily pack the small pieces back together up to chunk_size,
        # with `overlap` characters carried over between consecutive chunks.
        packed: List[str] = []
        current = ""
        for piece in raw_pieces:
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current:
                    packed.append(current)
                # Cap the carried overlap so carry + piece never exceeds chunk_size.
                carry_len = min(self.overlap, self.chunk_size - len(piece))
                carry = current[-carry_len:] if carry_len > 0 and current else ""
                current = carry + piece
        if current:
            packed.append(current)

        return [c.strip() for c in packed if c.strip()]
