"""
DocumentBasedChunker — chunks by the document's own logical structure
(sections under headings) instead of by character count. Every
resulting chunk is prefixed with the heading of the section it came
from, so a retrieved chunk never shows up with no idea what part of
the document it's from -- a real problem with pure character-based
splitting, where "2019 21,430 +2.3%" can get retrieved with no
indication it's GDP data for the United States.

Heading detection is heuristic (short line, no trailing sentence
punctuation) unless you pass an explicit `headings` set of known
titles -- prefer the explicit list whenever you know your document
family's structure in advance (e.g. a fixed report template), since
heuristics can misfire on documents with different formatting
conventions.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from src.chunking.base import BaseChunker
from src.chunking.recursive import RecursiveChunker


class DocumentBasedChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 50,
        headings: Optional[Set[str]] = None,
    ):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.headings = headings  # None => use the heuristic detector
        self._fallback = RecursiveChunker(chunk_size, overlap)

    def _is_heading(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        if self.headings is not None:
            return line in self.headings
        if line.startswith("[TABLE]") or not re.search(r"[A-Za-z]", line):
            return False  # table markers and number-only lines (page numbers, cells) are body text
        word_count = len(line.split())
        return word_count <= 8 and not line.endswith((".", ",", ";", "%"))

    def _split_into_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        sections: List[Tuple[Optional[str], List[str]]] = []
        current_heading: Optional[str] = None
        current_body: List[str] = []

        for line in text.split("\n"):
            if self._is_heading(line):
                if current_body or current_heading is not None:
                    sections.append((current_heading, current_body))
                current_heading = line.strip()
                current_body = []
            else:
                current_body.append(line)
        if current_body or current_heading is not None:
            sections.append((current_heading, current_body))

        return [(heading, "\n".join(body).strip()) for heading, body in sections]

    def split(self, text: str) -> List[str]:
        chunks: List[str] = []
        pending: List[str] = []  # headings with no body yet (e.g. "CHAPTER 1" then "GLOBAL OUTLOOK")
        for heading, body in self._split_into_sections(text):
            if not body:
                if heading:
                    pending.append(heading)  # never drop text: join into the next section's heading
                continue
            if pending:
                heading = "\n".join(pending + ([heading] if heading else []))
                pending = []
            section_text = f"{heading}\n{body}" if heading else body
            if len(section_text) <= self.chunk_size:
                chunks.append(section_text)
            else:
                prefix = f"{heading}\n" if heading else ""
                if len(prefix) > self.chunk_size // 2:
                    # Heading too long to repeat on every piece: split the whole section plainly.
                    chunks.extend(self._fallback.split(section_text))
                else:
                    # Size pieces so heading + piece still fits chunk_size.
                    budget = self.chunk_size - len(prefix)
                    splitter = RecursiveChunker(budget, min(self.overlap, budget // 2))
                    chunks.extend(prefix + piece for piece in splitter.split(body))
        if pending:  # trailing headings with nothing after them
            chunks.extend(self._fallback.split("\n".join(pending)))
        return chunks
