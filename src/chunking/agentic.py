"""
AgenticChunker — uses an LLM to decide chunk boundaries directly,
instead of any fixed character/token/heading rule. The LLM reads the
text and returns a list of self-contained, semantically coherent
passages that it judges make sense as standalone retrieval units.

This is the most expensive and slowest strategy here by a wide margin
(one LLM call per document/page, vs. zero for every rule-based
strategy in this package) and its quality is only as good as the
model and prompt. It's worth reaching for when a document has no
exploitable structure at all for the cheaper strategies to key off of
(e.g. dense narrative or legal text with no headings, inconsistent
paragraphing, and sentences that individually don't capture complete
thoughts).

This class takes no opinion on which LLM provider you use -- pass any
`llm_fn: Callable[[str], str]` that takes a prompt and returns the raw
model response (a one-line wrapper around whatever provider you build
in Step 7). If the call fails or returns something unparseable, this
falls back to RecursiveChunker rather than losing the document.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, List

from src.chunking.base import BaseChunker
from src.chunking.recursive import RecursiveChunker

logger = logging.getLogger(__name__)

AGENTIC_CHUNK_PROMPT = """You are splitting a document into retrieval-ready chunks for a RAG system.

Split the text below into a JSON array of strings. Rules:
- COPY the text verbatim. Do NOT summarise, paraphrase, reorder or omit anything:
  every sentence, number, year and table row of the input must appear in the output.
- Only decide WHERE to cut. Cut between topics so each chunk covers one coherent idea
  (keep a table together with its heading line, and never split a table row).
- You may prefix a chunk with a short heading from the text for context, but never
  invent new facts.
- Aim for chunks of about {target_size} characters: group related lines together and
  never output tiny chunks (under {min_size} characters) such as a single table row.

Return ONLY a valid JSON array of strings and nothing else -- no markdown
fences, no commentary.

TEXT:
\"\"\"
{text}
\"\"\"
"""


class AgenticChunker(BaseChunker):
    def __init__(self, llm_fn: Callable[[str], str], chunk_size: int = 500, overlap: int = 50):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.llm_fn = llm_fn
        self._fallback = RecursiveChunker(chunk_size, overlap)

    def split(self, text: str) -> List[str]:
        prompt = AGENTIC_CHUNK_PROMPT.format(target_size=self.chunk_size, min_size=self.chunk_size // 3, text=text)
        try:
            response = self.llm_fn(prompt)
            chunks = self._parse_response(response)
            if not chunks:
                raise ValueError("LLM returned an empty chunk list")
            missing = self._missing_numbers(text, chunks)
            if missing:
                raise ValueError(f"LLM output lost or changed numbers from the source: {sorted(missing)[:5]}")
            return self._merge_small(chunks)
        except Exception as e:
            logger.warning("agentic chunking failed (%s); falling back to RecursiveChunker", e)
            return self._fallback.split(text)

    def _merge_small(self, chunks: List[str]) -> List[str]:
        """Safety net: glue chunks shorter than chunk_size/3 onto their neighbour
        (never exceeding chunk_size), since LLMs sometimes over-split."""
        min_size = self.chunk_size // 3
        merged: List[str] = []
        for c in chunks:
            if merged and (len(merged[-1]) < min_size or len(c) < min_size) and len(merged[-1]) + len(c) + 1 <= self.chunk_size:
                merged[-1] = merged[-1] + "\n" + c
            else:
                merged.append(c)
        return merged

    @staticmethod
    def _parse_response(response: str) -> List[str]:
        # Models often add ```json fences or a sentence of commentary despite
        # being told not to -- strip fences, then take the outermost [...] .
        cleaned = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x.strip()]
        raise ValueError("LLM response was not a JSON array of strings")

    @staticmethod
    def _missing_numbers(source: str, chunks: List[str]) -> set:
        """Numbers (GDP figures, years, %) present in the source but absent from the
        chunks. A non-empty result means the LLM dropped or altered facts."""
        number = re.compile(r"\d[\d,]*\.?\d*")
        joined = " ".join(chunks)
        return {n.rstrip(".,") for n in number.findall(source)} - {n.rstrip(".,") for n in number.findall(joined)}
