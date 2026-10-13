"""
Retrieval building blocks (roadmap P2): keyword search, rank fusion, reranking, filters.

- BM25Index: classic keyword ranking (Okapi BM25). Catches exact terms that embeddings
  blur together: codes ("E27"), years, figures ("3,170"), names.
- rrf_fuse: Reciprocal Rank Fusion. Merges ranked lists using ranks only
  (score = sum of 1 / (60 + rank)), so vector cosine scores and BM25 scores, which are on
  different scales, never need to be normalized or weighted.
- CrossEncoderReranker: re-scores (question, chunk) pairs jointly with a small local model.
  More precise than either first-stage ranking, but slower, so it only sees the top candidates.
- Filters: {"source": "a.pdf"} or {"source": ["a.pdf", "b.pdf"], "page": 2}; every key must
  match, a list means "any of".
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.vectorstores.base import Filters, SearchResult, matches, normalize_filters

# Keep numbers whole: "3,170", "7.2", "2023", and codes like "e27".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were what which who will with".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """In-memory Okapi BM25 over chunk texts. k1/b are the standard defaults."""

    def __init__(self, records: Iterable[Tuple[str, dict]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs: List[Tuple[str, dict, Counter, int]] = []
        self._postings: Dict[str, List[int]] = defaultdict(list)
        for text, meta in records:
            tf = Counter(tokenize(text))
            i = len(self._docs)
            self._docs.append((text, meta, tf, sum(tf.values())))
            for term in tf:
                self._postings[term].append(i)
        n = len(self._docs)
        self._avg_len = (sum(d[3] for d in self._docs) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, k: int = 5, filters: Optional[Filters] = None) -> List[SearchResult]:
        flt = normalize_filters(filters)
        scores: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                _, _, tf, length = self._docs[i]
                f = tf[term]
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self._avg_len))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out: List[SearchResult] = []
        for i, s in ranked:
            text, meta, _, _ = self._docs[i]
            if flt and not matches(meta, flt):
                continue
            out.append(SearchResult(text=text, metadata=meta, score=s, scores={"keyword": s}))
            if len(out) == k:
                break
        return out


def rrf_fuse(ranked_lists: Sequence[List[SearchResult]], k: int, c: int = 60) -> List[SearchResult]:
    """Reciprocal Rank Fusion of several ranked lists, keyed on chunk_id."""
    fused: Dict[str, SearchResult] = {}
    rrf: Dict[str, float] = defaultdict(float)
    for results in ranked_lists:
        for rank, r in enumerate(results, start=1):
            cid = r.metadata["chunk_id"]
            rrf[cid] += 1.0 / (c + rank)
            if cid in fused:
                fused[cid].scores.update(r.scores)
            else:
                fused[cid] = SearchResult(text=r.text, metadata=r.metadata, score=0.0, scores=dict(r.scores))
    for cid, s in rrf.items():
        fused[cid].score = s
        fused[cid].scores["rrf"] = s
    return sorted(fused.values(), key=lambda r: -r.score)[:k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult], k: int) -> List[SearchResult]:
        if not results:
            return []
        scores = self._model.predict([(query, r.text) for r in results], show_progress_bar=False)
        for r, s in zip(results, scores):
            r.score = float(s)
            r.scores["rerank"] = float(s)
        return sorted(results, key=lambda r: -r.score)[:k]


@lru_cache(maxsize=2)
def get_reranker(name: str, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Optional[CrossEncoderReranker]:
    if name == "none":
        return None
    if name == "cross_encoder":
        return CrossEncoderReranker(model_name)
    raise ValueError(f"Unknown reranker: {name!r} (use 'none' or 'cross_encoder')")
