"""
RAG pipeline: the one entry point that ties loader, chunker, embedder and vector store together.

    from src.pipeline import RAGPipeline

    rag = RAGPipeline()                                  # defaults from .env
    rag = RAGPipeline(chunking_strategy="semantic", vector_store="qdrant")  # per-call overrides
    rag.ingest("report.pdf")                             # file, list of files, folder, or None (= PDF_DIR)
    hits = rag.query("What was GDP growth in 2023?", k=5)
    rag.list_documents()                                 # {"report.pdf": 42}
    rag.delete_document("report.pdf")

Rules it enforces:
  - One collection per embedding model: the collection name includes the model and its dimension,
    so vectors from different models can never be mixed (they are not comparable).
  - A document is identified by its file name (metadata["source"]). Re-ingesting a file replaces
    ALL of its old chunks, so changing chunker or chunk size leaves no stale chunks behind.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional

from src.chunking import AVAILABLE_STRATEGIES, get_chunker
from src.config import get_settings
from src.embeddings import EmbeddingError, EmbeddingProvider, get_embedder
from src.ingestion.loader import load_pdf
from src.logging_config import log_step
from src.generation import NOT_FOUND, SYSTEM_PROMPT, Answer, build_prompt, extract_citations, normalize_citations
from src.retrieval import BM25Index, get_reranker, rrf_fuse
from src.vectorstores import AVAILABLE_STORES, SearchResult, VectorStore, get_vector_store
from src.vectorstores.base import Filters

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDERS = ("sentence_transformers", "openai")
RETRIEVAL_MODES = ("hybrid", "vector", "keyword")
RERANKERS = ("none", "cross_encoder")

# Keyword (BM25) indexes, built from the store texts and shared by all pipelines in this process.
# Key: (store, location, collection) -> (chunk count when built, index).
_BM25_CACHE: Dict[tuple, tuple] = {}
_BM25_LOCK = threading.Lock()

# One store object per collection, shared by every pipeline in this process. Local stores
# (FAISS) keep data in memory, so separate instances would go stale and could overwrite
# each other's writes; server stores reuse one connection instead of opening one per request.
_STORES: Dict[tuple, VectorStore] = {}
_STORES_LOCK = threading.Lock()


@dataclass(frozen=True)
class PipelineConfig:
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: Optional[str]  # None = the provider's default model
    vector_store: str
    collection_name: str  # base name; the embedding model is appended (see collection_name())
    retrieval_mode: str = "hybrid"
    reranker: str = "none"
    llm_model: Optional[str] = None  # None = LLM_MODEL from .env

    def __post_init__(self):
        if self.chunking_strategy not in AVAILABLE_STRATEGIES:
            raise ValueError(f"chunking_strategy must be one of {AVAILABLE_STRATEGIES}, got {self.chunking_strategy!r}")
        if self.vector_store not in AVAILABLE_STORES:
            raise ValueError(f"vector_store must be one of {AVAILABLE_STORES}, got {self.vector_store!r}")
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"embedding_provider must be one of {EMBEDDING_PROVIDERS}, got {self.embedding_provider!r}")
        if self.llm_model is not None:
            from src.llm import resolve

            resolve(self.llm_model)  # raises ValueError for unknown models
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {self.retrieval_mode!r}")
        if self.reranker not in RERANKERS:
            raise ValueError(f"reranker must be one of {RERANKERS}, got {self.reranker!r}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.collection_name):
            raise ValueError("collection_name must start with a letter and contain only letters, digits and _")

    @classmethod
    def from_settings(cls, **overrides) -> "PipelineConfig":
        """Defaults from .env; any field can be overridden. None means "use the default"."""
        unknown = set(overrides) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown pipeline option(s): {sorted(unknown)}")
        s = get_settings()
        values = dict(
            chunking_strategy=s.chunking_strategy,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            embedding_provider=s.embedding_provider,
            embedding_model=None,
            vector_store=s.vector_store,
            collection_name=s.collection_name,
            retrieval_mode=s.retrieval_mode,
            reranker=s.reranker,
            llm_model=s.llm_model,
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def with_overrides(self, **overrides) -> "PipelineConfig":
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown pipeline option(s): {sorted(unknown)}")
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


@dataclass
class IngestReport:
    collection: str
    files: List[dict] = field(default_factory=list)    # {source, pages, chunks, replaced}
    skipped: List[dict] = field(default_factory=list)  # {source, reason}

    @property
    def total_chunks(self) -> int:
        return sum(f["chunks"] for f in self.files)


def reset_shared_state() -> None:
    """Forget shared stores and keyword indexes (tests, or after deleting data directories)."""
    with _STORES_LOCK:
        _STORES.clear()
    with _BM25_LOCK:
        _BM25_CACHE.clear()


def collection_name(base: str, embedder: EmbeddingProvider) -> str:
    """e.g. rag_docs + all-MiniLM-L6-v2 (384) -> rag_docs_all_minilm_l6_v2_384.

    Kept <= 50 chars and SQL-safe so every backend accepts it (pgvector also derives
    index names from it, and Postgres identifiers are limited to 63 characters).
    """
    model = embedder.model_name.rsplit("/", 1)[-1]
    name = re.sub(r"[^a-z0-9]+", "_", f"{base}_{model}_{embedder.dimension}".lower()).strip("_")
    if len(name) > 50:
        name = f"{name[:41]}_{hashlib.sha1(name.encode()).hexdigest()[:8]}"
    return name


@lru_cache(maxsize=4)
def _cached_embedder(provider: str, model: Optional[str]) -> EmbeddingProvider:
    """Embedding models are slow to load; share one instance per (provider, model)."""
    return get_embedder(provider, model)


def _pdf_paths(paths: str | Iterable[str] | None) -> List[str]:
    if paths is None:
        paths = [str(get_settings().pdf_path)]
    elif isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out: List[str] = []
    for p in map(str, paths):
        if os.path.isdir(p):
            out.extend(os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(".pdf"))
        else:
            out.append(p)
    names = [os.path.basename(p) for p in out]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"documents are identified by file name; these appear more than once: {dupes}")
    return out


class RAGPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        embedder: Optional[EmbeddingProvider] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        llm=None,
        **overrides,
    ):
        """config: full config (defaults from .env if omitted); overrides: change single fields.
        embedder / llm_fn / llm: inject instances (mainly for tests); otherwise built from the config."""
        # Both paths validate: from_settings rejects unknown names, replace() re-runs __post_init__.
        self.config = config.with_overrides(**overrides) if config else PipelineConfig.from_settings(**overrides)
        self.embedder = embedder or _cached_embedder(self.config.embedding_provider, self.config.embedding_model)
        self._llm_fn = llm_fn
        self._llm = llm
        self._store: Optional[VectorStore] = None

    @property
    def collection(self) -> str:
        return collection_name(self.config.collection_name, self.embedder)

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            key = self._bm25_key()
            with _STORES_LOCK:
                if key not in _STORES:
                    _STORES[key] = get_vector_store(self.config.vector_store, self.embedder.dimension,
                                                    collection=self.collection)
                self._store = _STORES[key]
        return self._store

    def _bm25_key(self) -> tuple:
        s = get_settings()
        location = {"qdrant": s.qdrant_url, "pgvector": s.pg_dsn}.get(self.config.vector_store, s.data_dir)
        return self.config.vector_store, location, self.collection

    def _keyword_index(self) -> BM25Index:
        """BM25 over every chunk in the collection. Rebuilt when this process ingests/deletes,
        or when the store's chunk count changes (another process wrote to it)."""
        key, n = self._bm25_key(), self.store.count()
        with _BM25_LOCK:
            cached = _BM25_CACHE.get(key)
            if cached and cached[0] == n:
                return cached[1]
        with log_step(logger, "bm25.build", collection=self.collection, chunks=n):
            index = BM25Index(self.store.iter_records())
        with _BM25_LOCK:
            _BM25_CACHE[key] = (n, index)
        return index

    def _invalidate_keyword_index(self) -> None:
        with _BM25_LOCK:
            _BM25_CACHE.pop(self._bm25_key(), None)

    @contextmanager
    def _embedding_errors(self):
        """Report embedding failures as EmbeddingError with the cause and a fix, so they are not
        mistaken for LLM failures (e.g. a revoked OpenAI key while answering with a local model)."""
        try:
            yield
        except EmbeddingError:
            raise
        except Exception as e:
            provider = self.config.embedding_provider
            text = str(e)
            if type(e).__name__ == "AuthenticationError" or "401" in text or "OPENAI_API_KEY" in text:
                cause = "the API key is invalid, revoked or missing (OPENAI_API_KEY)"
            else:
                cause = f"{type(e).__name__}: {text[:200]}"
            hint = (" Fix the key, or set EMBEDDING_PROVIDER=sentence_transformers for free local embeddings"
                    " (then re-upload your documents: they go to a new collection).") if provider == "openai" else ""
            raise EmbeddingError(f"Embedding with {provider} failed: {cause}.{hint}") from e

    def _chunker(self):
        llm_fn = self._llm_fn
        if self.config.chunking_strategy == "agentic" and llm_fn is None:
            llm_fn = self._llm_fn = self._get_llm()
        c = self.config
        return get_chunker(c.chunking_strategy, c.chunk_size, c.chunk_overlap, embedder=self.embedder, llm_fn=llm_fn)

    def ingest(self, paths: str | Iterable[str] | None = None) -> IngestReport:
        """Load, chunk, embed and store PDFs. Each file replaces its previous chunks."""
        files = _pdf_paths(paths)
        report = IngestReport(collection=self.collection)
        chunker = self._chunker()
        with log_step(logger, "ingest", files=len(files), store=self.config.vector_store,
                      collection=self.collection, strategy=self.config.chunking_strategy) as out, \
                self.store.bulk():  # pgvector: first load into an empty table builds its index once, at the end
            for path in files:
                source = os.path.basename(path)
                try:
                    docs = load_pdf(path)
                except Exception as e:
                    logger.warning("skipping unreadable PDF %s (%s: %s)", source, type(e).__name__, e)
                    report.skipped.append({"source": source, "reason": f"{type(e).__name__}: {e}"})
                    continue
                if not docs:
                    report.skipped.append({"source": source, "reason": "no extractable text (scanned PDF needs OCR)"})
                    continue
                with self._embedding_errors():  # semantic chunking embeds too
                    chunks = chunker.chunk_documents(docs)
                    vectors = self.embedder.embed_documents([c.text for c in chunks])
                # Replace only after chunking/embedding succeeded, so a failure never loses the old version.
                replaced = self.store.delete_source(source)
                self.store.add(chunks, vectors)
                report.files.append({"source": source, "pages": len(docs), "chunks": len(chunks), "replaced": replaced})
            out.update(ingested=len(report.files), skipped=len(report.skipped), chunks=report.total_chunks)
        self._invalidate_keyword_index()
        return report

    def query(
        self,
        question: str,
        k: int = 5,
        filters: Optional[Filters] = None,
        mode: Optional[str] = None,
        reranker: Optional[str] = None,
    ) -> List[SearchResult]:
        """Retrieve the k most relevant chunks (answer generation is roadmap P3).

        mode: "hybrid" (vector + BM25 fused with RRF), "vector" or "keyword"; default from config.
        filters: e.g. {"source": "report.pdf"} or {"source": [...], "page": 2}.
        reranker: "cross_encoder" re-scores the top candidates with a local model; "none" skips it.
        Each result's `scores` shows the per-stage scores (vector, keyword, rrf, rerank).
        """
        if not question.strip():
            raise ValueError("question must not be empty")
        mode = mode or self.config.retrieval_mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
        reranker_name = reranker or self.config.reranker
        if reranker_name not in RERANKERS:
            raise ValueError(f"reranker must be one of {RERANKERS}, got {reranker_name!r}")
        rr = get_reranker(reranker_name, get_settings().reranker_model)
        # Fusion and reranking need a wider candidate pool than the final k.
        pool = max(k * 4, 20) if (mode == "hybrid" or rr) else k

        with log_step(logger, "query", mode=mode, k=k, reranker=reranker_name,
                      filters=sorted(filters) if filters else None) as out:
            ranked: List[List[SearchResult]] = []
            if mode in ("vector", "hybrid"):
                with self._embedding_errors():
                    vector = self.embedder.embed_query(question)
                ranked.append(self.store.search(vector, k=pool, filters=filters))
            if mode in ("keyword", "hybrid"):
                ranked.append(self._keyword_index().search(question, k=pool, filters=filters))
            results = rrf_fuse(ranked, k=pool) if mode == "hybrid" else ranked[0]
            if rr:
                results = rr.rerank(question, results, k=k)
            results = results[:k]
            out["hits"] = len(results)
        return results

    def _get_llm(self, model: Optional[str] = None):
        if model is None and self._llm is not None:
            return self._llm
        from src.llm import get_llm

        return get_llm(model or self.config.llm_model)

    def answer(
        self,
        question: str,
        k: int = 8,
        filters: Optional[Filters] = None,
        mode: Optional[str] = None,
        reranker: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> Answer:
        """Retrieve the k best chunks and have the LLM answer from them, with [n] citations.

        k defaults to 8 (query() uses 5): comparison questions need facts from several
        documents, and in testing the second document's chunk often ranked 6th.

        model: any catalog model or "<provider>:<model>" for this call; default config.llm_model.
        If nothing relevant is retrieved, returns the not-found answer without calling the LLM.
        """
        if model is not None:
            from src.llm import resolve

            resolve(model)  # fail fast on unknown models, before retrieval
        sources = self.query(question, k=k, filters=filters, mode=mode, reranker=reranker)
        if not sources:
            return Answer(question=question, text=NOT_FOUND, sources=[])
        llm = self._get_llm(model)
        with log_step(logger, "answer", sources=len(sources)) as out:
            resp = llm.generate(build_prompt(question, sources), system=SYSTEM_PROMPT, max_tokens=max_tokens)
            result = Answer(
                question=question, text=normalize_citations(resp.text.strip()), sources=sources,
                citations=extract_citations(normalize_citations(resp.text), sources), model=resp.model, provider=resp.provider,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            )
            out.update(model=resp.model, citations=len(result.citations), found=result.found)
        return result

    def list_documents(self) -> Dict[str, int]:
        return self.store.list_sources()

    def delete_document(self, source: str) -> int:
        removed = self.store.delete_source(source)
        self._invalidate_keyword_index()
        return removed
