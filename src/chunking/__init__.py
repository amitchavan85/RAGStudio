"""
Chunking strategies package.

    from src.chunking import get_chunker
    chunker = get_chunker("recursive", chunk_size=500, overlap=50)

Each strategy lives in its own file behind the shared BaseChunker
contract. Add a new one by dropping in a file and registering it below.
`semantic` needs an `embedder`; `agentic` needs an `llm_fn`.
"""
from src.chunking.agentic import AgenticChunker
from src.chunking.base import BaseChunker, Chunk
from src.chunking.document_based import DocumentBasedChunker
from src.chunking.fixed import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.sentence import SentenceChunker
from src.chunking.sliding_window import SlidingWindowChunker
from src.chunking.token_aware import TokenAwareChunker

AVAILABLE_STRATEGIES = (
    "fixed", "recursive", "sentence", "sliding_window",
    "document", "token", "semantic", "agentic",
)


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
    embedder=None,
    llm_fn=None,
) -> BaseChunker:
    """Factory: build a chunker by strategy name.

    chunk_size/overlap mean characters, except for `token` (tokens) and
    `sliding_window` (words; step = chunk_size - overlap).
    """
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence":
        return SentenceChunker(chunk_size=chunk_size)
    if strategy == "sliding_window":
        return SlidingWindowChunker(window_size_words=chunk_size, step_words=max(1, chunk_size - overlap))
    if strategy == "document":
        return DocumentBasedChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "token":
        return TokenAwareChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        return SemanticChunker(embedder=embedder, chunk_size=chunk_size, overlap=overlap)
    if strategy == "agentic":
        if llm_fn is None:
            raise ValueError("agentic chunking requires an llm_fn")
        return AgenticChunker(llm_fn=llm_fn, chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose from {AVAILABLE_STRATEGIES}")


def chunk_documents(docs, strategy: str = "recursive", chunk_size: int = 500, overlap: int = 50, **kwargs):
    """Convenience wrapper: build a chunker and apply it to documents."""
    return get_chunker(strategy, chunk_size, overlap, **kwargs).chunk_documents(docs)


__all__ = [
    "AVAILABLE_STRATEGIES", "BaseChunker", "Chunk", "get_chunker", "chunk_documents",
    "FixedSizeChunker", "RecursiveChunker", "SentenceChunker", "SlidingWindowChunker",
    "DocumentBasedChunker", "TokenAwareChunker", "SemanticChunker", "AgenticChunker",
]
