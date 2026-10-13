"""Request/response models of the HTTP API (v1). Every option block is optional; omitted
fields fall back to the server defaults (.env)."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

FilterValue = Union[str, int, float, bool, List[Union[str, int, float, bool]]]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typos in option names are errors, not silently ignored


# ---------- option blocks ----------

class IndexSpec(_Strict):
    """Free-form index (only when the server sets ALLOW_CUSTOM_INDEX=true)."""
    vector_store: Optional[Literal["chroma", "faiss", "qdrant", "pgvector"]] = None
    embedding_provider: Optional[Literal["sentence_transformers", "openai"]] = None
    embedding_model: Optional[str] = None
    collection: Optional[str] = None


IndexRef = Union[str, IndexSpec]  # a named index (default "default") or a custom spec


class ChunkingOptions(_Strict):
    strategy: Optional[str] = None
    chunk_size: Optional[int] = Field(None, ge=50, le=8000)
    overlap: Optional[int] = Field(None, ge=0, le=4000)


class RetrievalOptions(_Strict):
    mode: Optional[Literal["hybrid", "vector", "keyword"]] = None
    k: Optional[int] = Field(None, ge=1)
    reranker: Optional[Literal["none", "cross_encoder"]] = None
    filters: Optional[Dict[str, FilterValue]] = None


class LLMOptions(_Strict):
    model: Optional[str] = None
    max_tokens: int = Field(2048, ge=16, le=32000)


# ---------- requests ----------

class SearchRequest(_Strict):
    question: str = Field(..., min_length=1, max_length=4000)
    index: IndexRef = "default"
    retrieval: RetrievalOptions = RetrievalOptions()


class AnswerRequest(SearchRequest):
    llm: LLMOptions = LLMOptions()
    include_sources: bool = False
    stream: Literal[False] = False  # reserved: token streaming (SSE) comes later


class IngestOptions(_Strict):
    """JSON sent in the multipart form field `options` of POST /v1/documents."""
    index: IndexRef = "default"
    chunking: ChunkingOptions = ChunkingOptions()


# ---------- responses ----------

class ChunkOut(BaseModel):
    text: str
    source: Optional[str]
    page: Optional[int]
    chunk_id: Optional[str]
    score: float
    scores: Dict[str, float]


class SearchResponse(BaseModel):
    results: List[ChunkOut]
    collection: str
    request_id: str


class Citation(BaseModel):
    n: int
    source: Optional[str]
    page: Optional[int]
    chunk_id: Optional[str]
    snippet: str


class Usage(BaseModel):
    provider: Optional[str]
    model: Optional[str]
    input_tokens: int
    output_tokens: int


class AnswerResponse(BaseModel):
    answer: str
    found: bool
    citations: List[Citation]
    sources: Optional[List[ChunkOut]] = None
    usage: Usage
    collection: str
    request_id: str


class FileResult(BaseModel):
    source: str
    pages: int
    chunks: int
    replaced: int


class SkippedFile(BaseModel):
    source: str
    reason: str


class IngestResult(BaseModel):
    files: List[FileResult]
    skipped: List[SkippedFile]
    total_chunks: int
    collection: str


class Job(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    finished_at: Optional[str] = None
    files: List[str]
    result: Optional[IngestResult] = None
    error: Optional[str] = None


class DocumentOut(BaseModel):
    source: str
    chunks: int


class DocumentList(BaseModel):
    documents: List[DocumentOut]
    collection: str


class DeleteResult(BaseModel):
    source: str
    deleted_chunks: int
    collection: str


class IndexOut(BaseModel):
    name: str
    vector_store: str
    embedding_provider: str
    embedding_model: Optional[str]
    collection: Optional[str] = None
    documents: Optional[int] = None
    chunks: Optional[int] = None
    error: Optional[str] = None


class ModelOut(BaseModel):
    name: str
    provider: str
    tier: str
    input_per_m: float
    output_per_m: float
    available: bool
    reason: Optional[str] = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str
