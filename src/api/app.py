"""
HTTP API (roadmap P4) over RAGPipeline.

    python -m src.api            # serves on API_HOST:API_PORT; docs at /docs

Pipeline calls are blocking, so endpoints are plain `def` (FastAPI runs them in a thread
pool). All pipelines in the process share one store per collection (see src/pipeline.py).
FAISS and Chroma are single-process stores: run one worker, or use Qdrant / pgvector.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, Query, Request, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import ValidationError

from src.api import schemas as S
from src.api.core import (ApiError, JobStore, RateLimiter, identify_client, load_api_keys, named_indexes,
                          resolve_index, validate_index_specs)
from src.chunking import AVAILABLE_STRATEGIES
from src.config import get_settings
from src.logging_config import _request_id, new_request_id, setup_logging
from src.pipeline import RAGPipeline, RERANKERS, RETRIEVAL_MODES
from src.vectorstores import AVAILABLE_STORES

logger = logging.getLogger(__name__)


def _chunk_out(r) -> S.ChunkOut:
    return S.ChunkOut(text=r.text, source=r.metadata.get("source"), page=r.metadata.get("page"),
                      chunk_id=r.metadata.get("chunk_id"), score=r.score, scores=r.scores)


def _renumber_citations(text: str, citations: List[dict]) -> tuple[str, List[dict]]:
    """Answers cite positions in the retrieved list ([5], [6]); renumber the cited ones 1..n."""
    mapping = {c["n"]: i for i, c in enumerate(citations, 1)}
    text = re.sub(r"\[(\d+)\]", lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else m.group(0), text)
    return text, [{**c, "n": mapping[c["n"]]} for c in citations]


def create_app(embedder=None, llm=None) -> FastAPI:
    """embedder / llm: optional overrides (tests inject fakes; production builds them from config)."""
    jobs = JobStore()
    limiter = RateLimiter()
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                                  description="Client key from API_KEYS (not needed when the server has no keys configured)")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        validate_index_specs()
        keys = load_api_keys()  # fail at startup on a malformed API_KEYS
        if not keys:
            logger.warning("API_KEYS is not set: the API is open to anyone who can reach it (development only)")
        logger.info("API ready: indexes=%s custom_index=%s clients=%s", sorted(named_indexes()),
                    get_settings().allow_custom_index, sorted(keys))

        def warm_up():  # load embedding models + stores now, so the first user request isn't slow
            for name in named_indexes():
                try:
                    pipeline(name).store.count()
                except Exception as e:
                    logger.warning("warm-up of index %r failed: %s: %s", name, type(e).__name__, e)

        threading.Thread(target=warm_up, name="warm-up", daemon=True).start()
        yield
        jobs.shutdown()

    app = FastAPI(
        title="RAG API", version="1.0.0", lifespan=lifespan,
        description="Ingest PDFs, search them, and get grounded answers with citations. "
                    "Pick an index by name (GET /v1/indexes) and any LLM from GET /v1/models.",
    )

    # ---------- plumbing ----------

    def openapi_with_file_pickers():
        """FastAPI describes uploaded files as OpenAPI 3.1 `contentMediaType`, but Swagger UI
        (/docs) only draws a file picker for `format: binary`. Add it so files can be chosen there."""
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        spec = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        for schema in spec.get("components", {}).get("schemas", {}).values():
            for prop in schema.get("properties", {}).values():
                for node in (prop, prop.get("items", {})):
                    if node.get("contentMediaType") == "application/octet-stream":
                        node["format"] = "binary"
        app.openapi_schema = spec
        return spec

    app.openapi = openapi_with_file_pickers

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = new_request_id(request.headers.get("X-Request-ID"))
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        logger.info("%s %s -> %s client=%s ms=%d", request.method, request.url.path, response.status_code,
                    getattr(request.state, "client", "-"), (time.perf_counter() - start) * 1000)
        return response

    def error(status: int, code: str, message: str, details: Optional[dict] = None) -> JSONResponse:
        body = S.ErrorResponse(error=S.ErrorBody(code=code, message=message, details=details), request_id=_request_id.get())
        return JSONResponse(status_code=status, content=body.model_dump())

    @app.exception_handler(ApiError)
    async def _api_error(request, exc: ApiError):
        response = error(exc.status, exc.code, exc.message, exc.details)
        if exc.status == 429:
            response.headers["Retry-After"] = str(exc.details["retry_after_seconds"])
        if exc.status == 401:
            response.headers["WWW-Authenticate"] = "ApiKey"
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(request, exc: RequestValidationError):
        return error(422, "invalid_request", "Request validation failed", {"errors": json.loads(json.dumps(exc.errors(), default=str))})

    @app.exception_handler(ValueError)
    async def _value_error(request, exc: ValueError):  # bad option values from the pipeline (unknown model, filter, ...)
        return error(400, "invalid_option", str(exc))

    from src.llm import LLMRefusal, ProviderError

    @app.exception_handler(LLMRefusal)
    async def _refused(request, exc: LLMRefusal):
        return error(422, "refused", str(exc))

    async def _provider_error(request, exc: Exception):
        logger.warning("model provider failed: %s: %s", type(exc).__name__, exc)
        return error(502, "provider_error", f"{type(exc).__name__}: {exc}")

    async def _store_error(request, exc: Exception):
        logger.warning("vector store unavailable: %s: %s", type(exc).__name__, exc)
        return error(503, "store_unavailable", f"{type(exc).__name__}: {exc}")

    app.add_exception_handler(ProviderError, _provider_error)

    from src.embeddings import EmbeddingError

    @app.exception_handler(EmbeddingError)
    async def _embedding_error(request, exc: EmbeddingError):
        logger.warning("embedding failed: %s", exc)
        return error(502, "embedding_error", str(exc))
    for module, attr, handler in (
        ("openai", "APIError", _provider_error),
        ("anthropic", "APIError", _provider_error),
        ("qdrant_client.http.exceptions", "ResponseHandlingException", _store_error),
        ("qdrant_client.http.exceptions", "UnexpectedResponse", _store_error),
        ("psycopg2", "OperationalError", _store_error),
    ):
        try:
            app.add_exception_handler(getattr(__import__(module, fromlist=[attr]), attr), handler)
        except (ImportError, AttributeError):
            pass  # that client library is not installed

    @app.exception_handler(Exception)
    async def _unexpected(request, exc: Exception):
        logger.exception("unhandled error")
        return error(500, "internal_error", "Internal error; see server logs with this request_id")

    def pipeline(index: S.IndexRef, chunking: Optional[S.ChunkingOptions] = None,
                 retrieval: Optional[S.RetrievalOptions] = None, llm_model: Optional[str] = None) -> RAGPipeline:
        _, kwargs = resolve_index(index)
        if chunking:
            allowed = get_settings().allowed_chunkers or list(AVAILABLE_STRATEGIES)
            if chunking.strategy and chunking.strategy not in allowed:
                raise ApiError(400, "chunker_not_allowed", f"Chunking strategy {chunking.strategy!r} is not allowed here",
                               {"allowed": allowed})
            kwargs.update(chunking_strategy=chunking.strategy, chunk_size=chunking.chunk_size,
                          chunk_overlap=chunking.overlap)
        if retrieval:
            kwargs.update(retrieval_mode=retrieval.mode, reranker=retrieval.reranker)
        kwargs["llm_model"] = llm_model
        return RAGPipeline(embedder=embedder, llm=llm, **kwargs)

    def checked_k(retrieval: S.RetrievalOptions, default: int) -> int:
        k = retrieval.k or default
        if k > get_settings().max_k:
            raise ApiError(400, "k_too_large", f"k must be <= {get_settings().max_k}")
        return k

    def where(rag: RAGPipeline) -> str:
        return f"{rag.config.vector_store}:{rag.collection}"

    # ---------- auth + rate limits ----------

    def guard(request: Request, api_key: Optional[str] = Security(api_key_header)) -> str:
        """Authenticate the client (when API_KEYS is set) and apply its rate limits."""
        keys = load_api_keys()
        if keys:
            client = identify_client(api_key, keys)
            if client is None:
                code, msg = ("missing_api_key", "Send your key in the X-API-Key header") if not api_key \
                    else ("invalid_api_key", "Unknown API key")
                raise ApiError(401, code, msg)
        else:
            client = f"ip:{request.client.host if request.client else 'unknown'}"
        request.state.client = client
        s = get_settings()
        limiter.check(client, "all", s.rate_limit_per_minute)
        if request.method == "POST" and request.url.path == "/v1/answer":
            limiter.check(client, "answers", s.rate_limit_answers_per_minute)
        return client

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(guard)])

    # ---------- health ----------

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    def ready():
        problems = {}
        for name in named_indexes():
            try:
                pipeline(name).store.count()
            except Exception as e:
                problems[name] = f"{type(e).__name__}: {e}"
        if problems:
            raise ApiError(503, "not_ready", "Some indexes are unavailable", problems)
        return {"status": "ready", "indexes": sorted(named_indexes())}

    # ---------- discovery ----------

    @v1.get("/indexes", response_model=List[S.IndexOut], tags=["discovery"])
    def list_indexes():
        out = []
        for name, spec in named_indexes().items():
            item = S.IndexOut(name=name, vector_store=spec["vector_store"], embedding_provider=spec["embedding_provider"],
                              embedding_model=spec.get("embedding_model"))
            try:
                rag = pipeline(name)
                docs = rag.list_documents()
                item.collection, item.documents, item.chunks = where(rag), len(docs), sum(docs.values())
            except Exception as e:  # one broken index must not hide the others
                item.error = f"{type(e).__name__}: {e}"
            out.append(item)
        return out

    embed_checks: dict = {}  # (index, collection) -> (checked at, error or None); avoids an API call per page load

    @v1.get("/status", response_model=S.StatusOut, tags=["discovery"])
    def status(index: str = "default"):
        """Everything the UI needs to show whether asking will work: documents, embeddings, LLM."""
        out = S.StatusOut(index=index, llm_model=get_settings().llm_model)
        try:
            rag = pipeline(index)
            out.vector_store, out.embedding_provider = rag.config.vector_store, rag.config.embedding_provider
            out.embedding_model, out.collection = rag.embedder.model_name, where(rag)
        except ApiError:
            raise
        except Exception as e:  # e.g. embedding provider can't even start (missing key)
            out.embedding_error = str(e)
            return out
        try:
            docs = rag.list_documents()
            out.documents, out.chunks = len(docs), sum(docs.values())
        except Exception as e:
            out.store_error = f"{type(e).__name__}: {e}"
        key = (index, out.collection)
        checked = embed_checks.get(key)
        if checked is None or time.monotonic() - checked[0] > 60:
            try:
                with rag._embedding_errors():
                    rag.embedder.embed_query("health check")
                checked = (time.monotonic(), None)
            except Exception as e:
                checked = (time.monotonic(), str(e))
            embed_checks[key] = checked
        out.embedding_error = checked[1]
        out.embedding_ok = checked[1] is None
        default = next((m for m in list_models() if m.name in (out.llm_model, f"ollama:{out.llm_model}")), None)
        if default:
            out.llm_available, out.llm_reason = default.available, default.reason
        return out

    @v1.get("/models", response_model=List[S.ModelOut], tags=["discovery"])
    def list_models():
        from src.llm import MODELS
        from src.llm.ollama_provider import list_local_models

        from src.llm.keycheck import key_problem

        s = get_settings()
        # Does each provider's key actually work? (free "list models" call, cached 5 minutes)
        problems = {p: key_problem(p) for p in ("openai", "anthropic")}
        pulled = set(list_local_models(s.ollama_url))
        out = []
        for name, m in MODELS.items():
            if m.provider == "ollama":
                ok, why = name in pulled, None if name in pulled else f"not pulled (ollama pull {name})"
            else:
                ok, why = problems[m.provider] is None, problems[m.provider]
            out.append(S.ModelOut(name=name, provider=m.provider, tier=m.tier, input_per_m=m.input_per_m,
                                  output_per_m=m.output_per_m, available=ok, reason=why))
        # Any other model pulled into Ollama is usable too (free, local), as "ollama:<tag>".
        for tag in sorted(pulled - set(MODELS)):
            out.append(S.ModelOut(name=f"ollama:{tag}", provider="ollama", tier="local", input_per_m=0.0,
                                  output_per_m=0.0, available=True, reason=None))
        return out

    @v1.get("/options", tags=["discovery"])
    def options():
        s = get_settings()
        return {
            "indexes": sorted(named_indexes()),
            "allow_custom_index": s.allow_custom_index,
            "vector_stores": list(AVAILABLE_STORES) if s.allow_custom_index else None,
            "chunking_strategies": s.allowed_chunkers or list(AVAILABLE_STRATEGIES),
            "retrieval_modes": list(RETRIEVAL_MODES),
            "rerankers": list(RERANKERS),
            "defaults": {"chunking_strategy": s.chunking_strategy, "chunk_size": s.chunk_size,
                         "chunk_overlap": s.chunk_overlap, "retrieval_mode": s.retrieval_mode,
                         "reranker": s.reranker, "llm_model": s.llm_model, "search_k": 5, "answer_k": 8},
            "limits": {"max_upload_mb": s.max_upload_mb, "max_k": s.max_k},
        }

    # ---------- documents ----------

    @v1.post("/documents", tags=["documents"], status_code=202,
              responses={200: {"model": S.Job}, 202: {"model": S.Job}})
    def upload_documents(
        files: List[UploadFile] = File(..., description="PDF files; re-uploading a file name replaces that document"),
        options: Optional[str] = Form(None, description='JSON: {"index": "default", "chunking": {"strategy": "semantic"}}'),
        wait: bool = Query(False, description="true: run now and return the finished job (small files only)"),
    ):
        try:
            opts = S.IngestOptions(**json.loads(options)) if options else S.IngestOptions()
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise ApiError(422, "invalid_options", f"`options` must be IngestOptions JSON: {e}")
        rag = pipeline(opts.index, chunking=opts.chunking)  # validate everything before saving files
        names = [os.path.basename(f.filename or "") for f in files]
        if any(not n.lower().endswith(".pdf") for n in names):
            raise ApiError(415, "unsupported_file", "Only .pdf files are accepted", {"files": names})
        if len(set(names)) != len(names):
            raise ApiError(400, "duplicate_file_names", "Each uploaded file needs a different name")

        limit = get_settings().max_upload_mb * 1024 * 1024
        upload_dir = tempfile.mkdtemp(prefix="upload_", dir=_uploads_root())
        try:
            paths = []
            for f, name in zip(files, names):
                path = os.path.join(upload_dir, name)
                size = 0
                with open(path, "wb") as out:
                    while chunk := f.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ApiError(413, "file_too_large", f"{name} exceeds MAX_UPLOAD_MB={get_settings().max_upload_mb}")
                        out.write(chunk)
                with open(path, "rb") as fh:
                    if fh.read(5) != b"%PDF-":
                        raise ApiError(415, "unsupported_file", f"{name} is not a PDF")
                paths.append(path)
        except Exception:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise

        def run() -> S.IngestResult:
            try:
                report = rag.ingest(paths)
                keep = _files_dir(rag)
                for f in report.files:  # keep originals so the UI can open cited pages
                    shutil.move(os.path.join(upload_dir, f["source"]), os.path.join(keep, f["source"]))
                return S.IngestResult(files=[S.FileResult(**f) for f in report.files],
                                      skipped=[S.SkippedFile(**s) for s in report.skipped],
                                      total_chunks=report.total_chunks, collection=where(rag))
            finally:
                shutil.rmtree(upload_dir, ignore_errors=True)

        job = jobs.submit(names, run)
        if wait:
            return JSONResponse(status_code=200, content=jobs.wait(job.job_id).model_dump())
        return job

    @v1.get("/jobs/{job_id}", response_model=S.Job, tags=["documents"])
    def get_job(job_id: str):
        return jobs.get(job_id)

    @v1.get("/documents", response_model=S.DocumentList, tags=["documents"])
    def list_documents(index: str = "default"):
        rag = pipeline(index)
        docs = rag.list_documents()
        return S.DocumentList(documents=[S.DocumentOut(source=k, chunks=v) for k, v in sorted(docs.items())],
                              collection=where(rag))

    @v1.delete("/documents/{source}", response_model=S.DeleteResult, tags=["documents"])
    def delete_document(source: str, index: str = "default"):
        rag = pipeline(index)
        n = rag.delete_document(source)
        stored = os.path.join(_files_dir(rag), os.path.basename(source))
        if os.path.isfile(stored):
            os.remove(stored)
        if n == 0:
            raise ApiError(404, "document_not_found", f"{source!r} is not in {where(rag)}")
        return S.DeleteResult(source=source, deleted_chunks=n, collection=where(rag))

    @v1.get("/documents/{source}/file", tags=["documents"], response_class=FileResponse,
            responses={200: {"content": {"application/pdf": {}}}})
    def document_file(source: str, index: str = "default"):
        """The original PDF (for opening citations). Available for documents uploaded through the API."""
        if os.path.basename(source) != source or not source.lower().endswith(".pdf"):
            raise ApiError(400, "invalid_source", "source must be a PDF file name")
        rag = pipeline(index)
        path = os.path.join(_files_dir(rag), source)
        if not os.path.isfile(path):
            raise ApiError(404, "file_not_available",
                           f"No stored file for {source!r} (documents ingested with the CLI are not stored)")
        return FileResponse(path, media_type="application/pdf", filename=source, content_disposition_type="inline")

    # ---------- search & answer ----------

    @v1.post("/search", response_model=S.SearchResponse, tags=["query"])
    def search(req: S.SearchRequest):
        rag = pipeline(req.index, retrieval=req.retrieval)
        hits = rag.query(req.question, k=checked_k(req.retrieval, 5), filters=req.retrieval.filters)
        return S.SearchResponse(results=[_chunk_out(h) for h in hits], collection=where(rag), request_id=_request_id.get())

    @v1.post("/answer", response_model=S.AnswerResponse, tags=["query"])
    def answer(req: S.AnswerRequest):
        rag = pipeline(req.index, retrieval=req.retrieval, llm_model=req.llm.model)
        ans = rag.answer(req.question, k=checked_k(req.retrieval, 8), filters=req.retrieval.filters,
                         max_tokens=req.llm.max_tokens)
        text, cites = _renumber_citations(ans.text, ans.citations)
        by_chunk = {s.metadata.get("chunk_id"): s for s in ans.sources}
        citations = [S.Citation(**c, snippet=" ".join(by_chunk[c["chunk_id"]].text.split())[:300]) for c in cites]
        return S.AnswerResponse(
            answer=text, found=ans.found, citations=citations,
            sources=[_chunk_out(s) for s in ans.sources] if req.include_sources else None,
            usage=S.Usage(provider=ans.provider, model=ans.model, input_tokens=ans.input_tokens, output_tokens=ans.output_tokens),
            collection=where(rag), request_id=_request_id.get(),
        )

    app.include_router(v1)
    if os.path.isdir(_WEB_DIST):
        app.mount("/", _SpaStaticFiles(directory=_WEB_DIST, html=True), name="web")
    return app


# Python's MIME table depends on the OS (on Windows it comes from the registry and lacks ".mjs").
# Browsers refuse to run module scripts served as text/plain (the PDF viewer's worker is .mjs).
for _ext in (".js", ".mjs"):
    mimetypes.add_type("text/javascript", _ext)


class _SpaStaticFiles(StaticFiles):
    """Serves the web UI; unknown paths (client-side routes like /documents) get index.html."""

    async def get_response(self, path, scope):
        from starlette.exceptions import HTTPException as StarletteHTTPException

        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            # Decide on the original request path: `path` is already normalized (".." resolved,
            # backslashes on Windows), so it can no longer tell an API URL from a UI route.
            if e.status_code == 404 and not scope["path"].startswith(("/v1/", "/api/")):
                return await super().get_response("index.html", scope)
            raise


def _files_dir(rag: RAGPipeline) -> str:
    """Where original PDFs of one index's collection are kept: DATA_DIR/files/<store>_<collection>."""
    root = os.path.dirname(_uploads_root())
    path = os.path.join(root, "files", f"{rag.config.vector_store}_{rag.collection}")
    os.makedirs(path, exist_ok=True)
    return path


def _uploads_root() -> str:
    s = get_settings()
    root = s.data_dir if os.path.isabs(s.data_dir) else os.path.join(str(_project_root()), s.data_dir)
    path = os.path.join(root, "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def _project_root():
    from src.config import PROJECT_ROOT

    return PROJECT_ROOT


_WEB_DIST = str(_project_root() / "web" / "dist")

app = create_app()
