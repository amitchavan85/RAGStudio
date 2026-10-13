"""API internals: errors, named indexes, background ingest jobs."""
from __future__ import annotations

import contextvars
import hmac
import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from src.api.schemas import IndexRef, IndexSpec, Job
from src.config import get_settings

_INDEX_KEYS = {"vector_store", "embedding_provider", "embedding_model", "collection"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


# ---------- named indexes ----------

def named_indexes() -> Dict[str, dict]:
    """Indexes users may pick by name. From INDEXES in .env, or one "default" index."""
    s = get_settings()
    defaults = {"vector_store": s.vector_store, "embedding_provider": s.embedding_provider,
                "embedding_model": None, "collection": s.collection_name}
    if not s.indexes:
        return {"default": defaults}
    out = {}
    for name, spec in s.indexes.items():
        unknown = set(spec) - _INDEX_KEYS
        if unknown:
            raise ValueError(f"INDEXES[{name!r}] has unknown keys {sorted(unknown)}; allowed: {sorted(_INDEX_KEYS)}")
        out[name] = {**defaults, **spec}
    return out


def resolve_index(ref: IndexRef) -> Tuple[str, dict]:
    """Index reference -> (display name, pipeline keyword arguments)."""
    indexes = named_indexes()
    if isinstance(ref, str):
        if ref not in indexes:
            raise ApiError(404, "index_not_found", f"Unknown index {ref!r}", {"available": sorted(indexes)})
        spec, name = indexes[ref], ref
    else:
        if not get_settings().allow_custom_index:
            raise ApiError(403, "custom_index_disabled",
                           "Custom index specs are disabled; use a named index (see GET /v1/indexes).",
                           {"available": sorted(indexes)})
        base = next(iter(indexes.values()))
        spec = {**base, **{k: v for k, v in ref.model_dump().items() if v is not None}}
        name = "custom"
    return name, {
        "vector_store": spec["vector_store"],
        "embedding_provider": spec["embedding_provider"],
        "embedding_model": spec.get("embedding_model"),
        "collection_name": spec["collection"],
    }


# ---------- background ingest jobs ----------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    """In-memory ingest jobs (lost on restart; enough for a single API process).

    Runs at most `workers` ingests at a time and keeps the last `keep` jobs.
    """

    def __init__(self, workers: int = 2, keep: int = 500):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._futures: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def submit(self, files: List[str], fn: Callable[[], object]) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], status="queued", created_at=_now(), files=files)
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > self._keep:
                old_id, _ = self._jobs.popitem(last=False)
                self._futures.pop(old_id, None)
            ctx = contextvars.copy_context()  # keep the request id in the job's log lines
            self._futures[job.job_id] = self._pool.submit(ctx.run, self._run, job.job_id, fn)
        return job

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Job:
        """Block until the job finished (it never raises: failures are recorded on the job)."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.get(job_id)

    def _run(self, job_id: str, fn: Callable[[], object]) -> None:
        self._update(job_id, status="running")
        try:
            result = fn()
            self._update(job_id, status="succeeded", result=result, finished_at=_now())
        except Exception as e:  # recorded on the job; the client reads it from GET /v1/jobs/{id}
            message = e.message if isinstance(e, ApiError) else f"{type(e).__name__}: {e}"
            self._update(job_id, status="failed", error=message, finished_at=_now())

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id] = self._jobs[job_id].model_copy(update=fields)

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ApiError(404, "job_not_found", f"Unknown job {job_id!r}")
        return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def validate_index_specs() -> None:
    """Fail at startup (not on the first request) if INDEXES is malformed."""
    for name, spec in named_indexes().items():
        IndexSpec(**spec)  # checks vector_store / embedding_provider values


# ---------- auth + rate limits ----------

def load_api_keys() -> Dict[str, str]:
    """{client_name: key} from API_KEYS (JSON). Empty dict = authentication disabled."""
    raw = get_settings().api_keys.get_secret_value().strip()
    if not raw:
        return {}
    try:
        keys = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"API_KEYS must be a JSON object like {{\"webapp\": \"key\"}}: {e}") from None
    if not isinstance(keys, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in keys.items()):
        raise ValueError('API_KEYS must be a JSON object of client name -> non-empty key, e.g. {"webapp": "key"}')
    short = [name for name, key in keys.items() if len(key) < 16]
    if short:
        raise ValueError(f"API keys must be at least 16 characters (too short for: {short})")
    return keys


def identify_client(presented: Optional[str], keys: Dict[str, str]) -> Optional[str]:
    """Client name for a presented key (constant-time comparison), or None if it matches none."""
    if not presented:
        return None
    match = None
    for name, key in keys.items():
        if hmac.compare_digest(presented.encode(), key.encode()):
            match = name
    return match


class RateLimiter:
    """Token bucket per (client, bucket): `limit` requests per minute, refilled continuously.

    In memory, so limits are per API process (fine for the single-worker setup; several workers
    or servers would need a shared store such as Redis).
    """

    def __init__(self):
        self._buckets: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str, bucket: str, limit: int) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        rate = limit / 60.0
        with self._lock:
            tokens, last = self._buckets.get((client, bucket), (float(limit), now))
            tokens = min(float(limit), tokens + (now - last) * rate)
            if tokens < 1.0:
                retry = math.ceil((1.0 - tokens) / rate)
                self._buckets[(client, bucket)] = (tokens, now)
                raise ApiError(429, "rate_limited", f"Too many requests: limit is {limit} per minute for {bucket}",
                               {"limit_per_minute": limit, "bucket": bucket, "retry_after_seconds": retry})
            self._buckets[(client, bucket)] = (tokens - 1.0, now)
