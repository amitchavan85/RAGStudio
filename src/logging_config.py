"""
Logging setup: one call at startup, text output to console and to a
rotating file in LOG_DIR. Every log line carries a request id so one
ingest/query can be followed across all pipeline steps.

    from src.logging_config import setup_logging, new_request_id, log_step

    setup_logging()
    new_request_id()                       # once per ingest/query request
    with log_step(logger, "embed", n=44):  # one timed line per pipeline step
        ...
"""
from __future__ import annotations

import contextvars
import logging
import os
import time
import uuid
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_FORMAT = "%(asctime)s %(levelname)-7s [req=%(request_id)s] %(name)s: %(message)s"
_NOISY_LIBS = ("chromadb", "httpx", "httpx2", "httpcore", "openai", "urllib3", "sentence_transformers", "qdrant_client", "faiss")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def new_request_id(value: str | None = None) -> str:
    """Start a new request context (pass an incoming X-Request-ID to reuse it)."""
    rid = value or uuid.uuid4().hex[:8]
    _request_id.set(rid)
    return rid


def setup_logging(level: str | None = None, log_dir: str | None = None) -> None:
    """Configure the root logger. Safe to call more than once."""
    from src.config import get_settings

    s = get_settings()
    level = (level or s.log_level).upper()
    log_dir = s.log_dir if log_dir is None else log_dir
    if log_dir and not os.path.isabs(log_dir):
        from src.config import PROJECT_ROOT

        log_dir = str(PROJECT_ROOT / log_dir)  # same folder no matter where you run from

    root = logging.getLogger()
    for h in list(root.handlers):  # idempotent: drop handlers from a previous call
        if getattr(h, "_rag_handler", False):
            root.removeHandler(h)
            h.close()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(
            RotatingFileHandler(os.path.join(log_dir, "rag.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        )
    for h in handlers:
        h._rag_handler = True  # type: ignore[attr-defined]
        h.setFormatter(logging.Formatter(_FORMAT))
        h.addFilter(_RequestIdFilter())
        root.addHandler(h)
    root.setLevel(level)

    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def log_step(logger: logging.Logger, step: str, **fields):
    """Log one timed line per pipeline step: DEBUG on start, INFO on success, ERROR on failure.

    Yields a dict; add result fields to it (e.g. out["chunks"] = 44) and they
    are included in the final line. Only log counts/ids/durations here, never
    document text or secrets.
    """
    extra = dict(fields)
    logger.debug("%s start %s", step, _fmt(extra))
    start = time.perf_counter()
    try:
        yield extra
    except Exception:
        logger.exception("%s failed after %d ms %s", step, (time.perf_counter() - start) * 1000, _fmt(extra))
        raise
    logger.info("%s done %s ms=%d", step, _fmt(extra), (time.perf_counter() - start) * 1000)


def _fmt(fields: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())
