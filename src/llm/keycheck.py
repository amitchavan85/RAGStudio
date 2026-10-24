"""
Does a provider's API key actually work? Checked with the providers' free "list models" calls
(no tokens are billed) and cached, so the UI can show a revoked key before a question fails.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, Optional, Tuple

TTL_SECONDS = 300
_cache: Dict[str, Tuple[float, Optional[str]]] = {}
_lock = threading.Lock()


def _check(provider: str) -> Optional[str]:
    from src.config import get_settings

    s = get_settings()
    if provider == "openai":
        key = s.openai_api_key.get_secret_value()
        if not key:
            return "OPENAI_API_KEY not set"
        import openai

        try:
            openai.OpenAI(api_key=key, timeout=5.0, max_retries=0).models.list()
        except openai.AuthenticationError:
            return "OPENAI_API_KEY is invalid or revoked"
        except openai.PermissionDeniedError:
            return "OPENAI_API_KEY lacks permission"
        except Exception:
            return None  # network problem etc.: can't tell, don't block the model
        return None
    if provider == "anthropic":
        key = s.anthropic_api_key.get_secret_value() or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return "ANTHROPIC_API_KEY not set"
        import anthropic

        try:
            anthropic.Anthropic(api_key=key, timeout=5.0, max_retries=0).models.list(limit=1)
        except anthropic.AuthenticationError:
            return "ANTHROPIC_API_KEY is invalid or revoked"
        except anthropic.PermissionDeniedError:
            return "ANTHROPIC_API_KEY lacks permission"
        except Exception:
            return None
        return None
    return None


def key_problem(provider: str) -> Optional[str]:
    """None if the provider's key works (or can't be checked right now), else a short reason."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(provider)
        if hit and now - hit[0] < TTL_SECONDS:
            return hit[1]
    problem = _check(provider)
    with _lock:
        _cache[provider] = (now, problem)
    return problem


def clear_cache() -> None:
    with _lock:
        _cache.clear()
