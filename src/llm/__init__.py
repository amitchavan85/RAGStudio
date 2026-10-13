"""
LLM access across providers.

    from src.llm import get_llm
    llm = get_llm("claude-haiku-5-5")      # or "gpt-4o-mini", "ollama:llama3.2:3b"; None = LLM_MODEL
    resp = llm.generate("prompt", system="...")   # LLMResponse(text, model, provider, tokens...)
    llm("prompt")                                  # plain text, usable as the agentic chunker's llm_fn

The model name decides the provider (see catalog.py). Keys come from .env:
OPENAI_API_KEY, ANTHROPIC_API_KEY; Ollama needs none (OLLAMA_URL, default localhost).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, Optional

from src.llm.base import LLM, LLMRefusal, LLMResponse, ProviderError
from src.llm.catalog import MODELS, PROVIDERS, ModelInfo, resolve
from src.logging_config import log_step

logger = logging.getLogger(__name__)


class _LoggedLLM(LLM):
    """Adds one log line per call (provider, model, tokens, duration) around any provider."""

    def __init__(self, inner: LLM):
        super().__init__(inner.model)
        self.provider = inner.provider
        self._inner = inner

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 4096) -> LLMResponse:
        with log_step(logger, "llm", provider=self.provider, model=self.model, prompt_chars=len(prompt)) as out:
            resp = self._inner.generate(prompt, system=system, max_tokens=max_tokens)
            out.update(served_by=resp.model, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)
        return resp


@lru_cache(maxsize=8)
def get_llm(model: Optional[str] = None) -> LLM:
    """LLM client for a model name (default: LLM_MODEL). Clients are reused per model."""
    from src.config import get_settings

    s = get_settings()
    provider, model_id = resolve(model or s.llm_model)
    if provider == "openai":
        from src.llm.openai_provider import OpenAILLM

        inner: LLM = OpenAILLM(model_id, api_key=s.openai_api_key.get_secret_value())
    elif provider == "anthropic":
        from src.llm.anthropic_provider import AnthropicLLM

        inner = AnthropicLLM(model_id, api_key=s.anthropic_api_key.get_secret_value())
    else:
        from src.llm.ollama_provider import OllamaLLM

        inner = OllamaLLM(model_id, url=s.ollama_url)
    return _LoggedLLM(inner)


def get_llm_fn(model: Optional[str] = None) -> Callable[[str], str]:
    """prompt -> text function (what the agentic chunker takes)."""
    return get_llm(model)


__all__ = ["LLM", "LLMRefusal", "LLMResponse", "ProviderError", "MODELS", "ModelInfo", "PROVIDERS", "get_llm", "get_llm_fn", "resolve"]
