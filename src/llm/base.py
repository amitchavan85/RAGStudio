"""LLM provider contract: every provider turns (system, prompt) into an LLMResponse."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str
    model: str          # model that actually answered (may differ after a server-side fallback)
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None


class ProviderError(RuntimeError):
    """The model provider could not be reached or returned an error (e.g. Ollama down, model not pulled)."""


class LLMRefusal(RuntimeError):
    """The model declined to answer (safety policy). Not retried."""


class LLM(ABC):
    provider: str = "unknown"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 4096) -> LLMResponse:
        """One request -> one complete text response, with token usage."""

    def __call__(self, prompt: str) -> str:
        """Plain prompt -> text, the shape the agentic chunker expects (llm_fn)."""
        return self.generate(prompt).text
