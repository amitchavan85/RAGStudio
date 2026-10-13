"""Local models through Ollama's HTTP API (free; no API key). Pull a model first: `ollama pull llama3.2:3b`."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Optional

from src.llm.base import LLM, LLMResponse, ProviderError


def list_local_models(url: str = "http://localhost:11434") -> List[str]:
    """Models already pulled into the local Ollama server ([] if it is not running)."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(self, model: str, url: str = "http://localhost:11434", timeout: float = 300.0):
        super().__init__(model)
        self.url = url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 4096) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            raise ProviderError(f"Ollama error for {self.model!r}: {detail} (did you `ollama pull {self.model}`?)") from e
        except urllib.error.URLError as e:
            raise ProviderError(f"Ollama is not reachable at {self.url} ({e.reason})") from e
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            provider=self.provider,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            stop_reason=data.get("done_reason"),
        )
