"""Anthropic Claude models (Messages API, official `anthropic` SDK)."""
from __future__ import annotations

from typing import Optional

from src.llm.base import LLM, LLMRefusal, LLMResponse

# Models that accept server-side refusal fallbacks ("default" routing). On a policy decline the
# API re-runs the request on Anthropic's recommended fallback model inside the same call.
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str = ""):
        super().__init__(model)
        import anthropic

        # No key passed -> the SDK resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`.
        # The SDK retries 408/409/429/5xx and connection errors itself (max_retries=2).
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 4096) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.model in _FALLBACK_MODELS:
            resp = self._client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            resp = self._client.messages.create(**kwargs)

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"{resp.model} declined to answer (category: {category})")
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=resp.model,  # names the fallback model if one served the request
            provider=self.provider,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
        )
