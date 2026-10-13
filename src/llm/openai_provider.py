"""OpenAI chat models (Chat Completions API)."""
from __future__ import annotations

import logging
from typing import Optional

from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.base import LLM, LLMResponse

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    provider = "openai"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set (add it to .env or the environment)")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def _transient_errors(self):
        import openai

        return (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 4096) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        if self.model.startswith("gpt-4"):
            kwargs["temperature"] = 0  # gpt-5 family accepts only the default temperature

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15), reraise=True,
               retry=retry_if_exception_type(self._transient_errors()),
               before_sleep=before_sleep_log(logger, logging.WARNING))
        def call():
            return self._client.chat.completions.create(**kwargs)

        resp = call()
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            provider=self.provider,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop_reason=choice.finish_reason,
        )
