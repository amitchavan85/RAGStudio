"""
Model catalog: which provider serves a model, plus price and tier data.

Prices are USD per 1M tokens (standard, non-batch), checked 2026-10-13 against
developers.openai.com/api/docs/pricing and Anthropic's model table. They are recorded so
cost-based selection (budgets, tiers, an auto-router) can be added later; nothing uses
them yet. Update them here when providers change prices.

Any model not listed can still be used as "<provider>:<model>", e.g. "openai:gpt-5.1-mini"
or "ollama:mistral:7b" (Ollama tags contain a colon too; only the first one is the prefix).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

PROVIDERS = ("openai", "anthropic", "ollama")


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    tier: str              # local | economy | standard | premium (for future cost-based selection)
    input_per_m: float     # USD per 1M input tokens
    output_per_m: float    # USD per 1M output tokens


MODELS: Dict[str, ModelInfo] = {
    # OpenAI
    "gpt-5-nano": ModelInfo("openai", "economy", 0.05, 0.40),
    "gpt-4.1-nano": ModelInfo("openai", "economy", 0.10, 0.40),
    "gpt-4o-mini": ModelInfo("openai", "economy", 0.15, 0.60),
    "gpt-5.4-nano": ModelInfo("openai", "economy", 0.20, 1.25),
    "gpt-5-mini": ModelInfo("openai", "economy", 0.25, 2.00),
    "gpt-4.1-mini": ModelInfo("openai", "economy", 0.40, 1.60),
    "gpt-5.4-mini": ModelInfo("openai", "standard", 0.75, 4.50),
    "gpt-5": ModelInfo("openai", "standard", 1.25, 10.00),
    "gpt-5.1": ModelInfo("openai", "standard", 1.25, 10.00),
    "gpt-5.2": ModelInfo("openai", "standard", 1.75, 14.00),
    "gpt-4.1": ModelInfo("openai", "standard", 2.00, 8.00),
    "gpt-5.4": ModelInfo("openai", "premium", 2.50, 15.00),
    "gpt-5.5": ModelInfo("openai", "premium", 5.00, 30.00),
    # Anthropic
    "claude-haiku-5-5": ModelInfo("anthropic", "economy", 0.10, 0.50),
    "claude-sonnet-5-5": ModelInfo("anthropic", "standard", 2.00, 10.00),
    "claude-opus-5-5": ModelInfo("anthropic", "premium", 4.00, 20.00),
    "claude-fable-5-1": ModelInfo("anthropic", "premium", 10.00, 50.00),
    # Ollama (runs locally, free; pull first: `ollama pull llama3.2:3b`)
    "llama3.2:3b": ModelInfo("ollama", "local", 0.0, 0.0),
    "qwen2.5:7b": ModelInfo("ollama", "local", 0.0, 0.0),
}


def resolve(model: str) -> Tuple[str, str]:
    """Model name -> (provider, provider's model id)."""
    model = model.strip()
    if model in MODELS:
        return MODELS[model].provider, model
    prefix, sep, rest = model.partition(":")
    if sep and prefix in PROVIDERS and rest:
        return prefix, rest
    raise ValueError(
        f"Unknown model {model!r}. Use one of {sorted(MODELS)} "
        f"or '<provider>:<model>' with provider in {PROVIDERS}."
    )
