"""Live answer generation per provider. Each test skips when its provider is not available:
OpenAI needs OPENAI_API_KEY, Anthropic needs ANTHROPIC_API_KEY, Ollama needs a pulled model.
Uses the cheapest model of each provider; one short question costs well under a cent."""
import os

import pytest
from helpers import PDF_DIR, HashEmbedder

from src.config import Settings
from src.llm.ollama_provider import list_local_models
from src.pipeline import RAGPipeline

S = Settings()
OLLAMA_MODELS = list_local_models(S.ollama_url)

PROVIDERS = [
    pytest.param("gpt-4o-mini", marks=pytest.mark.skipif(not S.openai_api_key.get_secret_value(), reason="OPENAI_API_KEY not set")),
    pytest.param("claude-haiku-5-5", marks=pytest.mark.skipif(not S.anthropic_api_key.get_secret_value() and not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")),
    pytest.param(f"ollama:{OLLAMA_MODELS[0]}" if OLLAMA_MODELS else "ollama:none",
                 marks=pytest.mark.skipif(not OLLAMA_MODELS, reason="no Ollama model pulled (try: ollama pull llama3.2:3b)")),
]


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    from src.config import get_settings

    old = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(tmp_path_factory.mktemp("live_answers"))
    get_settings.cache_clear()
    p = RAGPipeline(vector_store="faiss", embedding_provider="sentence_transformers",
                    collection_name="livegen", embedder=HashEmbedder())
    p.ingest([os.path.join(PDF_DIR, "usa_gdp_report.pdf"), os.path.join(PDF_DIR, "ind_gdp_report.pdf")])
    yield p
    os.environ.pop("DATA_DIR", None) if old is None else os.environ.__setitem__("DATA_DIR", old)
    get_settings.cache_clear()


@pytest.mark.parametrize("model", PROVIDERS)
def test_grounded_answer_with_citation(rag, model):
    ans = rag.answer("What was the nominal GDP of the United States in 2023?", k=4, mode="keyword",
                     model=model, filters={"source": "usa_gdp_report.pdf"})
    assert "27,360" in ans.text, ans.text
    assert ans.citations, f"no [n] citation in: {ans.text}"
    assert all(c["source"] == "usa_gdp_report.pdf" for c in ans.citations)
    assert ans.input_tokens > 0 and ans.output_tokens > 0


@pytest.mark.parametrize("model", PROVIDERS)
def test_says_not_found_instead_of_guessing(rag, model):
    ans = rag.answer("What is the population of Mars?", k=3, model=model)
    assert not ans.found, ans.text
