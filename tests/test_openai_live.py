"""Live OpenAI check: skipped automatically unless OPENAI_API_KEY is set in .env / environment.

Makes one tiny embeddings request (a few tokens, well under a cent).
"""
import pytest

from src.config import Settings
from src.embeddings.openai_provider import OpenAIEmbeddings

KEY = Settings().openai_api_key.get_secret_value()

pytestmark = pytest.mark.skipif(not KEY, reason="OPENAI_API_KEY not set in .env or environment")


def test_openai_key_format():
    assert KEY.startswith("sk-"), "key does not look like an OpenAI key"
    assert KEY == KEY.strip()


def test_openai_key_works_for_embeddings():
    emb = OpenAIEmbeddings(api_key=KEY)
    vecs = emb.embed_documents(["GDP growth"])
    assert len(vecs) == 1
    assert len(vecs[0]) == emb.dimension == 1536
