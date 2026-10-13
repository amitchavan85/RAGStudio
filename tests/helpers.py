import hashlib
from pathlib import Path

import numpy as np

from src.embeddings.base import EmbeddingProvider
from src.llm.base import LLM, LLMResponse

# Sample PDFs used by the tests: a committed copy in tests/data, so tests (and CI) never depend
# on what is in your own data folder (src/data/pdfs or PDF_DIR).
PDF_DIR = str(Path(__file__).resolve().parent / "data" / "pdfs")


class HashEmbedder(EmbeddingProvider):
    """Deterministic bag-of-words hashing embedder; no model download."""

    dim = 64
    model_name = "hash-test"

    @property
    def dimension(self):
        return self.dim

    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            out.append(v.tolist())
        return out


class FakeLLM(LLM):
    """Records the prompt; answers with a fixed text."""
    provider = "fake"

    def __init__(self, reply="The answer is here [1].", model="fake-1"):
        super().__init__(model)
        self.reply, self.calls = reply, []

    def generate(self, prompt, system=None, max_tokens=4096):
        self.calls.append({"prompt": prompt, "system": system})
        return LLMResponse(text=self.reply, model=self.model, provider=self.provider, input_tokens=100, output_tokens=10)
