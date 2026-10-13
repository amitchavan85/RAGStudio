"""Roadmap P4: HTTP API contract (FastAPI test client, fake embedder + fake LLM, no network)."""
import json
import os

import pytest
from fastapi.testclient import TestClient
from helpers import PDF_DIR, FakeLLM, HashEmbedder

from src.api.app import create_app
from src.llm import LLMRefusal

USA = os.path.join(PDF_DIR, "usa_gdp_report.pdf")
IND = os.path.join(PDF_DIR, "ind_gdp_report.pdf")
INDEXES = {
    "default": {"vector_store": "faiss", "embedding_provider": "sentence_transformers", "collection": "apitest"},
    "archive": {"vector_store": "chroma", "embedding_provider": "sentence_transformers", "collection": "archive"},
}


def _pdf(path):
    return ("files", (os.path.basename(path), open(path, "rb"), "application/pdf"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INDEXES", json.dumps(INDEXES))
    monkeypatch.setenv("LOG_DIR", "")
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


@pytest.fixture
def llm():
    return FakeLLM(reply="India grew [5] and the US grew [2].")


@pytest.fixture
def client(env, llm):
    with TestClient(create_app(embedder=HashEmbedder(), llm=llm)) as c:
        yield c


@pytest.fixture
def loaded(client):
    r = client.post("/v1/documents?wait=true", files=[_pdf(USA), _pdf(IND)])
    assert r.status_code == 200, r.text
    return client


# ---------- health & discovery ----------

def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["indexes"] == ["archive", "default"]


def test_indexes_list_with_counts(loaded):
    by_name = {i["name"]: i for i in loaded.get("/v1/indexes").json()}
    assert by_name["default"]["vector_store"] == "faiss" and by_name["default"]["documents"] == 2
    assert by_name["archive"]["documents"] == 0
    assert by_name["default"]["collection"].startswith("faiss:apitest_hash_test_64")


def test_options_and_models(client):
    o = client.get("/v1/options").json()
    assert "semantic" in o["chunking_strategies"] and o["defaults"]["answer_k"] == 8
    assert o["allow_custom_index"] is False and o["vector_stores"] is None
    models = {m["name"]: m for m in client.get("/v1/models").json()}
    assert models["claude-haiku-5-5"]["provider"] == "anthropic"
    assert all(m["available"] or m["reason"] for m in models.values())


# ---------- documents ----------

def test_upload_wait_list_reupload_delete(loaded):
    docs = loaded.get("/v1/documents").json()
    assert {d["source"] for d in docs["documents"]} == {"usa_gdp_report.pdf", "ind_gdp_report.pdf"}

    opts = json.dumps({"chunking": {"strategy": "sentence", "chunk_size": 200}})
    job = loaded.post("/v1/documents?wait=true", files=[_pdf(USA)], data={"options": opts}).json()
    assert job["status"] == "succeeded" and job["result"]["files"][0]["replaced"] > 0

    r = loaded.delete("/v1/documents/usa_gdp_report.pdf")
    assert r.status_code == 200 and r.json()["deleted_chunks"] == job["result"]["files"][0]["chunks"]
    assert loaded.delete("/v1/documents/usa_gdp_report.pdf").json()["error"]["code"] == "document_not_found"


def test_upload_returns_job_to_poll(client):
    r = client.post("/v1/documents", files=[_pdf(USA)])
    assert r.status_code == 202 and r.json()["status"] in ("queued", "running", "succeeded")
    from src.api import app as app_module  # noqa: F401  (job finishes in the background)

    for _ in range(200):
        job = client.get(f"/v1/jobs/{r.json()['job_id']}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        import time

        time.sleep(0.05)
    assert job["status"] == "succeeded" and job["result"]["total_chunks"] > 0


def test_upload_into_named_index(client):
    r = client.post("/v1/documents?wait=true", files=[_pdf(IND)], data={"options": json.dumps({"index": "archive"})})
    assert r.json()["result"]["collection"].startswith("chroma:archive")
    assert [d["source"] for d in client.get("/v1/documents?index=archive").json()["documents"]] == ["ind_gdp_report.pdf"]
    assert client.get("/v1/documents").json()["documents"] == []  # default index untouched


@pytest.mark.parametrize("files, status, code", [
    ([("files", ("notes.txt", b"hello", "text/plain"))], 415, "unsupported_file"),
    ([("files", ("fake.pdf", b"not really a pdf", "application/pdf"))], 415, "unsupported_file"),
    ([("files", ("a.pdf", b"%PDF-1", "application/pdf")), ("files", ("a.pdf", b"%PDF-1", "application/pdf"))], 400, "duplicate_file_names"),
])
def test_upload_rejects_bad_files(client, files, status, code):
    r = client.post("/v1/documents?wait=true", files=files)
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_upload_size_limit(env, llm):
    env.setenv("MAX_UPLOAD_MB", "0")
    from src.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app(embedder=HashEmbedder(), llm=llm)) as c:
        r = c.post("/v1/documents?wait=true", files=[_pdf(USA)])
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_upload_option_errors(client, env):
    assert client.post("/v1/documents", files=[_pdf(USA)], data={"options": "{not json"}).json()["error"]["code"] == "invalid_options"
    assert client.post("/v1/documents", files=[_pdf(USA)], data={"options": '{"chunkin": {}}'}).status_code == 422  # typo
    r = client.post("/v1/documents", files=[_pdf(USA)], data={"options": '{"index": "nope"}'})
    assert r.status_code == 404 and r.json()["error"]["details"]["available"] == ["archive", "default"]
    env.setenv("ALLOWED_CHUNKERS", '["recursive"]')
    from src.config import get_settings

    get_settings.cache_clear()
    r = client.post("/v1/documents", files=[_pdf(USA)], data={"options": '{"chunking": {"strategy": "agentic"}}'})
    assert r.status_code == 400 and r.json()["error"]["code"] == "chunker_not_allowed"


def test_failed_ingest_is_reported_on_the_job(client, monkeypatch):
    def boom(texts):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(HashEmbedder, "embed_documents", lambda self, texts: boom(texts))
    job = client.post("/v1/documents?wait=true", files=[_pdf(USA)]).json()
    assert job["status"] == "failed" and "embedding service down" in job["error"]


def test_unknown_job(client):
    assert client.get("/v1/jobs/nope").status_code == 404


# ---------- custom index specs ----------

def test_custom_index_disabled_by_default(client):
    r = client.post("/v1/search", json={"question": "GDP", "index": {"vector_store": "qdrant"}})
    assert r.status_code == 403 and r.json()["error"]["code"] == "custom_index_disabled"


def test_custom_index_when_enabled(env, llm):
    env.setenv("ALLOW_CUSTOM_INDEX", "true")
    from src.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app(embedder=HashEmbedder(), llm=llm)) as c:
        opts = json.dumps({"index": {"vector_store": "chroma", "collection": "mine"}})
        assert c.post("/v1/documents?wait=true", files=[_pdf(USA)], data={"options": opts}).json()["status"] == "succeeded"
        r = c.post("/v1/search", json={"question": "27,360", "index": {"vector_store": "chroma", "collection": "mine"}})
        assert r.status_code == 200 and r.json()["collection"].startswith("chroma:mine")


# ---------- search & answer ----------

def test_search_with_options_and_filters(loaded):
    r = loaded.post("/v1/search", json={"question": "27,360", "retrieval": {"mode": "keyword", "k": 3,
                                                                          "filters": {"source": "usa_gdp_report.pdf"}}})
    body = r.json()
    assert r.status_code == 200 and len(body["results"]) <= 3
    assert {x["source"] for x in body["results"]} == {"usa_gdp_report.pdf"} and "keyword" in body["results"][0]["scores"]


@pytest.mark.parametrize("payload, status", [
    ({"question": ""}, 422),
    ({"question": "x", "retrieval": {"mode": "fuzzy"}}, 422),
    ({"question": "x", "retrieval": {"k": 999}}, 400),
    ({"question": "x", "retrieval": {"filters": {"bad key": 1}}}, 400),
    ({"question": "x", "unknown_field": 1}, 422),
])
def test_search_validation(loaded, payload, status):
    assert loaded.post("/v1/search", json=payload).status_code == status


def test_answer_renumbers_citations_and_reports_usage(loaded, llm):
    r = loaded.post("/v1/answer", json={"question": "Compare India and US growth", "include_sources": True},
                    headers={"X-Request-ID": "test-req-1"})
    body = r.json()
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "test-req-1" and body["request_id"] == "test-req-1"
    assert body["answer"] == "India grew [1] and the US grew [2]."           # [5],[2] -> [1],[2]
    assert [c["n"] for c in body["citations"]] == [1, 2] and all(c["snippet"] for c in body["citations"])
    assert len(body["sources"]) == 8 and body["found"] is True
    assert body["usage"] == {"provider": "fake", "model": "fake-1", "input_tokens": 100, "output_tokens": 10}


def test_answer_model_validation_and_refusal(loaded, llm, monkeypatch):
    r = loaded.post("/v1/answer", json={"question": "x", "llm": {"model": "no-such-model"}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_option"

    def refuse(prompt, system=None, max_tokens=4096):
        raise LLMRefusal("model declined (category: cyber)")

    monkeypatch.setattr(llm, "generate", refuse)
    r = loaded.post("/v1/answer", json={"question": "GDP"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "refused"


def test_answer_on_empty_index_skips_llm(client, llm):
    r = client.post("/v1/answer", json={"question": "GDP"})
    assert r.status_code == 200 and r.json()["found"] is False and llm.calls == []


def test_openapi_docs_available(client):
    spec = client.get("/openapi.json").json()
    assert {"/v1/answer", "/v1/search", "/v1/documents", "/v1/indexes", "/v1/models"} <= set(spec["paths"])


def test_provider_failure_maps_to_502(loaded, llm, monkeypatch):
    from src.llm import ProviderError

    def down(prompt, system=None, max_tokens=4096):
        raise ProviderError("Ollama is not reachable at http://localhost:11434")

    monkeypatch.setattr(llm, "generate", down)
    r = loaded.post("/v1/answer", json={"question": "GDP"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "provider_error"


def test_swagger_ui_shows_file_pickers_for_uploads(client):
    """Swagger UI (/docs) needs format=binary to render a file chooser for each uploaded file."""
    spec = client.get("/openapi.json").json()
    ref = spec["paths"]["/v1/documents"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    files = spec["components"]["schemas"][ref.split("/")[-1]]["properties"]["files"]
    assert files["type"] == "array" and files["items"]["format"] == "binary"


def test_uploaded_pdf_is_kept_served_and_deleted(loaded):
    r = loaded.get("/v1/documents/usa_gdp_report.pdf/file")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-") and "inline" in r.headers["content-disposition"]
    assert loaded.get("/v1/documents/missing.pdf/file").json()["error"]["code"] == "file_not_available"
    assert loaded.get("/v1/documents/..%2F..%2Fsecret.pdf/file").status_code in (400, 404)
    loaded.delete("/v1/documents/usa_gdp_report.pdf")
    assert loaded.get("/v1/documents/usa_gdp_report.pdf/file").status_code == 404


def test_web_ui_is_served_with_spa_fallback_but_api_404s_stay_json(env, llm, tmp_path, monkeypatch):
    from src.api import app as app_module

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>RAG Studio</title>")
    (dist / "assets").mkdir()
    (dist / "assets" / "pdf.worker.min.mjs").write_text("export {};")
    monkeypatch.setattr(app_module, "_WEB_DIST", str(dist))
    with TestClient(create_app(embedder=HashEmbedder(), llm=llm)) as c:
        assert "RAG Studio" in c.get("/").text
        assert "RAG Studio" in c.get("/documents").text  # client-side route -> index.html
        r = c.get("/v1/nope")
        assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
        assert c.get("/health").json() == {"status": "ok"}
        # module scripts must be served as JavaScript, or the browser refuses to run them (PDF viewer worker)
        assert c.get("/assets/pdf.worker.min.mjs").headers["content-type"].startswith("text/javascript")
