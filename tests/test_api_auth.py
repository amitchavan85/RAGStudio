"""Roadmap P6: API keys and rate limits."""
import json

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, HashEmbedder

from src.api.app import create_app
from src.api.core import RateLimiter, ApiError, identify_client, load_api_keys

KEYS = {"webapp": "webapp-key-0123456789", "ci": "ci-key-abcdefghijklmn"}
INDEXES = {"default": {"vector_store": "faiss", "embedding_provider": "sentence_transformers", "collection": "authtest"}}


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    from src.config import get_settings

    def make(keys=KEYS, per_minute=120, answers_per_minute=30):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        monkeypatch.setenv("INDEXES", json.dumps(INDEXES))
        monkeypatch.setenv("LOG_DIR", "")
        monkeypatch.setenv("API_KEYS", json.dumps(keys) if keys else "")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", str(per_minute))
        monkeypatch.setenv("RATE_LIMIT_ANSWERS_PER_MINUTE", str(answers_per_minute))
        get_settings.cache_clear()
        return TestClient(create_app(embedder=HashEmbedder(), llm=FakeLLM()))

    yield make
    get_settings.cache_clear()


def test_keys_required_on_v1_but_not_on_health_or_docs(make_client):
    with make_client() as c:
        r = c.get("/v1/options")
        assert r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key"
        assert r.headers["WWW-Authenticate"] == "ApiKey"
        r = c.get("/v1/options", headers={"X-API-Key": "wrong-key-0000000000"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
        assert c.get("/v1/options", headers={"X-API-Key": KEYS["ci"]}).status_code == 200
        for open_path in ("/health", "/ready", "/docs", "/openapi.json"):
            assert c.get(open_path).status_code == 200, open_path


def test_swagger_has_authorize_button(make_client):
    with make_client() as c:
        spec = c.get("/openapi.json").json()
    assert spec["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey", "in": "header", "name": "X-API-Key",
        "description": "Client key from API_KEYS (not needed when the server has no keys configured)"}


def test_rate_limit_returns_429_with_retry_after_per_client(make_client):
    with make_client(per_minute=3) as c:
        web, ci = {"X-API-Key": KEYS["webapp"]}, {"X-API-Key": KEYS["ci"]}
        assert [c.get("/v1/options", headers=web).status_code for _ in range(3)] == [200, 200, 200]
        r = c.get("/v1/options", headers=web)
        assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
        assert int(r.headers["Retry-After"]) >= 1
        assert c.get("/v1/options", headers=ci).status_code == 200  # other clients are unaffected
        assert c.get("/health").status_code == 200  # health is never limited


def test_answers_have_their_own_stricter_limit(make_client):
    with make_client(answers_per_minute=1) as c:
        h = {"X-API-Key": KEYS["webapp"]}
        assert c.post("/v1/answer", json={"question": "GDP?"}, headers=h).status_code == 200
        r = c.post("/v1/answer", json={"question": "GDP?"}, headers=h)
        assert r.status_code == 429 and r.json()["error"]["details"]["bucket"] == "answers"
        assert c.post("/v1/search", json={"question": "GDP?"}, headers=h).status_code == 200  # search still works


def test_open_mode_without_keys_limits_by_ip(make_client):
    with make_client(keys=None, per_minute=2) as c:
        assert [c.get("/v1/options").status_code for _ in range(3)] == [200, 200, 429]


def test_malformed_or_weak_keys_fail_at_startup(make_client):
    for bad in ("not json", '["a list"]', '{"short": "abc"}'):
        app_client = make_client(keys=None)
        import os

        os.environ["API_KEYS"] = bad
        from src.config import get_settings

        get_settings.cache_clear()
        with pytest.raises(ValueError):
            with app_client:
                pass


def test_identify_client_and_limiter_units():
    assert identify_client(KEYS["ci"], KEYS) == "ci"
    assert identify_client("nope", KEYS) is None and identify_client(None, KEYS) is None
    limiter = RateLimiter()
    limiter.check("a", "all", 2)
    limiter.check("a", "all", 2)
    with pytest.raises(ApiError) as e:
        limiter.check("a", "all", 2)
    assert e.value.status == 429
    limiter.check("a", "all", 0)  # 0 = unlimited


def test_keys_are_never_logged(make_client, caplog):
    import logging

    with make_client() as c, caplog.at_level(logging.INFO):
        c.get("/v1/options", headers={"X-API-Key": KEYS["webapp"]})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "client=webapp" in text and KEYS["webapp"] not in text


def test_load_api_keys_empty_means_disabled(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("API_KEYS", "")
    get_settings.cache_clear()
    try:
        assert load_api_keys() == {}
    finally:
        get_settings.cache_clear()
