from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.rate_limit import TokenBucket, TokenBucketLimiter


def test_token_bucket_allows_capacity_then_blocks() -> None:
    bucket = TokenBucket(rate_per_second=10.0, capacity=2)
    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()


def test_token_bucket_disabled_rate_zero() -> None:
    limiter = TokenBucketLimiter(rate_per_min=0)
    for _ in range(100):
        assert limiter.try_acquire("any-client")


def test_token_bucket_limiter_per_client_isolation() -> None:
    limiter = TokenBucketLimiter(rate_per_min=1)
    assert limiter.try_acquire("a")
    assert not limiter.try_acquire("a")
    assert limiter.try_acquire("b")
    assert not limiter.try_acquire("b")


def test_api_health_liveness() -> None:
    from src.api import main as api

    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_query_rejects_overlong_question_without_pipeline() -> None:
    from src.api import main as api

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(api.settings, "api_max_question_chars", 8)
    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.post("/api/query", json={"question": "x" * 200, "history": []})
    monkeypatch.undo()
    assert resp.status_code == 400


def test_api_auth_required_when_configured() -> None:
    from src.api import main as api

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(api.settings, "api_auth_username", "reader")
    monkeypatch.setattr(api.settings, "api_auth_password", "secret")
    client = TestClient(api.app, raise_server_exceptions=False)
    try:
        resp = client.get("/api/evidence/does-not-exist")
        assert resp.status_code == 401
    finally:
        monkeypatch.undo()


def test_api_rate_limit_429(monkeypatch) -> None:
    from src.api import main as api

    monkeypatch.setattr(api, "_limiter", TokenBucketLimiter(rate_per_min=1))

    def fake_get_pipeline():
        raise HTTPException(status_code=503, detail="pipeline unavailable (test)")

    monkeypatch.setattr(api, "get_pipeline", fake_get_pipeline)
    client = TestClient(api.app, raise_server_exceptions=False)
    payload = {"question": "hi", "history": []}
    first = client.post("/api/query", json=payload)
    assert first.status_code == 503
    second = client.post("/api/query", json=payload)
    assert second.status_code == 429


def test_readiness_returns_503_before_index() -> None:
    from src.api import main as api

    def fake_get_pipeline():
        raise HTTPException(status_code=503, detail="no index (test)")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(api, "get_pipeline", fake_get_pipeline)
    monkeypatch.setattr(api, "_pipeline", None)
    client = TestClient(api.app, raise_server_exceptions=False)
    try:
        resp = client.get("/api/ready")
        assert resp.status_code == 503
        assert resp.json()["ready"] is False
    finally:
        monkeypatch.undo()
