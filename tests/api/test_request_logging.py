"""Request logging tests: one structured record per request, correlated when it fails."""

import logging

import httpx
import pytest

from src.api.models import DependencyHealthResponse


async def test_api_logging_01_request_record_carries_correlation_and_duration(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A completed request emits one record with status, duration and request id."""
    healthy = DependencyHealthResponse(status="healthy", detail="ok")
    monkeypatch.setattr("src.api.routes.health._check_redis_health", lambda: healthy)
    monkeypatch.setattr("src.api.routes.health._check_qdrant_health", lambda: healthy)

    with caplog.at_level(logging.INFO):
        response = await api_client.get("/health", headers={"X-Request-ID": "req-log-1"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-log-1"

    records = [r for r in caplog.records if getattr(r, "event", None) == "request"]
    assert len(records) == 1

    record = records[0]
    assert record.request_id == "req-log-1"
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status_code == 200
    assert record.duration_ms >= 0


async def test_api_logging_02_unhandled_error_keeps_request_id(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unhandled error is logged with a traceback while correlation is still set."""

    def boom() -> DependencyHealthResponse:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.api.routes.health._check_redis_health", boom)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        await api_client.get("/health", headers={"X-Request-ID": "req-log-2"})

    records = [r for r in caplog.records if getattr(r, "event", None) == "request_failed"]
    assert len(records) == 1

    record = records[0]
    assert record.request_id == "req-log-2"
    assert record.path == "/health"
    assert record.duration_ms >= 0
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
