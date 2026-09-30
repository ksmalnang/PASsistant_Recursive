"""HTTP-layer tests for ``GET /health`` (IT-011)."""

from __future__ import annotations

import httpx
import pytest

from src.api.models import DependencyHealthResponse
from src.config import get_settings


def _dependency(status: str, detail: str) -> DependencyHealthResponse:
    """Build a dependency health result for the patched checker."""
    return DependencyHealthResponse(status=status, detail=detail)


async def test_api_11_health_reports_dependency_status(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency states are reported, and any unhealthy dependency degrades status."""
    monkeypatch.setattr(
        "src.api.routes.health._check_redis_health",
        lambda: _dependency("healthy", "Redis responded to ping."),
    )
    monkeypatch.setattr(
        "src.api.routes.health._check_qdrant_health",
        lambda: _dependency("healthy", "Qdrant responded to collection listing."),
    )

    healthy = await api_client.get("/health")

    assert healthy.status_code == 200
    body = healthy.json()
    assert body["status"] == "healthy"
    assert body["version"]
    assert body["environment"] == get_settings().APP_ENV
    assert body["redis"]["status"] == "healthy"
    assert body["qdrant"]["status"] == "healthy"

    monkeypatch.setattr(
        "src.api.routes.health._check_qdrant_health",
        lambda: _dependency("unhealthy", "connection refused"),
    )

    degraded = await api_client.get("/health")

    assert degraded.status_code == 200
    degraded_body = degraded.json()
    assert degraded_body["status"] == "degraded"
    assert degraded_body["qdrant"] == {"status": "unhealthy", "detail": "connection refused"}
    assert degraded_body["redis"]["status"] == "healthy"

    monkeypatch.setattr(
        "src.api.routes.health._check_redis_health",
        lambda: _dependency("disabled", "REDIS_URL is not configured."),
    )
    monkeypatch.setattr(
        "src.api.routes.health._check_qdrant_health",
        lambda: _dependency("healthy", "Qdrant responded to collection listing."),
    )

    disabled = await api_client.get("/health")

    assert disabled.status_code == 200
    assert disabled.json()["status"] == "healthy"
