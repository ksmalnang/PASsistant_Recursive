"""Minimum production-readiness checks against a live local server (SMK-001..SMK-006).

The server is started with the documented entry point (uvicorn on ``src.api:app``)
and every check travels over real HTTP, so a missing port, broken startup, or
misconfigured route fails the suite before deployment.
"""

from __future__ import annotations

import httpx
import pytest

from src.config import get_settings
from tests.doubles import ChatBackend, FakeClock, FakeDocumentProcessor
from tests.factories import make_agent_state, make_chat_payload, make_upload

pytestmark = pytest.mark.smoke

ANSWER = "Syarat kelulusan mencakup 144 SKS dengan IPK minimal 2.75."
# Mirrors the default rolling window of ``src.guardrails.rate_limit.InMemoryRateLimiter``.
RATE_LIMIT_WINDOW_SECONDS = 60


async def test_smoke_001_application_startup_exposes_api(live_server_url: str) -> None:
    """The process starts, binds a port, and serves the documented API surface."""
    async with httpx.AsyncClient(base_url=live_server_url) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["version"]
    for path in ("/chat", "/chat/stream", "/upload", "/health"):
        assert path in schema["paths"]


async def test_smoke_002_health_reports_dependency_state(live_server_url: str) -> None:
    """Health returns the documented contract with a truthful overall status."""
    async with httpx.AsyncClient(base_url=live_server_url) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["version"]
    assert body["environment"]
    for dependency in ("redis", "qdrant"):
        assert body[dependency]["status"] in {"healthy", "unhealthy", "disabled"}

    expected_status = (
        "healthy"
        if all(body[name]["status"] != "unhealthy" for name in ("redis", "qdrant"))
        else "degraded"
    )
    assert body["status"] == expected_status


async def test_smoke_003_basic_chat_turn(
    live_server_url: str,
    chat_backend: ChatBackend,
    valid_academic_message: str,
) -> None:
    """One valid chat request returns a non-empty answer and the requested thread id."""
    chat_backend.final_state_factory = lambda thread_id: make_agent_state(
        response=ANSWER, session_id=thread_id
    )

    async with httpx.AsyncClient(base_url=live_server_url) as client:
        response = await client.post(
            "/chat",
            json=make_chat_payload(valid_academic_message, thread_id="smoke-thread"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == ANSWER
    assert body["thread_id"] == "smoke-thread"


async def test_smoke_004_input_guard_blocks_injection(
    live_server_url: str,
    chat_backend: ChatBackend,
    prompt_injection_message: str,
) -> None:
    """A known injection message is rejected with 400, never a 500."""
    async with httpx.AsyncClient(base_url=live_server_url) as client:
        response = await client.post(
            "/chat",
            json=make_chat_payload(prompt_injection_message),
        )

    assert response.status_code == 400
    assert "prompt_injection" in response.json()["detail"]
    assert chat_backend.agents == []


async def test_smoke_005_upload_returns_document_status(
    live_server_url: str,
    documents_backend: FakeDocumentProcessor,
    sample_pdf_bytes: bytes,
) -> None:
    """A small valid PDF is accepted and reported with an identifier and status."""
    async with httpx.AsyncClient(base_url=live_server_url, timeout=30.0) as client:
        response = await client.post(
            "/upload",
            files=[make_upload("smoke-policy.pdf", sample_pdf_bytes, "application/pdf")],
        )

    assert response.status_code == 201
    result = response.json()[0]
    assert result["document_id"]
    assert result["filename"] == "smoke-policy.pdf"
    assert result["status"] == "completed"
    assert result["error"] is None


async def test_smoke_006_rate_limit_rejects_and_recovers(
    live_server_url: str,
    chat_backend: ChatBackend,
    fake_clock: FakeClock,
    valid_academic_message: str,
) -> None:
    """Exceeding the configured threshold yields 429; the service recovers afterwards."""
    limit = get_settings().RATE_LIMIT_PER_MINUTE
    assert limit >= 1
    chat_backend.final_state_factory = lambda thread_id: make_agent_state(
        response="ok", session_id=thread_id
    )
    payload = make_chat_payload(valid_academic_message, thread_id="smoke-rate-limit")

    async with httpx.AsyncClient(base_url=live_server_url) as client:
        statuses = [
            (await client.post("/chat", json=payload)).status_code for _ in range(limit + 1)
        ]

        assert statuses.count(200) == limit
        assert statuses[-1] == 429
        assert (await client.get("/health")).status_code == 200

        fake_clock.advance(RATE_LIMIT_WINDOW_SECONDS + 1)

        recovered = await client.post("/chat", json=payload)

    assert recovered.status_code == 200
