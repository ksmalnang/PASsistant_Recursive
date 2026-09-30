"""HTTP-layer tests for per-client rate limiting (IT-012)."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from src.config import get_settings
from tests.doubles import ChatBackend, FakeClock
from tests.factories import make_agent_state, make_chat_payload

# Mirrors the default rolling window of ``src.guardrails.rate_limit.InMemoryRateLimiter``.
RATE_LIMIT_WINDOW_SECONDS = 60


async def test_api_12_rate_limit_rejects_excess_and_recovers_after_window(
    make_api_client: Callable[..., httpx.AsyncClient],
    chat_backend: ChatBackend,
    fake_clock: FakeClock,
    valid_academic_message: str,
) -> None:
    """One IP is throttled at the configured limit, then allowed after the window."""
    limit = get_settings().RATE_LIMIT_PER_MINUTE
    assert limit >= 1

    chat_backend.final_state_factory = lambda thread_id: make_agent_state(
        response="ok", session_id=thread_id
    )
    payload = make_chat_payload(valid_academic_message, thread_id="rate-limit-thread")

    async with make_api_client() as client:
        for _ in range(limit):
            allowed = await client.post("/chat", json=payload)
            assert allowed.status_code == 200

        blocked = await client.post("/chat", json=payload)

        assert blocked.status_code == 429
        assert "Rate limit exceeded" in blocked.json()["detail"]
        assert len(chat_backend.agents) == 1

        async with make_api_client(host="198.51.100.23") as other_client:
            assert (await other_client.post("/chat", json=payload)).status_code == 200
            assert (await other_client.post("/chat", json=payload)).status_code == 200

        fake_clock.advance(RATE_LIMIT_WINDOW_SECONDS + 1)

        recovered = await client.post("/chat", json=payload)

        assert recovered.status_code == 200
