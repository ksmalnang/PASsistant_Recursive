"""Unit tests for the in-memory rate limiter (UT-011)."""

from src.config import get_settings
from src.guardrails.rate_limit import InMemoryRateLimiter
from tests.doubles import FakeClock


def test_unit_011_rate_limiter_enforces_limit_and_resets_window(fake_clock: FakeClock) -> None:
    """Excess requests are rejected per client and recover after the rolling window."""
    limit = get_settings().RATE_LIMIT_PER_MINUTE
    window_seconds = 60
    assert limit > 0

    limiter = InMemoryRateLimiter(limit=limit, window_seconds=window_seconds)
    client_a = "198.51.100.7"
    client_b = "203.0.113.9"

    for _ in range(limit):
        assert limiter.allow(client_a) is True
    assert limiter.allow(client_a) is False

    # A different client has an independent counter.
    assert limiter.allow(client_b) is True

    # Just before the window expires the excess request is still rejected.
    fake_clock.advance(window_seconds - 0.001)
    assert limiter.allow(client_a) is False

    # At the window edge the oldest events expire and the client is allowed again.
    fake_clock.advance(0.001)
    assert limiter.allow(client_a) is True
