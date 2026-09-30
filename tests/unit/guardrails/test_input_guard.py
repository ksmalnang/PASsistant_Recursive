"""Unit tests for the input guard (UT-001..UT-004, UT-007)."""

import pytest

from src.guardrails.input_guard import InputGuard


@pytest.fixture
def guard() -> InputGuard:
    """Provide a fresh input guard."""
    return InputGuard()


def test_unit_001_input_guard_accepts_valid_message(guard: InputGuard) -> None:
    """A benign academic question must pass and keep the trimmed text."""
    message = "What are the graduation requirements?"

    result = guard.validate(message)

    assert result.safe is True
    assert result.reason is None
    assert result.sanitized == message


def test_unit_002_input_guard_rejects_empty_message(guard: InputGuard) -> None:
    """Empty and whitespace-only messages must be rejected as empty."""
    for message in ("", "   ", "\n\t  \n"):
        result = guard.validate(message)

        assert result.safe is False
        assert result.reason == "empty_message"


def test_unit_003_input_guard_rejects_too_long_message(guard: InputGuard) -> None:
    """Messages above the length limit are rejected; the exact limit is allowed."""
    boundary = guard.validate("a" * InputGuard.MAX_MESSAGE_LENGTH)
    assert boundary.safe is True

    oversized = guard.validate("a" * (InputGuard.MAX_MESSAGE_LENGTH + 1))

    assert oversized.safe is False
    assert oversized.reason == "message_too_long"
    assert oversized.sanitized is None


def test_unit_004_input_guard_rejects_prompt_injection(
    guard: InputGuard, prompt_injection_message: str
) -> None:
    """A classic override attempt must be rejected without echoing a sanitized value."""
    result = guard.validate(prompt_injection_message)

    assert result.safe is False
    assert result.reason == "prompt_injection"
    assert result.sanitized is None


@pytest.mark.parametrize(
    "message",
    [
        "Kapan sidang? 😭",
        "How to daftar wisuda?",
        "متى موعد حفل التخرج؟",
        "毕业典礼什么时候进行？",
    ],
)
def test_unit_007_input_guard_preserves_unicode_messages(guard: InputGuard, message: str) -> None:
    """Valid Unicode and mixed-language input must be accepted and preserved."""
    result = guard.validate(message)

    assert result.safe is True
    assert result.reason is None
    assert result.sanitized == message.strip()
