"""Unit tests for API request/event models (UT-008, UT-009)."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from src.api.models import ChatRequest, ChatStreamEvent


def test_unit_008_chat_request_validates_message_length() -> None:
    """Valid messages are accepted; empty and oversized messages raise validation errors."""
    minimum = ChatRequest(message="x")
    maximum = ChatRequest(message="x" * 4000)
    assert minimum.message == "x"
    assert len(maximum.message) == 4000

    valid = ChatRequest(message="Apa syarat kelulusan program sarjana?")
    assert valid.thread_id is None

    with pytest.raises(ValidationError) as empty_error:
        ChatRequest(message="")
    assert empty_error.value.errors()[0]["loc"] == ("message",)

    with pytest.raises(ValidationError) as oversized_error:
        ChatRequest(message="x" * 4001)
    assert oversized_error.value.errors()[0]["loc"] == ("message",)


def test_unit_009_chat_stream_event_validates_event_type() -> None:
    """Supported stream event types pass; unknown types raise a validation error."""
    timestamp = datetime.now(UTC)

    for event_type in ("run.started", "message.delta", "run.completed"):
        event = ChatStreamEvent(
            event_id="run-1:1",
            event_type=event_type,
            thread_id="thread-1",
            run_id="run-1",
            timestamp=timestamp,
            sequence=1,
            data={"step": event_type},
        )
        dumped = event.model_dump(mode="json")
        assert dumped["event_type"] == event_type
        assert dumped["sequence"] == 1
        assert isinstance(dumped["timestamp"], str)

    invalid_payload: dict[str, Any] = {
        "event_id": "run-1:2",
        "event_type": "run.updated",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "timestamp": timestamp,
        "sequence": 2,
    }
    with pytest.raises(ValidationError) as invalid_error:
        ChatStreamEvent(**invalid_payload)
    assert invalid_error.value.errors()[0]["loc"] == ("event_type",)
