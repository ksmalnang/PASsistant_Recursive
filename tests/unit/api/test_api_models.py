"""Unit tests for API stream event models (UT-009)."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from src.api.models import ChatStreamEvent


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
