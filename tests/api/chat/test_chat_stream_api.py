"""HTTP-layer tests for ``POST /chat/stream`` (IT-009).

The SSE framing, replay buffer, and event sequencing run as production code; the
agent double decides what the run emits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from src.services.contracts import AgentStreamUpdate
from tests.doubles import ChatBackend, RecordingChatAgent
from tests.factories import make_agent_state, make_chat_payload
from tests.helpers import parse_sse_frames

STREAM_ANSWER = "Syarat kelulusan mencakup 144 SKS dengan IPK minimal 2.75."


class _FailingStreamAgent(RecordingChatAgent):
    """Agent double whose stream aborts after the first status update."""

    async def stream_chat(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> AsyncIterator[AgentStreamUpdate]:
        """Emit one status update and then fail, as an upstream outage would."""
        del message, files
        yield AgentStreamUpdate(kind="status", node="retrieval", payload={})
        raise RuntimeError("upstream streaming failure")


@pytest.fixture
def scripted_backend(chat_backend: ChatBackend) -> ChatBackend:
    """Script the final workflow state used by the streaming agent double."""
    chat_backend.final_state_factory = lambda thread_id: make_agent_state(
        response=STREAM_ANSWER,
        session_id=thread_id,
    )
    return chat_backend


async def test_api_09_chat_stream_emits_ordered_sse_events(
    api_client: httpx.AsyncClient,
    scripted_backend: ChatBackend,
    valid_academic_message: str,
) -> None:
    """Streaming emits ordered run/delta events, then a single terminal event."""
    response = await api_client.post(
        "/chat/stream",
        json=make_chat_payload(valid_academic_message),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = parse_sse_frames(response.text)
    events = [frame["event"] for frame in frames]
    assert events[0] == "run.started"
    assert events[-1] == "run.completed"
    assert "message.delta" in events
    assert "run.failed" not in events
    assert all(frame["id"] for frame in frames)

    sequences = [frame["payload"]["sequence"] for frame in frames]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)

    streamed_text = "".join(
        frame["data"]["text"] for frame in frames if frame["event"] == "message.delta"
    )
    assert streamed_text == STREAM_ANSWER
    assert frames[-1]["data"]["response"] == STREAM_ANSWER
    assert {frame["payload"]["thread_id"] for frame in frames} == {
        scripted_backend.last_agent.session_id
    }

    scripted_backend.agent_factory = lambda thread_id: _FailingStreamAgent(
        session_id=thread_id or "failing-stream-thread"
    )
    failed_response = await api_client.post(
        "/chat/stream",
        json=make_chat_payload(valid_academic_message),
    )

    assert failed_response.status_code == 200
    failed_frames = parse_sse_frames(failed_response.text)
    failed_events = [frame["event"] for frame in failed_frames]
    assert failed_events[-1] == "run.failed"
    assert "run.completed" not in failed_events
    assert failed_frames[-1]["data"]["message"] == "upstream streaming failure"
