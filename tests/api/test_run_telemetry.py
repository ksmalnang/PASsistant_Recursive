"""Run-level telemetry records emitted by the streaming chat path."""

import logging
from collections.abc import AsyncIterator

import httpx
import pytest

from src.services.contracts import AgentStreamUpdate
from tests.doubles import ChatBackend, RecordingChatAgent
from tests.factories import make_agent_state, make_chat_payload

ANSWER = "Syarat kelulusan mencakup 144 SKS dengan IPK minimal 2.75."


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
    """Script the final workflow state replayed by the streaming agent double."""
    chat_backend.final_state_factory = lambda thread_id: make_agent_state(
        response=ANSWER,
        session_id=thread_id,
    )
    return chat_backend


def _records(caplog: pytest.LogCaptureFixture, event: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if getattr(record, "event", None) == event]


async def test_run_telemetry_emits_stage_and_summary_records(
    api_client: httpx.AsyncClient,
    scripted_backend: ChatBackend,
    valid_academic_message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A streamed turn emits one record per stage plus one summary for the run."""
    with caplog.at_level(logging.INFO):
        response = await api_client.post(
            "/chat/stream",
            data=make_chat_payload(valid_academic_message),
        )

    assert response.status_code == 200

    stages = _records(caplog, "stage")
    assert [record.stage for record in stages] == ["retrieval", "response_generation"]
    assert all(record.duration_ms >= 0 for record in stages)
    assert all(record.run_id and record.session_id for record in stages)

    summaries = _records(caplog, "run_summary")
    assert len(summaries) == 1

    summary = summaries[0]
    assert summary.outcome == "completed"
    assert summary.total_ms >= 0
    assert set(summary.stages) == {"retrieval", "response_generation"}
    assert summary.tokens == {}
    assert summary.run_id == stages[0].run_id
    assert summary.session_id == stages[0].session_id


async def test_run_telemetry_marks_failed_runs(
    api_client: httpx.AsyncClient,
    chat_backend: ChatBackend,
    valid_academic_message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run that fails still summarises, marked as failed."""
    chat_backend.agent_factory = lambda thread_id: _FailingStreamAgent(session_id=thread_id)

    with caplog.at_level(logging.INFO):
        response = await api_client.post(
            "/chat/stream",
            data=make_chat_payload(valid_academic_message),
        )

    assert response.status_code == 200
    assert [record.stage for record in _records(caplog, "stage")] == ["retrieval"]

    summaries = _records(caplog, "run_summary")
    assert len(summaries) == 1
    assert summaries[0].outcome == "failed"
