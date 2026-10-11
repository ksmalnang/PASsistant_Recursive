"""Unit tests for per-run timing and token accounting."""

from types import SimpleNamespace

import pytest

from src.clients.decision import (
    DecisionResponse,
    JevDecisionClient,
    NoulAnswer,
    NoulQuestion,
    Usage,
)
from src.telemetry import RunTelemetry, current_run, end_run, start_run
from tests.doubles import FakeClock


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Drive telemetry timing from a deterministic clock."""
    fake = FakeClock()
    monkeypatch.setattr("src.telemetry.time", SimpleNamespace(perf_counter=fake.monotonic))
    return fake


def test_telemetry_stage_timings_and_ttft(clock: FakeClock) -> None:
    """Stage boundaries charge elapsed time to the stage that just finished."""
    telemetry = RunTelemetry(run_id="run-1", session_id="sess-1")

    clock.advance(0.5)
    assert telemetry.record_stage("router_node") == 500

    clock.advance(0.25)
    telemetry.mark_first_token()
    telemetry.mark_first_token()

    clock.advance(1.0)
    assert telemetry.record_stage("generate_response") == 1250

    summary = telemetry.summary()

    assert summary["run_id"] == "run-1"
    assert summary["session_id"] == "sess-1"
    assert summary["stages"] == {"router_node": 500, "generate_response": 1250}
    assert summary["ttft_ms"] == 750
    assert summary["total_ms"] == 1750


def test_telemetry_accumulates_repeated_stages_and_tokens(clock: FakeClock) -> None:
    """A retried stage accumulates, and both token sources share one total."""
    telemetry = RunTelemetry(run_id="run-2", session_id="sess-2")

    clock.advance(0.2)
    telemetry.record_stage("handle_error")
    clock.advance(0.3)
    telemetry.record_stage("generate_response")
    clock.advance(0.4)
    telemetry.record_stage("generate_response")

    assert telemetry.stages["generate_response"] == 700

    telemetry.record_tokens("typesafe/jev-1.13", input_tokens=100, output_tokens=20, cost=0.001)
    telemetry.record_tokens("typesafe/jev-1.13", input_tokens=50, output_tokens=10, cost=0.002)
    telemetry.usage_callback = SimpleNamespace(
        usage_metadata={
            "deepseek/flash": {"input_tokens": 300, "output_tokens": 40, "total_tokens": 340}
        }
    )

    tokens = telemetry.summary()["tokens"]

    assert tokens["typesafe/jev-1.13"] == {
        "input": 150,
        "output": 30,
        "total": 180,
        "cost": pytest.approx(0.003),
    }
    assert tokens["deepseek/flash"] == {"input": 300, "output": 40, "total": 340, "cost": 0.0}


def test_telemetry_run_context_is_scoped() -> None:
    """The accumulator only exists inside a run, and outlives its context."""
    assert current_run() is None

    token = start_run("run-3", "sess-3")
    run = current_run()
    assert run is not None
    assert (run.run_id, run.session_id) == ("run-3", "sess-3")

    end_run(token)

    assert current_run() is None
    assert run.summary()["run_id"] == "run-3"


async def test_telemetry_records_decision_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jev token and cost totals reach the run instead of being discarded."""
    client = JevDecisionClient(api_key="test-key")

    async def fake_post(
        payload: dict[str, object], *, session_id: str | None = None
    ) -> DecisionResponse:
        return DecisionResponse(
            model="typesafe/jev-1.13",
            answers={"q": NoulAnswer(noul=0.9)},
            usage=Usage(input_tokens=120, output_tokens=15, cost=0.0004),
        )

    monkeypatch.setattr(client, "_post", fake_post)

    token = start_run("run-4", "sess-4")
    try:
        await client.ask(
            {"q": NoulQuestion(instructions="i", criteria_true="t", criteria_false="f")},
            state="question",
        )
        run = current_run()
        assert run is not None
        summary = run.summary()
    finally:
        end_run(token)
        await client.aclose()

    assert summary["tokens"]["typesafe/jev-1.13"] == {
        "input": 120,
        "output": 15,
        "total": 135,
        "cost": pytest.approx(0.0004),
    }
