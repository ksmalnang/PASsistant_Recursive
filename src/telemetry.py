"""Per-run timing and token accounting for chat turns.

The accumulator lives in a ContextVar rather than on the process-wide clients,
which are shared by every session.
"""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

_run: ContextVar[RunTelemetry | None] = ContextVar("run_telemetry", default=None)


@dataclass
class RunTelemetry:
    """Timing and token totals accumulated while one chat run executes."""

    run_id: str
    session_id: str
    usage_callback: Any = None
    """LangChain callback collecting LLM usage for this run, if one was attached."""
    started_at: float = field(default_factory=lambda: time.perf_counter())
    first_token_at: float | None = None
    stages: dict[str, int] = field(default_factory=dict)
    tokens: dict[str, dict[str, float]] = field(default_factory=dict)
    _stage_started_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._stage_started_at = self.started_at

    def mark_first_token(self) -> None:
        """Record when the first answer token was ready. Idempotent."""
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def record_stage(self, stage: str) -> int:
        """Charge the time since the previous stage boundary to ``stage``.

        Stages can run more than once per run (a retried node), so durations
        accumulate under the stage name.
        """
        now = time.perf_counter()
        elapsed_ms = round((now - self._stage_started_at) * 1000)
        self._stage_started_at = now
        self.stages[stage] = self.stages.get(stage, 0) + elapsed_ms
        return elapsed_ms

    def record_tokens(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Add one call's token counts, and its cost when the provider reports one."""
        totals = self.tokens.setdefault(model, {"input": 0, "output": 0, "total": 0, "cost": 0.0})
        totals["input"] += input_tokens
        totals["output"] += output_tokens
        totals["total"] += input_tokens + output_tokens
        totals["cost"] += cost

    def summary(self) -> dict[str, Any]:
        """Fold in the LLM callback and return the run-summary fields.

        Call once per run: merging usage twice would double-count it.
        """
        if self.usage_callback is not None:
            for model, usage in (self.usage_callback.usage_metadata or {}).items():
                self.record_tokens(
                    model,
                    input_tokens=int(usage.get("input_tokens", 0) or 0),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                )

        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "total_ms": round((time.perf_counter() - self.started_at) * 1000),
            "stages": dict(self.stages),
            "tokens": {model: dict(totals) for model, totals in self.tokens.items()},
        }
        if self.first_token_at is not None:
            summary["ttft_ms"] = round((self.first_token_at - self.started_at) * 1000)
        return summary


def start_run(run_id: str, session_id: str) -> Token[RunTelemetry | None]:
    """Install a fresh accumulator for the current run context."""
    return _run.set(RunTelemetry(run_id=run_id, session_id=session_id))


def end_run(token: Token[RunTelemetry | None]) -> None:
    """Restore the run context that was active before ``start_run``."""
    _run.reset(token)


def current_run() -> RunTelemetry | None:
    """Return the accumulator for the current run, or None outside a run."""
    return _run.get()
