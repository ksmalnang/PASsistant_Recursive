"""Release gates that re-run the mapped test subsets (RT-001..RT-006).

Regression in this plan means re-executing the existing unit, API, and system
tests that protect a change area - not duplicating their assertions. Each gate
spawns the same pytest command a release engineer would run, so a red gate names
the subset it protects and shows that subset's own failure output.

Run only the gates with::

    uv run pytest -m regression

System tests inside the gate subsets skip unless ``PASSISTANT_SYSTEM_TESTS=1``
is exported, so gates stay offline-safe and fast by default.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBPROCESS_TIMEOUT_SECONDS = 900
SUMMARY_TAIL_LINES = 15


def _run_subset(subset: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Re-run one subset with pytest in a fresh interpreter."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *subset,
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def _tail(text: str, lines: int = SUMMARY_TAIL_LINES) -> str:
    """Return the last lines of captured pytest output."""
    captured = [line for line in text.splitlines() if line.strip()]
    return "\n".join(captured[-lines:])


def _assert_subset_is_green(gate: str, subset: Sequence[str]) -> None:
    """Fail the gate when its subset reports failures, errors, or no tests."""
    result = _run_subset(subset)
    summary = f"{_tail(result.stdout)}\n--- stderr ---\n{_tail(result.stderr, 5)}"

    assert result.returncode == 0, f"{gate} subset is red:\n{summary}"
    assert "no tests ran" not in result.stdout, f"{gate} subset selected no tests:\n{summary}"


def test_regression_001_core_chat_gate() -> None:
    """RT-001: chat, routing, validation, and record-flow tests stay green."""
    _assert_subset_is_green(
        "RT-001",
        [
            "tests/unit/guardrails/test_input_guard.py",
            "tests/unit/api/test_api_models.py",
            "tests/unit/services",
            "tests/api/chat",
            "tests/system",
        ],
    )


def test_regression_002_guardrail_gate() -> None:
    """RT-002: injection rejection and PII/system-prompt masking stay unchanged."""
    _assert_subset_is_green(
        "RT-002",
        [
            "tests/unit/guardrails",
            "tests/api/chat/test_chat_api.py",
            "tests/system",
        ],
    )


def test_regression_003_knowledge_base_gate() -> None:
    """RT-003: valid documents stay searchable and invalid ones stay rejected."""
    _assert_subset_is_green(
        "RT-003",
        [
            "tests/api/documents",
            "tests/system",
        ],
    )


def test_regression_004_streaming_gate() -> None:
    """RT-004: event schema, sequencing, completion, and resume stay compatible."""
    _assert_subset_is_green(
        "RT-004",
        [
            "tests/unit/api/test_api_models.py",
            "tests/api/chat/test_chat_stream_api.py",
            "tests/system",
        ],
    )


def test_regression_005_infrastructure_gate() -> None:
    """RT-005: dependency health reporting and core API behavior stay correct."""
    _assert_subset_is_green(
        "RT-005",
        [
            "tests/api/health",
            "tests/smoke",
        ],
    )


def test_regression_006_rate_limit_and_concurrency_gate() -> None:
    """RT-006: per-IP limits, window reset, and concurrency stay correct."""
    _assert_subset_is_green(
        "RT-006",
        [
            "tests/unit/guardrails/test_rate_limit.py",
            "tests/api/ratelimit",
            "tests/system",
        ],
    )
