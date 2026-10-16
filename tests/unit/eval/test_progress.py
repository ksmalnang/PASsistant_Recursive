"""Unit tests for the progress logging helpers."""

from __future__ import annotations

import logging
import time

from src.eval.progress import Heartbeat, Progress


def test_progress_logs_count_and_eta(caplog) -> None:
    with caplog.at_level(logging.INFO):
        progress = Progress(2, "pipeline")
        progress.advance("q_001")
        progress.advance("q_002")

    messages = [record.getMessage() for record in caplog.records]
    assert "[pipeline] 1/2 q_001" in messages[0]
    assert "eta" in messages[0]
    assert "[pipeline] 2/2 q_002" in messages[1]


def test_heartbeat_logs_start_progress_and_finish(caplog) -> None:
    with caplog.at_level(logging.INFO), Heartbeat("generation", interval_seconds=0.01):
        time.sleep(0.03)

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "generation started"
    assert any("still running (elapsed" in message for message in messages)
    assert messages[-1].startswith("generation finished in")
