"""Progress logging for sequential work and long blocking steps."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 20.0


class Progress:
    """Log `[i/n]` lines with elapsed time and an ETA estimate."""

    def __init__(self, total: int, label: str):
        self._total = max(total, 1)
        self._label = label
        self._completed = 0
        self._started_at = time.perf_counter()

    def advance(self, item: str) -> None:
        self._completed += 1
        elapsed = time.perf_counter() - self._started_at
        remaining = self._total - self._completed
        eta = elapsed / self._completed * remaining
        logger.info(
            "[%s] %d/%d %s | elapsed %s | eta %s",
            self._label,
            self._completed,
            self._total,
            item,
            _format_seconds(elapsed),
            _format_seconds(eta),
        )


class Heartbeat:
    """Log elapsed time periodically while one blocking step runs.

    RAGAS prints its own tqdm bars for testset generation, but those need a real
    terminal and can collapse to a bare number in a captured console. This emits a
    plain log line so the step is still visibly alive.
    """

    def __init__(self, label: str, interval_seconds: float = HEARTBEAT_SECONDS):
        self._label = label
        self._interval = interval_seconds
        self._started_at = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="eval-heartbeat", daemon=True)

    def __enter__(self) -> Heartbeat:
        logger.info("%s started", self._label)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        logger.info(
            "%s finished in %s",
            self._label,
            _format_seconds(time.perf_counter() - self._started_at),
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            logger.info(
                "%s still running (elapsed %s)",
                self._label,
                _format_seconds(time.perf_counter() - self._started_at),
            )


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(max(seconds, 0)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
