"""Small parsing helpers shared by streaming tests."""

from __future__ import annotations

import json
from typing import Any


def parse_sse_frames(body: str) -> list[dict[str, Any]]:
    """Parse an SSE response body into structured frames.

    Every frame exposes:
    - ``id``: transport event identifier (``Last-Event-ID`` cursor).
    - ``event``: event type such as ``run.started`` or ``message.delta``.
    - ``payload``: full decoded ``ChatStreamEvent`` (envelope plus data).
    - ``data``: the event's ``data`` object (``text``, ``response``, ...).
    """
    frames: list[dict[str, Any]] = []
    for raw_frame in body.split("\n\n"):
        lines = [line for line in raw_frame.splitlines() if line.strip()]
        if not lines:
            continue

        frame: dict[str, Any] = {"id": None, "event": None, "payload": None, "data": None}
        data_chunks: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                frame["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("id:"):
                frame["id"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_chunks.append(line.split(":", 1)[1].strip())
        if data_chunks:
            payload = json.loads("\n".join(data_chunks))
            frame["payload"] = payload
            frame["data"] = payload.get("data") or {}
        frames.append(frame)
    return frames
