"""Asynchronous bulk reindex jobs.

Job records live in Redis under reindex_job:{job_id} (with an in-process mirror
so a local run without Redis still works). The task itself runs in a single
process; a multi-worker deployment would need a distributed queue. A queued or
running job whose heartbeat is older than the staleness timeout is reported as
failed/interrupted, which covers the owning process having been restarted.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from functools import partial
from typing import Any

from src.clients.redis import RedisCache, get_cache
from src.config import get_settings
from src.services.reindex import (
    REINDEXED,
    SKIPPED_NO_CACHE,
    OCRReindexService,
    ReindexResult,
    get_reindex_service,
)

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "reindex_job:"
ACTIVE_JOB_KEY = "reindex_job:active"
STALE_AFTER_SECONDS = 300
TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobAlreadyRunningError(RuntimeError):
    """Raised when a second bulk job is requested while one is active."""

    def __init__(self, active_job_id: str) -> None:
        self.active_job_id = active_job_id
        super().__init__(f"reindex job {active_job_id} is already active")


class ReindexJobStore:
    """Persistence for bulk job records, Redis-backed with an in-process mirror."""

    def __init__(
        self,
        cache: RedisCache | None = None,
        *,
        ttl_seconds: int | None = None,
        stale_after_seconds: int = STALE_AFTER_SECONDS,
    ) -> None:
        self._cache = cache if cache is not None else get_cache()
        settings = get_settings()
        self._ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else settings.REINDEX_JOB_TTL_SECONDS
        )
        self._stale_after_seconds = stale_after_seconds
        self._memory: dict[str, dict[str, Any]] = {}
        self._memory_active: str | None = None

    def save(self, job: dict[str, Any]) -> None:
        self._memory[job["job_id"]] = dict(job)
        self._cache.set_json(self._key(job["job_id"]), job, ttl_seconds=self._ttl_seconds)

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._read(job_id)
        if job is None:
            return None
        if job.get("status") in {"queued", "running"} and self._is_stale(job):
            job["status"] = "failed"
            job["error"] = "interrupted"
            job["finished_at"] = _now()
            job["current_document"] = None
            self.save(job)
        return job

    def delete(self, job_id: str) -> None:
        self._memory.pop(job_id, None)
        self._cache.delete(self._key(job_id))

    def active_job_id(self) -> str | None:
        """Return the active job id, clearing one that already finished."""
        active = self._memory_active
        if active is None:
            payload = self._cache.get_json(ACTIVE_JOB_KEY)
            active = payload.get("job_id") if isinstance(payload, dict) else None
        if active is None:
            return None
        job = self.get(active)
        if job is None or job.get("status") not in {"queued", "running"}:
            self.clear_active(active)
            return None
        return active

    def set_active(self, job_id: str) -> None:
        self._memory_active = job_id
        self._cache.set_json(
            ACTIVE_JOB_KEY,
            {"job_id": job_id},
            ttl_seconds=self._ttl_seconds,
        )

    def clear_active(self, job_id: str) -> None:
        if self._memory_active == job_id:
            self._memory_active = None
        payload = self._cache.get_json(ACTIVE_JOB_KEY)
        if isinstance(payload, dict) and payload.get("job_id") == job_id:
            self._cache.delete(ACTIVE_JOB_KEY)

    def request_cancel(self, job_id: str) -> bool:
        """Flag a queued/running job for cancellation; False if not cancellable."""
        job = self.get(job_id)
        if job is None or job.get("status") in TERMINAL_STATUSES:
            return False
        job["cancel_requested"] = True
        self.save(job)
        return True

    def _key(self, job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}{job_id}"

    def _read(self, job_id: str) -> dict[str, Any] | None:
        if job_id in self._memory:
            return dict(self._memory[job_id])
        payload = self._cache.get_json(self._key(job_id))
        return dict(payload) if isinstance(payload, dict) else None

    def _is_stale(self, job: dict[str, Any]) -> bool:
        heartbeat = job.get("heartbeat_at")
        if not heartbeat:
            return True
        try:
            moment = datetime.fromisoformat(str(heartbeat))
        except ValueError:
            return True
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return (datetime.now(UTC) - moment).total_seconds() > self._stale_after_seconds


class ReindexJobRunner:
    """Start and supervise asynchronous bulk reindex jobs."""

    def __init__(
        self,
        service: OCRReindexService,
        store: ReindexJobStore,
        *,
        instance_id: str | None = None,
    ) -> None:
        self._service = service
        self._store = store
        self._instance_id = instance_id or uuid.uuid4().hex
        self._tasks: set[asyncio.Task[None]] = set()

    def create_job(self, document_ids: list[str], *, rebuild_text: bool) -> dict[str, Any]:
        """Queue a bulk job, or raise when another job is already active."""
        active = self._store.active_job_id()
        if active is not None:
            raise JobAlreadyRunningError(active)

        job: dict[str, Any] = {
            "job_id": uuid.uuid4().hex,
            "status": "queued",
            "rebuild_text": rebuild_text,
            "total": len(document_ids),
            "processed": 0,
            "succeeded": 0,
            "skipped_no_cache": 0,
            "failed": 0,
            "current_document": None,
            "started_at": _now(),
            "finished_at": None,
            "results": [],
            "error": None,
            "instance_id": self._instance_id,
            "heartbeat_at": _now(),
            "cancel_requested": False,
        }
        self._store.save(job)
        self._store.set_active(job["job_id"])
        task = asyncio.create_task(self._run(job["job_id"], list(document_ids), rebuild_text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._store.get(job_id)

    def request_cancel(self, job_id: str) -> bool:
        return self._store.request_cancel(job_id)

    async def _run(
        self,
        job_id: str,
        document_ids: list[str],
        rebuild_text: bool,
    ) -> None:
        job = self._store.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["heartbeat_at"] = _now()
        self._store.save(job)
        try:
            await self._service.reindex_many(
                document_ids,
                rebuild_text=rebuild_text,
                on_result=partial(self._record_result, job_id),
                should_cancel=partial(self._cancelled, job_id),
            )
        except Exception as exc:
            logger.error("Reindex job failed", exc_info=True, extra={"job_id": job_id})
            self._finalize(job_id, status="failed", error=str(exc))
            return
        finally:
            self._store.clear_active(job_id)

        current = self._store.get(job_id) or {}
        if current.get("status") == "running":
            if current.get("cancel_requested"):
                self._finalize(job_id, status="cancelled")
            elif int(current.get("failed", 0)) > 0:
                self._finalize(job_id, status="completed_with_errors")
            else:
                self._finalize(job_id, status="completed")

    def _record_result(self, job_id: str, result: ReindexResult) -> None:
        job = self._store.get(job_id)
        if job is None:
            return
        job["processed"] = int(job.get("processed", 0)) + 1
        job["current_document"] = result.document_id
        job["heartbeat_at"] = _now()
        if result.status == REINDEXED:
            job["succeeded"] = int(job.get("succeeded", 0)) + 1
        elif result.status == SKIPPED_NO_CACHE:
            job["skipped_no_cache"] = int(job.get("skipped_no_cache", 0)) + 1
        else:
            job["failed"] = int(job.get("failed", 0)) + 1
        job.setdefault("results", []).append(result.to_dict())
        self._store.save(job)

    def _cancelled(self, job_id: str) -> bool:
        job = self._store.get(job_id)
        return bool(job and job.get("cancel_requested"))

    def _finalize(self, job_id: str, *, status: str, error: str | None = None) -> None:
        job = self._store.get(job_id)
        if job is None:
            return
        job["status"] = status
        job["finished_at"] = _now()
        job["current_document"] = None
        if error is not None:
            job["error"] = error
        self._store.save(job)


_default_job_runner: ReindexJobRunner | None = None


def get_job_runner() -> ReindexJobRunner:
    """Return the process-wide bulk reindex job runner."""
    global _default_job_runner
    if _default_job_runner is None:
        _default_job_runner = ReindexJobRunner(
            service=get_reindex_service(),
            store=ReindexJobStore(),
        )
    return _default_job_runner
