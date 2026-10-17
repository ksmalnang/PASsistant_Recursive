"""HTTP tests for the OCR-cache reindex endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.clients.redis import RedisCache
from src.services.document_locks import DocumentLockRegistry
from src.services.ocr_cache import OCRCache
from src.services.reindex import OCRReindexService
from src.services.reindex_jobs import ReindexJobRunner, ReindexJobStore
from tests.doubles import FakeIndexer
from tests.factories import seed_ocr_cache

ADMIN_KEY = "test-admin-key"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}

REINDEX_ENDPOINTS = [
    ("post", "/documents/doc-1/reindex", {}),
    ("post", "/documents/reindex/by-filename/handbook.pdf", {}),
    ("post", "/documents/reindex-jobs", {"json": {}}),
    ("get", "/documents/reindex-jobs/job-1", {}),
    ("delete", "/documents/reindex-jobs/job-1", {}),
]


@dataclass
class ReindexBackend:
    cache: OCRCache
    indexer: FakeIndexer
    service: OCRReindexService
    runner: ReindexJobRunner
    store: ReindexJobStore
    locks: DocumentLockRegistry


def _disabled_redis() -> RedisCache:
    return RedisCache(redis_url=None, key_prefix="test", default_ttl_seconds=300)


@pytest.fixture
def reindex_backend(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> ReindexBackend:
    from src.api.routes import documents

    cache = OCRCache(tmp_path / "cache")
    indexer = FakeIndexer()
    locks = DocumentLockRegistry()
    service = OCRReindexService(cache=cache, indexer=indexer, locks=locks)
    store = ReindexJobStore(cache=_disabled_redis(), stale_after_seconds=300)
    runner = ReindexJobRunner(service=service, store=store)

    monkeypatch.setattr(documents, "get_reindex_service", lambda: service)
    monkeypatch.setattr(documents, "get_job_runner", lambda: runner)
    monkeypatch.setattr(
        documents,
        "get_settings",
        lambda: SimpleNamespace(ADMIN_API_KEY=ADMIN_KEY),
    )
    return ReindexBackend(
        cache=cache,
        indexer=indexer,
        service=service,
        runner=runner,
        store=store,
        locks=locks,
    )


async def _wait_for_job(
    runner: ReindexJobRunner,
    job_id: str,
    *,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    terminal = {"completed", "completed_with_errors", "failed", "cancelled"}
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        job = runner.get_job(job_id)
        if job is not None and job.get("status") in terminal:
            return job
        assert loop.time() < deadline, f"job did not finish in time: {job}"
        await asyncio.sleep(0)


class BlockingIndexer(FakeIndexer):
    """FakeIndexer that waits for a release event before indexing."""

    def __init__(self, release: asyncio.Event) -> None:
        super().__init__()
        self.release = release

    async def index_document(
        self,
        document: Any,
        *,
        replace_existing: bool = False,
    ) -> Any:
        await self.release.wait()
        return await super().index_document(document, replace_existing=replace_existing)


@pytest.mark.parametrize(("method", "path", "kwargs"), REINDEX_ENDPOINTS)
async def test_reindex_endpoints_reject_missing_key(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
    method: str,
    path: str,
    kwargs: dict[str, Any],
) -> None:
    response = await getattr(api_client, method)(path, **kwargs)

    assert response.status_code == 403


@pytest.mark.parametrize(("method", "path", "kwargs"), REINDEX_ENDPOINTS)
async def test_reindex_endpoints_reject_wrong_key(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
    method: str,
    path: str,
    kwargs: dict[str, Any],
) -> None:
    response = await getattr(api_client, method)(
        path,
        headers={"X-Admin-Key": "wrong"},
        **kwargs,
    )

    assert response.status_code == 403


async def test_reindex_endpoints_forbidden_when_key_unset(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.routes import documents

    monkeypatch.setattr(
        documents,
        "get_settings",
        lambda: SimpleNamespace(ADMIN_API_KEY=None),
    )

    response = await api_client.post(
        "/documents/doc-1/reindex",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 403
    assert "ADMIN_API_KEY" in response.json()["detail"]


async def test_reindex_by_id_succeeds(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1", filename="handbook.pdf")

    response = await api_client.post(
        "/documents/doc-1/reindex",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-1",
        "filename": "handbook.pdf",
        "status": "reindexed",
        "chunks": 1,
    }
    assert reindex_backend.indexer.replace_flags == [True]


async def test_reindex_by_id_missing_cache_returns_404(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    response = await api_client.post(
        "/documents/missing/reindex",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "no OCR cache for missing, re-upload required"


async def test_reindex_by_id_locked_returns_409(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")

    with reindex_backend.locks.hold("doc-1") as acquired:
        assert acquired is True
        response = await api_client.post(
            "/documents/doc-1/reindex",
            headers=ADMIN_HEADERS,
        )

    assert response.status_code == 409


async def test_reindex_indexing_failure_returns_500(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")
    reindex_backend.indexer.fail_for.add("doc-1")

    response = await api_client.post(
        "/documents/doc-1/reindex",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 500
    assert "indexing exploded" in response.json()["detail"]


async def test_reindex_by_filename_returns_all_matches(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1", filename="shared.pdf")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-2", filename="shared.pdf")

    response = await api_client.post(
        "/documents/reindex/by-filename/shared.pdf",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert {item["document_id"] for item in body} == {"doc-1", "doc-2"}
    assert {item["status"] for item in body} == {"reindexed"}


async def test_reindex_by_filename_missing_returns_404(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    response = await api_client.post(
        "/documents/reindex/by-filename/absent.pdf",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 404


async def test_bulk_job_lifecycle_reports_progress(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1", filename="a.pdf")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-2", filename="b.pdf")

    created = await api_client.post(
        "/documents/reindex-jobs",
        json={"document_ids": ["doc-1", "doc-2"]},
        headers=ADMIN_HEADERS,
    )

    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    assert body["total"] == 2
    assert created.headers["location"] == f"/documents/reindex-jobs/{body['job_id']}"

    job = await _wait_for_job(reindex_backend.runner, body["job_id"])
    assert job["status"] == "completed"
    assert job["processed"] == 2
    assert job["succeeded"] == 2
    assert job["failed"] == 0

    status = await api_client.get(
        f"/documents/reindex-jobs/{body['job_id']}",
        headers=ADMIN_HEADERS,
    )
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "completed"
    assert payload["rebuild_text"] is False
    assert {result["document_id"] for result in payload["results"]} == {"doc-1", "doc-2"}


async def test_bulk_job_records_skipped_and_failed_without_stopping(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-bad")
    reindex_backend.indexer.fail_for.add("doc-bad")

    created = await api_client.post(
        "/documents/reindex-jobs",
        json={"document_ids": ["doc-1", "missing", "doc-bad"]},
        headers=ADMIN_HEADERS,
    )
    job = await _wait_for_job(reindex_backend.runner, created.json()["job_id"])

    assert job["status"] == "completed_with_errors"
    assert job["processed"] == 3
    assert job["succeeded"] == 1
    assert job["skipped_no_cache"] == 1
    assert job["failed"] == 1
    by_id = {result["document_id"]: result for result in job["results"]}
    assert by_id["missing"]["status"] == "skipped_no_cache"
    assert by_id["doc-bad"]["status"] == "failed"
    assert by_id["doc-bad"]["error"] == "indexing exploded for doc-bad"


async def test_bulk_job_defaults_to_all_cached(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-2")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-3")

    created = await api_client.post(
        "/documents/reindex-jobs",
        json={},
        headers=ADMIN_HEADERS,
    )

    assert created.status_code == 202
    assert created.json()["total"] == 3
    job = await _wait_for_job(reindex_backend.runner, created.json()["job_id"])
    assert job["status"] == "completed"


async def test_second_bulk_job_conflicts_with_active_job(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.routes import documents

    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-2")
    release = asyncio.Event()
    service = OCRReindexService(
        cache=reindex_backend.cache,
        indexer=BlockingIndexer(release),
        locks=reindex_backend.locks,
    )
    runner = ReindexJobRunner(service=service, store=reindex_backend.store)
    monkeypatch.setattr(documents, "get_reindex_service", lambda: service)
    monkeypatch.setattr(documents, "get_job_runner", lambda: runner)

    first = await api_client.post(
        "/documents/reindex-jobs",
        json={"document_ids": ["doc-1", "doc-2"]},
        headers=ADMIN_HEADERS,
    )
    await asyncio.sleep(0)

    second = await api_client.post(
        "/documents/reindex-jobs",
        json={"document_ids": ["doc-1"]},
        headers=ADMIN_HEADERS,
    )

    assert second.status_code == 409
    assert first.json()["job_id"] in second.json()["detail"]

    release.set()
    job = await _wait_for_job(runner, first.json()["job_id"])
    assert job["status"] == "completed"


async def test_unknown_job_returns_404(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    read = await api_client.get(
        "/documents/reindex-jobs/nope",
        headers=ADMIN_HEADERS,
    )
    deleted = await api_client.delete(
        "/documents/reindex-jobs/nope",
        headers=ADMIN_HEADERS,
    )

    assert read.status_code == 404
    assert deleted.status_code == 404


async def test_stale_running_job_is_reported_interrupted(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    reindex_backend.store.save(
        {
            "job_id": "stale-1",
            "status": "running",
            "rebuild_text": False,
            "total": 1,
            "processed": 0,
            "succeeded": 0,
            "skipped_no_cache": 0,
            "failed": 0,
            "current_document": "doc-1",
            "started_at": "2020-01-01T00:00:00+00:00",
            "finished_at": None,
            "results": [],
            "error": None,
            "instance_id": "dead-process",
            "heartbeat_at": "2020-01-01T00:00:00+00:00",
            "cancel_requested": False,
        }
    )

    response = await api_client.get(
        "/documents/reindex-jobs/stale-1",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] == "interrupted"


async def test_cancel_running_job_stops_after_current_document(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.routes import documents

    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-2")
    release = asyncio.Event()
    service = OCRReindexService(
        cache=reindex_backend.cache,
        indexer=BlockingIndexer(release),
        locks=reindex_backend.locks,
    )
    runner = ReindexJobRunner(service=service, store=reindex_backend.store)
    monkeypatch.setattr(documents, "get_reindex_service", lambda: service)
    monkeypatch.setattr(documents, "get_job_runner", lambda: runner)

    created = await api_client.post(
        "/documents/reindex-jobs",
        json={"document_ids": ["doc-1", "doc-2"]},
        headers=ADMIN_HEADERS,
    )
    job_id = created.json()["job_id"]
    await asyncio.sleep(0)

    cancelled = await api_client.delete(
        f"/documents/reindex-jobs/{job_id}",
        headers=ADMIN_HEADERS,
    )
    assert cancelled.status_code == 200

    release.set()
    job = await _wait_for_job(runner, job_id)
    assert job["status"] == "cancelled"
    assert job["processed"] == 1


async def test_cancel_finished_job_returns_409(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")
    created = await api_client.post(
        "/documents/reindex-jobs",
        json={"document_ids": ["doc-1"]},
        headers=ADMIN_HEADERS,
    )
    job_id = created.json()["job_id"]
    await _wait_for_job(reindex_backend.runner, job_id)

    response = await api_client.delete(
        f"/documents/reindex-jobs/{job_id}",
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 409


async def test_bulk_job_records_locked_document_and_continues(
    api_client: httpx.AsyncClient,
    reindex_backend: ReindexBackend,
) -> None:
    seed_ocr_cache(reindex_backend.cache, document_id="doc-1")
    seed_ocr_cache(reindex_backend.cache, document_id="doc-2")

    with reindex_backend.locks.hold("doc-2") as acquired:
        assert acquired is True
        created = await api_client.post(
            "/documents/reindex-jobs",
            json={"document_ids": ["doc-1", "doc-2"]},
            headers=ADMIN_HEADERS,
        )
        job = await _wait_for_job(reindex_backend.runner, created.json()["job_id"])

    assert job["status"] == "completed_with_errors"
    by_id = {result["document_id"]: result for result in job["results"]}
    assert by_id["doc-1"]["status"] == "reindexed"
    assert by_id["doc-2"]["status"] == "failed"
    assert by_id["doc-2"]["error"] == "locked"
