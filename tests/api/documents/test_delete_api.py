"""HTTP tests for DELETE /documents/by-filename with cache and file cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.services.document_deletion import DocumentDeletionService
from src.services.document_locks import DocumentLockRegistry
from src.services.ocr_cache import OCRCache
from src.utils.tools.document import DocumentTools
from tests.factories import seed_ocr_cache


class _FakeDeletionVectorStore:
    def __init__(self) -> None:
        self.document_ids: set[str] = set()
        self.deleted: list[str] = []

    def find_document_ids_by_filename(self, filename: str) -> set[str]:
        return set(self.document_ids)

    def delete_document_chunks(self, document_id: str) -> None:
        self.deleted.append(document_id)


class _RecordingCache:
    def __init__(self) -> None:
        self.prefixes: list[str] = []

    def delete_prefix(self, prefix: str) -> None:
        self.prefixes.append(prefix)

    def add_to_set(self, *args: Any, **kwargs: Any) -> None:
        pass


@dataclass
class DeleteBackend:
    vector_store: _FakeDeletionVectorStore
    cache: OCRCache
    raw_dir: Path
    recording: _RecordingCache
    locks: DocumentLockRegistry


@pytest.fixture
def delete_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DeleteBackend:
    from src.api.routes import documents

    cache = OCRCache(tmp_path / "cache")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    document_tools = DocumentTools()
    document_tools.raw_dir = raw_dir
    vector_store = _FakeDeletionVectorStore()
    recording = _RecordingCache()
    locks = DocumentLockRegistry()
    service = DocumentDeletionService(
        vector_store=vector_store,
        ocr_cache=cache,
        document_tools=document_tools,
        cache=recording,
        locks=locks,
    )
    monkeypatch.setattr(documents, "get_document_deletion_service", lambda: service)
    return DeleteBackend(
        vector_store=vector_store,
        cache=cache,
        raw_dir=raw_dir,
        recording=recording,
        locks=locks,
    )


async def test_delete_removes_chunks_cache_and_raw_file(
    api_client: httpx.AsyncClient,
    delete_backend: DeleteBackend,
) -> None:
    seed_ocr_cache(delete_backend.cache, document_id="doc-1", filename="handbook.pdf")
    raw_file = delete_backend.raw_dir / "doc-1.pdf"
    raw_file.write_bytes(b"%PDF raw")
    delete_backend.vector_store.document_ids = {"doc-1"}

    response = await api_client.delete("/documents/by-filename/handbook.pdf")

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"
    assert delete_backend.vector_store.deleted == ["doc-1"]
    assert delete_backend.cache.exists("doc-1") is False
    assert raw_file.exists() is False
    assert "search:" in delete_backend.recording.prefixes
    assert "doc:doc-1:" in delete_backend.recording.prefixes


async def test_delete_keep_files_preserves_cache_and_raw_file(
    api_client: httpx.AsyncClient,
    delete_backend: DeleteBackend,
) -> None:
    seed_ocr_cache(delete_backend.cache, document_id="doc-1", filename="handbook.pdf")
    raw_file = delete_backend.raw_dir / "doc-1.pdf"
    raw_file.write_bytes(b"%PDF raw")
    delete_backend.vector_store.document_ids = {"doc-1"}

    response = await api_client.delete(
        "/documents/by-filename/handbook.pdf",
        params={"keep_files": "true"},
    )

    assert response.status_code == 200
    assert delete_backend.vector_store.deleted == ["doc-1"]
    assert delete_backend.cache.exists("doc-1") is True
    assert raw_file.exists() is True
    assert "search:" in delete_backend.recording.prefixes


async def test_delete_without_chunks_returns_404(
    api_client: httpx.AsyncClient,
    delete_backend: DeleteBackend,
) -> None:
    response = await api_client.delete("/documents/by-filename/absent.pdf")

    assert response.status_code == 404
    assert delete_backend.vector_store.deleted == []


async def test_delete_locked_document_returns_409(
    api_client: httpx.AsyncClient,
    delete_backend: DeleteBackend,
) -> None:
    seed_ocr_cache(delete_backend.cache, document_id="doc-1", filename="handbook.pdf")
    delete_backend.vector_store.document_ids = {"doc-1"}

    with delete_backend.locks.hold("doc-1") as acquired:
        assert acquired is True
        response = await api_client.delete("/documents/by-filename/handbook.pdf")

    assert response.status_code == 409
    assert delete_backend.vector_store.deleted == []
    assert delete_backend.cache.exists("doc-1") is True
