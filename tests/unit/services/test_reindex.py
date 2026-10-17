"""Tests for reindexing documents from the OCR cache without OCR."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.services.document_locks import DocumentLockRegistry
from src.services.document_processing import (
    DocumentIngestionService,
    DocumentProcessingService,
)
from src.services.indexing import DocumentIndexingService
from src.services.ocr_cache import OCRCache
from src.services.reindex import (
    FAILED,
    REINDEXED,
    DocumentLockedError,
    NoOCRCacheError,
    OCRReindexService,
)
from src.utils.state import DocumentUpload
from src.utils.tools.ocr import GLMOCRTool
from tests.doubles import (
    FakeIndexer,
    FakeTextExtractor,
    FakeUploadPreparer,
)

PDF_BYTES = b"%PDF-1.4 reindex payload"
TABLE_HTML = "<table><tr><th>Nama</th><th>NIM</th></tr><tr><td>Adit</td><td>1</td></tr></table>"


class _NoopCache:
    def delete_prefix(self, prefix: str) -> None:
        pass

    def add_to_set(self, key: str, *values: str, ttl_seconds: int | None = None) -> None:
        pass


class _FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class _FakeQdrant:
    def __init__(self, events: list[tuple]) -> None:
        self._events = events
        self.points: list = []

    def upsert(self, collection_name: str, points: list) -> None:
        self.points.extend(points)
        self._events.append(("upsert", len(points)))


class _FakeVectorStore:
    collection_name = "test-collection"
    bm25_vector_name = "bm25"

    def __init__(self, events: list[tuple]) -> None:
        self._events = events
        self.client = _FakeQdrant(events)

    def ensure_collection(self) -> None:
        pass

    def _ensure_collection_for_vector_size(self, vector_size: int) -> None:
        pass

    def _get_embeddings(self) -> _FakeEmbeddings:
        return _FakeEmbeddings()

    def _supports_bm25_vectors(self) -> bool:
        return False

    def delete_document_chunks(self, document_id: str) -> None:
        self._events.append(("delete", document_id))


async def _ingest(
    tmp_path: Path,
    *,
    cache: OCRCache,
    filename: str = "handbook.pdf",
) -> tuple[DocumentUpload, FakeIndexer]:
    indexer = FakeIndexer()
    service = DocumentIngestionService(
        upload_preparer=FakeUploadPreparer(tmp_path / "raw"),
        processor=DocumentProcessingService(FakeTextExtractor()),
        indexer=indexer,
        ocr_cache=cache,
    )
    document = await service.ingest_upload(PDF_BYTES, filename)
    return document, indexer


async def test_reindex_matches_fresh_ingest_without_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reindex rebuilds the same text and page status and never calls OCR."""
    cache = OCRCache(tmp_path / "cache")
    fresh, _ = await _ingest(tmp_path, cache=cache)

    async def fail_ocr(*args: object, **kwargs: object) -> None:
        raise AssertionError("reindex must not call OCR")

    monkeypatch.setattr(GLMOCRTool, "extract_text", fail_ocr)
    reindex_indexer = FakeIndexer()
    service = OCRReindexService(
        cache=cache,
        indexer=reindex_indexer,
        locks=DocumentLockRegistry(),
    )

    result = await service.reindex_document(fresh.document_id)

    assert result.status == REINDEXED
    assert result.filename == "handbook.pdf"
    rebuilt = reindex_indexer.documents[0]
    assert rebuilt.extracted_text == fresh.extracted_text
    assert rebuilt.ocr_page_status == fresh.ocr_page_status
    assert rebuilt.layout_details == fresh.layout_details
    assert rebuilt.document_id == fresh.document_id
    assert reindex_indexer.replace_flags == [True]


async def test_reindex_deletes_existing_points_before_upsert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """replace_existing removes old points after embeddings, before upsert."""
    events: list[tuple] = []
    monkeypatch.setattr("src.services.indexing.get_cache", _NoopCache)
    service = DocumentIndexingService(vector_store=_FakeVectorStore(events))
    document = DocumentUpload(
        document_id="doc-replace",
        filename="handbook.pdf",
        file_path="data/raw/doc-replace.pdf",
        mime_type="application/pdf",
        file_size=1,
        extracted_text="hello world",
    )

    await service.index_document(document, replace_existing=True)

    assert events[0] == ("delete", "doc-replace")
    assert events[1][0] == "upsert"


async def test_reindex_without_replace_does_not_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple] = []
    monkeypatch.setattr("src.services.indexing.get_cache", _NoopCache)
    service = DocumentIndexingService(vector_store=_FakeVectorStore(events))
    document = DocumentUpload(
        document_id="doc-plain",
        filename="handbook.pdf",
        file_path="data/raw/doc-plain.pdf",
        mime_type="application/pdf",
        file_size=1,
        extracted_text="hello world",
    )

    await service.index_document(document)

    assert [event[0] for event in events] == ["upsert"]


async def test_rebuild_text_rederives_with_cached_fallback(tmp_path: Path) -> None:
    """rebuild_text uses layout blocks and falls back to cached page text."""
    cache = OCRCache(tmp_path / "cache")
    fresh, _ = await _ingest(tmp_path, cache=cache)

    entry = cache.load(fresh.document_id)
    assert entry is not None
    entry.pages[0].layout_details = [{"label": "table", "content": TABLE_HTML}]
    entry.pages[0].extracted_text = "STALE PAGE TEXT"
    entry.pages[1].layout_details = []
    entry.pages[1].extracted_text = "KEEP CACHED"
    cache.save(entry.meta, entry.pages)

    reindex_indexer = FakeIndexer()
    service = OCRReindexService(
        cache=cache,
        indexer=reindex_indexer,
        locks=DocumentLockRegistry(),
    )

    await service.reindex_document(fresh.document_id, rebuild_text=True)

    rebuilt = reindex_indexer.documents[0].extracted_text
    assert "| Nama | NIM |" in rebuilt
    assert "STALE PAGE TEXT" not in rebuilt
    assert "KEEP CACHED" in rebuilt


async def test_reindex_missing_cache_reports_clear_error(tmp_path: Path) -> None:
    service = OCRReindexService(
        cache=OCRCache(tmp_path / "cache"),
        indexer=FakeIndexer(),
        locks=DocumentLockRegistry(),
    )

    with pytest.raises(NoOCRCacheError) as excinfo:
        await service.reindex_document("missing-doc")

    assert str(excinfo.value) == "no OCR cache for missing-doc, re-upload required"


async def test_reindex_by_filename_matches_all_entries(tmp_path: Path) -> None:
    cache = OCRCache(tmp_path / "cache")
    first, _ = await _ingest(tmp_path, cache=cache, filename="shared.pdf")
    second_indexer = FakeIndexer()
    second_service = DocumentIngestionService(
        upload_preparer=FakeUploadPreparer(tmp_path / "raw2"),
        processor=DocumentProcessingService(FakeTextExtractor()),
        indexer=second_indexer,
        ocr_cache=cache,
    )
    second = await second_service.ingest_upload(PDF_BYTES, "shared.pdf")

    reindex_indexer = FakeIndexer()
    service = OCRReindexService(
        cache=cache,
        indexer=reindex_indexer,
        locks=DocumentLockRegistry(),
    )

    results = await service.reindex_by_filename("shared.pdf")

    assert {result.document_id for result in results} == {first.document_id, second.document_id}
    assert all(result.status == REINDEXED for result in results)
    assert set(service.all_cached_document_ids()) == {first.document_id, second.document_id}

    with pytest.raises(NoOCRCacheError):
        await service.reindex_by_filename("absent.pdf")


async def test_reindex_locked_document_raises(tmp_path: Path) -> None:
    cache = OCRCache(tmp_path / "cache")
    fresh, _ = await _ingest(tmp_path, cache=cache)
    registry = DocumentLockRegistry()
    service = OCRReindexService(cache=cache, indexer=FakeIndexer(), locks=registry)

    with registry.hold(fresh.document_id) as acquired:
        assert acquired is True
        with pytest.raises(DocumentLockedError):
            await service.reindex_document(fresh.document_id)


async def test_reindex_records_indexing_failure(tmp_path: Path) -> None:
    cache = OCRCache(tmp_path / "cache")
    fresh, _ = await _ingest(tmp_path, cache=cache)

    class FailingIndexer:
        async def index_document(self, document: DocumentUpload, *, replace_existing: bool = False):
            raise RuntimeError("qdrant unavailable")

    service = OCRReindexService(
        cache=cache,
        indexer=FailingIndexer(),
        locks=DocumentLockRegistry(),
    )

    result = await service.reindex_document(fresh.document_id)

    assert result.status == FAILED
    assert result.error == "qdrant unavailable"
    assert result.filename == "handbook.pdf"


async def test_ingested_cache_sha_matches_raw_file(tmp_path: Path) -> None:
    cache = OCRCache(tmp_path / "cache")
    fresh, _ = await _ingest(tmp_path, cache=cache)

    entry = cache.load(fresh.document_id)

    assert entry is not None
    assert entry.meta.source_file_sha256 == hashlib.sha256(PDF_BYTES).hexdigest()


async def test_reindex_produces_same_chunks_and_source_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache reindex yields the same chunk text and page locations as ingest."""
    cache = OCRCache(tmp_path / "cache")
    fresh, _ = await _ingest(tmp_path, cache=cache)
    monkeypatch.setattr("src.services.indexing.get_cache", _NoopCache)
    vector_store = _FakeVectorStore([])
    indexer = DocumentIndexingService(vector_store=vector_store)

    await indexer.index_document(fresh)
    fresh_points = [
        (point.payload["text"], point.payload["source_locations"])
        for point in vector_store.client.points
    ]
    vector_store.client.points.clear()

    service = OCRReindexService(
        cache=cache,
        indexer=indexer,
        locks=DocumentLockRegistry(),
    )
    await service.reindex_document(fresh.document_id)
    rebuilt_points = [
        (point.payload["text"], point.payload["source_locations"])
        for point in vector_store.client.points
    ]

    assert rebuilt_points == fresh_points
    assert any(location["page"] for _, locations in fresh_points for location in locations)
