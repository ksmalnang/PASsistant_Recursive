"""Ingestion-path tests for OCR cache persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.document_processing import (
    DocumentIngestionService,
    DocumentProcessingService,
)
from src.services.ocr_cache import OCRCache
from src.utils.state import ProcessingStatus
from tests.doubles import (
    FakeIndexer,
    FakeTextExtractor,
    FakeUploadPreparer,
    make_ocr_result,
)

PDF_BYTES = b"%PDF-1.4 test payload"


def _ingestion_service(
    tmp_path: Path,
    *,
    cache: OCRCache | None,
    indexer: FakeIndexer | None = None,
    extractor: FakeTextExtractor | None = None,
    processor: DocumentProcessingService | None = None,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        upload_preparer=FakeUploadPreparer(tmp_path / "raw"),
        processor=processor or DocumentProcessingService(extractor or FakeTextExtractor()),
        indexer=indexer if indexer is not None else FakeIndexer(),
        ocr_cache=cache,
    )


async def test_knowledge_base_ingestion_writes_ocr_cache(tmp_path: Path) -> None:
    """POST /upload persists raw per-page OCR output next to the document."""
    cache = OCRCache(tmp_path / "cache")
    indexer = FakeIndexer()
    service = _ingestion_service(tmp_path, cache=cache, indexer=indexer)

    document = await service.ingest_upload(PDF_BYTES, "handbook.pdf")

    assert document.processing_status is ProcessingStatus.COMPLETED
    assert cache.exists(document.document_id)
    entry = cache.load(document.document_id)
    assert entry is not None
    assert entry.meta.filename == "handbook.pdf"
    assert entry.meta.raw_file == Path(document.file_path).name
    assert entry.meta.source_file_sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
    assert entry.meta.ocr_page_status == document.ocr_page_status
    assert entry.meta.num_pages == 2
    assert [page.extracted_text for page in entry.pages] == ["Halo dunia", "Tabel nilai"]
    assert entry.pages[0].layout_details[0]["content"] == "Halo dunia"
    assert indexer.documents == [document]


async def test_chat_upload_path_does_not_write_cache(tmp_path: Path) -> None:
    """The chat-upload graph path never indexes, so it must not write a cache."""
    cache = OCRCache(tmp_path / "cache")
    processor = DocumentProcessingService(FakeTextExtractor())
    service = _ingestion_service(tmp_path, cache=cache, processor=processor)

    document = service.prepare_upload(PDF_BYTES, "chat.pdf")
    await processor.process_document(document)

    assert document.processing_status is ProcessingStatus.COMPLETED
    assert cache.list() == []
    assert cache.exists(document.document_id) is False


async def test_cache_write_failure_does_not_fail_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache write error is logged and ingestion continues normally."""
    cache = OCRCache(tmp_path / "cache")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cache, "save", boom)
    service = _ingestion_service(tmp_path, cache=cache)

    document = await service.ingest_upload(PDF_BYTES, "handbook.pdf")

    assert document.processing_status is ProcessingStatus.COMPLETED
    assert cache.exists(document.document_id) is False


async def test_cache_disabled_skips_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OCR_CACHE_ENABLED=false disables cache writes entirely."""
    monkeypatch.setattr(
        "src.services.document_processing.get_settings",
        lambda: SimpleNamespace(OCR_CACHE_ENABLED=False, GLM_LAYOUT_MODEL="glm-ocr"),
    )
    service = _ingestion_service(tmp_path, cache=None)

    document = await service.ingest_upload(PDF_BYTES, "handbook.pdf")

    assert document.processing_status is ProcessingStatus.COMPLETED
    assert not (tmp_path / "cache").exists()


def test_make_ocr_result_failed_page_shape() -> None:
    """The test double keeps failed pages with status and error."""
    result = make_ocr_result(page_texts=["ok", ""], statuses=["success", "failed"])

    assert result.page_results is not None
    assert result.page_results[1]["status"] == "failed"
    assert result.page_results[1]["error"] == "provider error"
    assert result.page_texts == ["ok", ""]
