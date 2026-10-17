"""Re-index documents from the persisted OCR cache, without re-running OCR.

The service rebuilds the OCR result through the same GLM-OCR helpers used at
ingestion, so chunking and page-number derivation stay identical, then calls
the existing indexing service with replace_existing=True.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.config import get_settings
from src.services.contracts import DocumentIndexer
from src.services.document_locks import (
    DocumentLockedError,
    DocumentLockRegistry,
    document_locks,
)
from src.services.indexing import DocumentIndexingService
from src.services.ocr_cache import OCRCache, OCRCacheEntry
from src.utils.state import DocumentType, DocumentUpload, OCRResult, ProcessingStatus
from src.utils.tools.ocr import GLMOCRTool, OCRPageResult

logger = logging.getLogger(__name__)

REINDEXED = "reindexed"
SKIPPED_NO_CACHE = "skipped_no_cache"
FAILED = "failed"


class NoOCRCacheError(LookupError):
    """Raised when a document has no complete OCR cache entry."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"no OCR cache for {document_id}, re-upload required")


@dataclass(slots=True)
class ReindexResult:
    """Outcome of reindexing one document from cache."""

    document_id: str
    filename: str
    status: str
    chunks: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "status": self.status,
            "chunks": self.chunks,
            "error": self.error,
        }


class OCRReindexService:
    """Rebuild and index documents from their cached OCR output."""

    def __init__(
        self,
        cache: OCRCache | None = None,
        indexer: DocumentIndexer | None = None,
        locks: DocumentLockRegistry | None = None,
    ) -> None:
        self._cache = cache if cache is not None else OCRCache()
        self._indexer = indexer if indexer is not None else DocumentIndexingService()
        self._locks = locks if locks is not None else document_locks

    async def reindex_document(
        self,
        document_id: str,
        *,
        rebuild_text: bool = False,
    ) -> ReindexResult:
        """Reindex one document. Raises for a missing cache or a lock conflict."""
        with self._locks.hold(document_id) as acquired:
            if not acquired:
                raise DocumentLockedError(document_id)
            entry = self._cache.load(document_id)
            if entry is None:
                raise NoOCRCacheError(document_id)
            return await self._reindex_entry(entry, rebuild_text=rebuild_text)

    async def reindex_by_filename(
        self,
        filename: str,
        *,
        rebuild_text: bool = False,
    ) -> list[ReindexResult]:
        """Reindex every cached document with the given filename."""
        document_ids = self._cache.find_by_filename(filename)
        if not document_ids:
            raise NoOCRCacheError(filename)
        return [
            await self.reindex_document(document_id, rebuild_text=rebuild_text)
            for document_id in document_ids
        ]

    def all_cached_document_ids(self) -> list[str]:
        """Return every document id that currently has an OCR cache entry."""
        return sorted(meta.document_id for meta in self._cache.list())

    def document_ids_for_filename(self, filename: str) -> list[str]:
        """Return cached document ids whose metadata carries this filename."""
        return self._cache.find_by_filename(filename)

    async def reindex_many(
        self,
        document_ids: list[str],
        *,
        rebuild_text: bool = False,
        on_result: Callable[[ReindexResult], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[ReindexResult]:
        """Reindex a list of documents, recording per-document outcomes."""
        results: list[ReindexResult] = []
        for document_id in document_ids:
            if should_cancel is not None and should_cancel():
                break
            try:
                result = await self.reindex_document(document_id, rebuild_text=rebuild_text)
            except NoOCRCacheError as exc:
                result = ReindexResult(
                    document_id=document_id,
                    filename="",
                    status=SKIPPED_NO_CACHE,
                    error=str(exc),
                )
            except DocumentLockedError:
                result = ReindexResult(
                    document_id=document_id,
                    filename="",
                    status=FAILED,
                    error="locked",
                )
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    async def _reindex_entry(
        self,
        entry: OCRCacheEntry,
        *,
        rebuild_text: bool,
    ) -> ReindexResult:
        document = self._rebuild_document(entry, rebuild_text=rebuild_text)
        try:
            indexing = await self._indexer.index_document(document, replace_existing=True)
        except Exception as exc:
            logger.error(
                "Reindexing failed",
                exc_info=True,
                extra={"document_id": entry.meta.document_id, "ocr_error": str(exc)},
            )
            return ReindexResult(
                document_id=entry.meta.document_id,
                filename=entry.meta.filename,
                status=FAILED,
                error=str(exc),
            )
        logger.info(
            "Document reindexed from OCR cache",
            extra={
                "document_id": entry.meta.document_id,
                "chunk_count": indexing.chunk_count,
                "rebuild_text": rebuild_text,
            },
        )
        return ReindexResult(
            document_id=entry.meta.document_id,
            filename=entry.meta.filename,
            status=REINDEXED,
            chunks=indexing.chunk_count,
        )

    def _rebuild_document(
        self,
        entry: OCRCacheEntry,
        *,
        rebuild_text: bool,
    ) -> DocumentUpload:
        """Rebuild the ingestion-time document from cached OCR output."""
        result = self._rebuild_ocr_result(entry, rebuild_text=rebuild_text)
        self._verify_page_status(entry, result)

        raw_file = Path(entry.meta.raw_file) if entry.meta.raw_file else None
        raw_path = (get_settings().raw_data_dir / raw_file) if raw_file else None
        file_size = raw_path.stat().st_size if raw_path is not None and raw_path.is_file() else 0
        return DocumentUpload(
            document_id=entry.meta.document_id,
            filename=entry.meta.filename,
            file_path=str(raw_path) if raw_path is not None else "",
            document_type=self._document_type(entry.meta.document_type),
            mime_type=entry.meta.file_type,
            file_size=file_size,
            document_title=entry.meta.doc_title,
            extracted_text=result.text,
            text_quality_score=result.text_quality_score,
            processing_status=ProcessingStatus.COMPLETED,
            num_pages=result.num_pages,
            layout_details=result.layout_details,
            parsed_pages=result.parsed_pages,
            failed_pages=list(result.failed_pages or []),
            ocr_warnings=list(result.ocr_warnings or []),
            ocr_page_status=list(result.page_results or []),
        )

    def _rebuild_ocr_result(
        self,
        entry: OCRCacheEntry,
        *,
        rebuild_text: bool,
    ) -> OCRResult:
        """Reuse the ingestion-time OCR assembly so offsets and joins match."""
        page_results: list[OCRPageResult] = []
        for page in entry.pages:
            text = page.extracted_text
            if rebuild_text:
                derived = GLMOCRTool._text_from_layout_details(page.layout_details)
                text = derived or page.extracted_text
            page_results.append(
                OCRPageResult(
                    page_index=page.page_index,
                    status=page.status,
                    text=text,
                    layout_details=list(page.layout_details),
                    error=page.error,
                )
            )
        return GLMOCRTool._build_ocr_result_from_pages(
            page_results=page_results,
            total_pages=entry.meta.num_pages,
        )

    @staticmethod
    def _document_type(value: str) -> DocumentType:
        try:
            return DocumentType(value)
        except ValueError:
            return DocumentType.OTHER

    @staticmethod
    def _verify_page_status(entry: OCRCacheEntry, result: OCRResult) -> None:
        """Confirm the rebuilt page status matches the ingestion-time snapshot."""
        expected = entry.meta.ocr_page_status
        rebuilt = result.page_results or []
        if expected and rebuilt != expected:
            logger.warning(
                "Rebuilt OCR page status differs from the ingestion snapshot",
                extra={"document_id": entry.meta.document_id},
            )


_default_reindex_service: OCRReindexService | None = None


def get_reindex_service() -> OCRReindexService:
    """Return the process-wide reindex service."""
    global _default_reindex_service
    if _default_reindex_service is None:
        _default_reindex_service = OCRReindexService()
    return _default_reindex_service
