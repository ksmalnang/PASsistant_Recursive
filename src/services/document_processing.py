"""Focused services for document preparation and ingestion."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import get_settings
from src.services.contracts import (
    DocumentIndexer,
    DocumentTextExtractor,
    DocumentUploadPreparer,
)
from src.services.ocr_cache import CachedPage, OCRCache, OCRCacheMeta
from src.utils.state import DocumentUpload, OCRResult, ProcessingStatus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentProcessingResult:
    """Outcome for a batch of pending document operations."""

    processed_documents: list[DocumentUpload]
    errors: list[str]


class DocumentProcessingService:
    """Process stored documents through OCR."""

    def __init__(
        self,
        text_extractor: DocumentTextExtractor,
    ):
        self._text_extractor = text_extractor

    async def process_pending_documents(
        self,
        documents: list[DocumentUpload],
    ) -> DocumentProcessingResult:
        """Process a batch of pending documents."""
        processed: list[DocumentUpload] = []
        errors: list[str] = []

        for document in documents:
            try:
                await self.process_document(document)
                processed.append(document)
                logger.info(
                    "Document processed successfully",
                    extra={
                        "document_id": document.document_id,
                        "quality_score": document.text_quality_score,
                    },
                )
            except Exception as exc:
                document.processing_status = ProcessingStatus.FAILED
                document.processing_error = str(exc)
                errors.append(f"{document.filename}: {exc}")
                logger.error("Document processing failed: %s", exc)

        return DocumentProcessingResult(
            processed_documents=processed,
            errors=errors,
        )

    async def process_document(self, document: DocumentUpload) -> OCRResult:
        """Process a single document in place and return the raw OCR result."""
        try:
            document.processing_status = ProcessingStatus.PROCESSING
            document.processing_error = None
            result = await self._text_extractor.extract_text(
                document.file_path,
                document.document_type,
            )
            document.extracted_text = result.text
            document.text_quality_score = result.text_quality_score
            document.num_pages = result.num_pages
            document.layout_details = result.layout_details
            document.parsed_pages = result.parsed_pages
            document.failed_pages = list(result.failed_pages or [])
            document.ocr_warnings = list(result.ocr_warnings or [])
            document.ocr_page_status = list(result.page_results or [])
            if document.ocr_warnings:
                document.processing_error = "; ".join(document.ocr_warnings)
            document.processing_status = ProcessingStatus.COMPLETED
            return result
        except Exception as exc:
            document.processing_status = ProcessingStatus.FAILED
            document.processing_error = str(exc)
            raise


class DocumentIngestionService:
    """Prepare uploads, run OCR, persist the OCR cache, and index."""

    OCR_PROVIDER = "zhipu"

    def __init__(
        self,
        upload_preparer: DocumentUploadPreparer,
        processor: DocumentProcessingService,
        indexer: DocumentIndexer | None = None,
        ocr_cache: OCRCache | None = None,
    ):
        self._upload_preparer = upload_preparer
        self._processor = processor
        self._indexer = indexer
        self._ocr_cache = ocr_cache

    def prepare_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Persist an uploaded file and return its metadata."""
        return self._upload_preparer.save_upload(file_bytes, filename)

    async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Prepare and fully ingest an uploaded document (knowledge-base path)."""
        document = self.prepare_upload(file_bytes, filename)
        result = await self._processor.process_document(document)
        self._write_ocr_cache(document, result)
        if self._indexer is not None and document.extracted_text:
            result_indexing = await self._indexer.index_document(document)
            document.ingestion_report = {
                "chunk_count": result_indexing.chunk_count,
                "indexed_chars": result_indexing.indexed_chars,
                "issues": result_indexing.issues,
            }
            logger.info(
                "Document indexed into vector store",
                extra={
                    "document_id": document.document_id,
                    "chunk_count": result_indexing.chunk_count,
                },
            )
        return document

    def _resolve_cache(self) -> OCRCache | None:
        """Return the configured OCR cache, creating it lazily and only when enabled."""
        if self._ocr_cache is not None:
            return self._ocr_cache
        if not get_settings().OCR_CACHE_ENABLED:
            return None
        self._ocr_cache = OCRCache()
        return self._ocr_cache

    def _write_ocr_cache(self, document: DocumentUpload, result: OCRResult) -> None:
        """Persist raw per-page OCR output; a failure never fails ingestion."""
        cache = self._resolve_cache()
        if cache is None:
            return
        try:
            cache.save(*self._build_cache_entry(document, result))
        except Exception as exc:
            logger.warning(
                "OCR cache write failed; continuing ingestion",
                extra={"document_id": document.document_id, "ocr_error": str(exc)},
            )

    def _build_cache_entry(
        self,
        document: DocumentUpload,
        result: OCRResult,
    ) -> tuple[OCRCacheMeta, list[CachedPage]]:
        layouts = result.layout_details or []
        texts = result.page_texts or []
        pages = [
            CachedPage(
                page_index=index,
                page=index + 1,
                status=str(status.get("status", "unknown")),
                layout_details=list(layouts[index]) if index < len(layouts) else [],
                extracted_text=texts[index] if index < len(texts) else "",
                error=status.get("error"),
            )
            for index, status in enumerate(result.page_results or [])
        ]
        meta = OCRCacheMeta(
            document_id=document.document_id,
            filename=document.filename,
            doc_title=document.document_title,
            document_type=document.document_type.value,
            file_type=document.mime_type,
            num_pages=document.num_pages,
            raw_file=Path(document.file_path).name,
            source_file_sha256=self._file_sha256(document.file_path),
            ocr_provider=self.OCR_PROVIDER,
            ocr_model=get_settings().GLM_LAYOUT_MODEL,
            ocr_timestamp=datetime.now(UTC).isoformat(),
            ocr_page_status=list(document.ocr_page_status),
        )
        return meta, pages

    @staticmethod
    def _file_sha256(file_path: str) -> str:
        """Hash the stored raw file, returning an empty string if it is missing."""
        path = Path(file_path)
        if not path.is_file():
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
