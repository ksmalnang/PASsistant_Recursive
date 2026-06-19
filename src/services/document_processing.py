"""Focused services for document preparation and ingestion."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.services.contracts import (
    DocumentIndexer,
    DocumentTextExtractor,
    DocumentUploadPreparer,
)
from src.utils.state import DocumentUpload, ProcessingStatus

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

    async def process_document(self, document: DocumentUpload) -> None:
        """Process a single document in place."""
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
        except Exception as exc:
            document.processing_status = ProcessingStatus.FAILED
            document.processing_error = str(exc)
            raise


class DocumentIngestionService:
    """Prepare uploads and delegate full document ingestion."""

    def __init__(
        self,
        upload_preparer: DocumentUploadPreparer,
        processor: DocumentProcessingService,
        indexer: DocumentIndexer | None = None,
    ):
        self._upload_preparer = upload_preparer
        self._processor = processor
        self._indexer = indexer

    def prepare_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Persist an uploaded file and return its metadata."""
        return self._upload_preparer.save_upload(file_bytes, filename)

    async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Prepare and fully ingest an uploaded document."""
        document = self.prepare_upload(file_bytes, filename)
        await self._processor.process_document(document)
        if self._indexer is not None and document.extracted_text:
            result = await self._indexer.index_document(document)
            document.ingestion_report = {
                "chunk_count": result.chunk_count,
                "indexed_chars": result.indexed_chars,
                "issues": result.issues,
            }
            logger.info(
                "Document indexed into vector store",
                extra={
                    "document_id": document.document_id,
                    "chunk_count": result.chunk_count,
                },
            )
        return document
