"""Document processing node."""

from src.services.contracts import (
    DocumentIndexer,
    DocumentTextExtractor,
    DocumentUploadPreparer,
)
from src.services.document_processing import (
    DocumentIngestionService,
    DocumentProcessingService,
)
from src.services.indexing import DocumentIndexingService
from src.utils.state import AgentState, DocumentUpload
from src.utils.tools import DocumentTools, GLMOCRTool


class DocumentProcessingNode:
    """
    Handles document OCR and indexing.

    Workflow:
    1. Save uploaded file
    2. Extract text using GLM-4 OCR
    3. Split, embed, and index chunks into Qdrant
    4. Update document status
    """

    def __init__(
        self,
        text_extractor: DocumentTextExtractor | None = None,
        upload_preparer: DocumentUploadPreparer | None = None,
        processing_service: DocumentProcessingService | None = None,
        ingestion_service: DocumentIngestionService | None = None,
        indexer: DocumentIndexer | None = None,
    ):
        self._processing_service = processing_service or DocumentProcessingService(
            text_extractor=text_extractor or GLMOCRTool(),
        )
        resolved_indexer = indexer if indexer is not None else DocumentIndexingService()
        self._ingestion_service = ingestion_service or DocumentIngestionService(
            upload_preparer=upload_preparer or DocumentTools(),
            processor=self._processing_service,
            indexer=resolved_indexer,
        )

    async def run(self, state: AgentState) -> dict:
        """
        Process pending documents in state.

        Args:
            state: Current agent state with pending documents

        Returns:
            State updates with processed documents
        """
        if not state.pending_documents:
            return {}
        result = await self._processing_service.process_pending_documents(
            state.pending_documents
        )

        updates = {
            "processed_documents": state.processed_documents + result.processed_documents,
            "pending_documents": [],
        }
        quality_warnings = [
            document.quality_warning
            for document in result.processed_documents
            if document.quality_warning
        ]
        if quality_warnings:
            updates["quality_warning"] = " ; ".join(quality_warnings)
        if result.errors:
            updates["error"] = "; ".join(result.errors)
        return updates

    async def _process_document(self, document: DocumentUpload) -> None:
        """Run OCR and vector storage for a single document."""
        await self._processing_service.process_document(document)

    def prepare_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """
        Save an uploaded file and return local metadata only.

        Args:
            file_bytes: Raw file bytes
            filename: Original filename

        Returns:
            DocumentUpload persisted to raw storage and ready for later processing
        """
        return self._ingestion_service.prepare_upload(file_bytes, filename)

    async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """
        Save and fully ingest an uploaded document.

        Args:
            file_bytes: Raw file bytes
            filename: Original filename

        Returns:
            Fully processed DocumentUpload
        """
        return await self._ingestion_service.ingest_upload(file_bytes, filename)


_default_document_processor: DocumentProcessingNode | None = None


def get_document_processor() -> DocumentProcessingNode:
    """Return the process-wide document processor."""
    global _default_document_processor
    if _default_document_processor is None:
        _default_document_processor = DocumentProcessingNode()
    return _default_document_processor
