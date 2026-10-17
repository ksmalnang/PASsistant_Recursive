"""Delete ingested documents: chunks, search cache, OCR cache, and raw files."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import Any

from src.clients.redis import RedisCache, get_cache
from src.services.document_locks import (
    DocumentLockedError,
    DocumentLockRegistry,
    document_locks,
)
from src.services.ocr_cache import OCRCache
from src.utils.tools.document import DocumentTools

logger = logging.getLogger(__name__)


class DocumentDeletionService:
    """Remove everything a document owns, coordinated by per-document locks."""

    def __init__(
        self,
        vector_store: Any | None = None,
        ocr_cache: OCRCache | None = None,
        document_tools: DocumentTools | None = None,
        cache: RedisCache | None = None,
        locks: DocumentLockRegistry | None = None,
    ) -> None:
        if vector_store is None:
            from src.utils.vector_store import VectorStoreTools

            vector_store = VectorStoreTools()
        self._vs = vector_store
        self._ocr_cache = ocr_cache if ocr_cache is not None else OCRCache()
        self._document_tools = document_tools if document_tools is not None else DocumentTools()
        self._cache = cache if cache is not None else get_cache()
        self._locks = locks if locks is not None else document_locks

    def find_document_ids(self, filename: str) -> set[str]:
        """Return document ids with chunks carrying the given filename."""
        return self._vs.find_document_ids_by_filename(filename)

    def delete_documents(self, filename: str, *, keep_files: bool = False) -> list[str]:
        """Delete every document matching a filename, or raise if one is locked."""
        document_ids = sorted(self.find_document_ids(filename))
        if not document_ids:
            return []

        # Hold every lock up front so a conflict cannot leave a partial delete.
        with ExitStack() as stack:
            for document_id in document_ids:
                acquired = stack.enter_context(self._locks.hold(document_id))
                if not acquired:
                    raise DocumentLockedError(document_id)

            for document_id in document_ids:
                raw_path = None if keep_files else self._resolve_raw_file(document_id)
                self._vs.delete_document_chunks(document_id)
                self._invalidate_search_cache(document_id)
                if keep_files:
                    continue
                self._ocr_cache.delete(document_id)
                if raw_path is not None:
                    self._document_tools.delete_file(str(raw_path))
                logger.info(
                    "Deleted document by filename",
                    extra={"deleted_filename": filename, "document_id": document_id},
                )
        return document_ids

    def _resolve_raw_file(self, document_id: str):
        entry = self._ocr_cache.load(document_id)
        raw_name = entry.meta.raw_file if entry is not None else None
        return self._document_tools.resolve_stored_file(document_id, raw_name)

    def _invalidate_search_cache(self, document_id: str) -> None:
        """Drop cached retrieval results that may reference this document."""
        self._cache.delete_prefix("search:")
        self._cache.delete_prefix(f"doc:{document_id}:")


_default_deletion_service: DocumentDeletionService | None = None


def get_document_deletion_service() -> DocumentDeletionService:
    """Return the process-wide document deletion service."""
    global _default_deletion_service
    if _default_deletion_service is None:
        _default_deletion_service = DocumentDeletionService()
    return _default_deletion_service
