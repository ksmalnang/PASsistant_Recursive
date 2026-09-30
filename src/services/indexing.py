"""Document indexing pipeline using RecursiveCharacterTextSplitter."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import PointStruct

from src.clients.redis import RedisCache, get_cache
from src.config import get_settings
from src.utils.state import DocumentUpload

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkMetadata:
    """Metadata for a single document chunk."""

    chunk_id: str
    chunk_index: int
    text: str
    page_numbers: list[int]


@dataclass(slots=True)
class IndexingResult:
    """Outcome of indexing a single document."""

    document_id: str
    chunk_count: int
    indexed_chars: int
    issues: list[str] = field(default_factory=list)


class DocumentIndexingService:
    """Split, embed, and upsert document chunks into Qdrant.

    This service takes a fully processed ``DocumentUpload`` (OCR text already
    populated) and produces searchable vector chunks in the configured Qdrant
    collection.

    It reuses the ``VectorStoreTools`` facade for embeddings, BM25 encoding,
    and Qdrant client access so that no duplicate connections are created.
    """

    _EMBEDDING_BATCH_SIZE = 32

    def __init__(self, vector_store: Any | None = None) -> None:
        if vector_store is None:
            from src.utils.vector_store import VectorStoreTools

            vector_store = VectorStoreTools()
        self._vs = vector_store
        self._cache: RedisCache = get_cache()

        settings = get_settings()
        self._chunk_size = settings.CHUNK_SIZE
        self._chunk_overlap = settings.CHUNK_OVERLAP
        self._chunk_separators = list(settings.CHUNK_SEPARATORS)

    async def index_document(self, document: DocumentUpload) -> IndexingResult:
        """Split, embed, and upsert a single document into the vector store.

        Args:
            document: A ``DocumentUpload`` whose ``extracted_text`` has already
                been populated by the OCR pipeline.

        Returns:
            An ``IndexingResult`` summarising what was indexed.
        """
        text = document.extracted_text or ""
        if not text.strip():
            return IndexingResult(
                document_id=document.document_id,
                chunk_count=0,
                indexed_chars=0,
                issues=["No extracted text available for indexing."],
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._chunk_separators,
            length_function=len,
            is_separator_regex=False,
        )
        raw_chunks: list[str] = splitter.split_text(text)
        if not raw_chunks:
            return IndexingResult(
                document_id=document.document_id,
                chunk_count=0,
                indexed_chars=0,
                issues=["Text splitter produced no chunks."],
            )

        page_boundaries = self._build_page_boundaries(document)
        chunks = self._build_chunk_metadata(
            document_id=document.document_id,
            raw_chunks=raw_chunks,
            full_text=text,
            page_boundaries=page_boundaries,
        )

        self._vs.ensure_collection()
        points = await self._build_points(chunks, document)
        self._upsert_points(points)

        self._invalidate_document_cache(document.document_id)

        indexed_chars = sum(len(chunk.text) for chunk in chunks)
        logger.info(
            "Document indexed",
            extra={
                "document_id": document.document_id,
                "doc_filename": document.filename,
                "chunk_count": len(chunks),
                "indexed_chars": indexed_chars,
            },
        )
        return IndexingResult(
            document_id=document.document_id,
            chunk_count=len(chunks),
            indexed_chars=indexed_chars,
        )

    def _build_page_boundaries(self, document: DocumentUpload) -> list[tuple[int, int]]:
        """Build a list of (start_offset, end_offset) pairs per page.

        Uses the per-page OCR results stored on the document to reconstruct
        character offset boundaries.  When page-level data is unavailable the
        entire text is treated as page 1.
        """
        page_statuses = document.ocr_page_status
        if not page_statuses:
            total = len(document.extracted_text or "")
            return [(0, total)]

        boundaries: list[tuple[int, int]] = []
        offset = 0
        for page_status in page_statuses:
            text_length = page_status.get("text_length", 0)
            if text_length > 0:
                boundaries.append((offset, offset + text_length))
                # Account for the "\n\n" page separator added by OCR assembly
                offset += text_length + 2
            else:
                # Pages with no text still occupy a boundary slot
                boundaries.append((offset, offset))
        return boundaries

    def _build_chunk_metadata(
        self,
        *,
        document_id: str,
        raw_chunks: list[str],
        full_text: str,
        page_boundaries: list[tuple[int, int]],
    ) -> list[ChunkMetadata]:
        """Assign a chunk ID, index, and page numbers to each raw chunk."""
        chunks: list[ChunkMetadata] = []
        search_start = 0

        for index, chunk_text in enumerate(raw_chunks):
            chunk_start = full_text.find(chunk_text, search_start)
            if chunk_start == -1:
                # Fallback: if exact match fails, use current search position
                chunk_start = search_start
            chunk_end = chunk_start + len(chunk_text)
            # Advance search start past the overlap region
            search_start = max(chunk_start + 1, chunk_end - self._chunk_overlap)

            page_numbers = self._pages_for_span(chunk_start, chunk_end, page_boundaries)

            chunks.append(
                ChunkMetadata(
                    chunk_id=f"{document_id}:chunk:{index}",
                    chunk_index=index,
                    text=chunk_text,
                    page_numbers=page_numbers,
                )
            )
        return chunks

    def _pages_for_span(
        self,
        start: int,
        end: int,
        page_boundaries: list[tuple[int, int]],
    ) -> list[int]:
        """Return 1-based page numbers that overlap with [start, end)."""
        pages: list[int] = []
        for page_index, (page_start, page_end) in enumerate(page_boundaries):
            if page_start >= end:
                break
            if page_end <= start:
                continue
            pages.append(page_index + 1)  # 1-based for user display
        return pages or [1]

    async def _build_points(
        self,
        chunks: list[ChunkMetadata],
        document: DocumentUpload,
    ) -> list[PointStruct]:
        """Embed chunks and build Qdrant point structs in batches."""
        embeddings_client = self._vs._get_embeddings()
        all_texts = [chunk.text for chunk in chunks]
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(all_texts), self._EMBEDDING_BATCH_SIZE):
            batch = all_texts[batch_start : batch_start + self._EMBEDDING_BATCH_SIZE]
            batch_embeddings = await embeddings_client.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        if all_embeddings:
            self._vs._ensure_collection_for_vector_size(len(all_embeddings[0]))

        points: list[PointStruct] = []
        for chunk, embedding in zip(chunks, all_embeddings, strict=True):
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_id": document.document_id,
                "filename": document.filename,
                "doc_title": document.document_title,
                "document_type": document.document_type.value,
                "chunk_type": "chunk",
                "chunk_index": chunk.chunk_index,
                "source_locations": [{"page": page} for page in chunk.page_numbers],
            }

            vectors: dict[str, Any] | list[float] = embedding
            bm25_vector = None
            if self._vs._supports_bm25_vectors():
                bm25_vector = self._vs._build_bm25_vector(chunk.text)

            if bm25_vector is not None:
                vectors = {
                    "": embedding,
                    self._vs.bm25_vector_name: bm25_vector,
                }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload=payload,
                )
            )
        return points

    def _upsert_points(self, points: list[PointStruct]) -> None:
        """Upsert points into Qdrant in manageable batches."""
        batch_size = 64
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            self._vs.client.upsert(
                collection_name=self._vs.collection_name,
                points=batch,
            )

    def _invalidate_document_cache(self, document_id: str) -> None:
        """Invalidate cached search results that may reference this document.

        Performs both the legacy blanket invalidation and a per-document
        invalidation by tracking indexed document IDs in a Redis set.
        """
        # Blanket search cache invalidation (existing behaviour)
        self._cache.delete_prefix("search:")

        # Track indexed document for future granular invalidation
        self._cache.add_to_set("indexed_documents", document_id, ttl_seconds=0)
        # Delete any per-document cache entries
        self._cache.delete_prefix(f"doc:{document_id}:")
