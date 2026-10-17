"""Characterization tests for chunk-to-page mapping used by indexing.

The page boundary arithmetic ("+2" for the OCR page separator) couples the
indexing service to the OCR assembly join. Reindex-from-cache must reuse the
same code path, so this locks the current mapping down before refactoring.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.services.indexing import DocumentIndexingService
from src.utils.state import DocumentUpload


def _service() -> DocumentIndexingService:
    return DocumentIndexingService(vector_store=SimpleNamespace())


def _document(
    extracted_text: str,
    page_statuses: list[dict],
) -> DocumentUpload:
    return DocumentUpload(
        document_id="doc-1",
        filename="handbook.pdf",
        file_path="data/raw/doc-1.pdf",
        mime_type="application/pdf",
        file_size=10,
        extracted_text=extracted_text,
        ocr_page_status=page_statuses,
    )


def test_page_boundaries_skip_empty_pages_without_advancing_offset() -> None:
    """An empty page keeps its slot but does not consume a separator offset."""
    document = _document(
        "AAA\n\nCCC",
        [
            {"page_index": 0, "status": "success", "text_length": 3},
            {"page_index": 1, "status": "empty", "text_length": 0},
            {"page_index": 2, "status": "success", "text_length": 3},
        ],
    )

    assert _service()._build_page_boundaries(document) == [(0, 3), (5, 5), (5, 8)]


def test_page_boundaries_without_page_status_treat_text_as_one_page() -> None:
    """A document with no per-page status is treated as a single page."""
    document = _document("AAAA", [])

    assert _service()._build_page_boundaries(document) == [(0, 4)]


def test_pages_for_span_returns_all_overlapping_one_based_pages() -> None:
    """Spans map to every overlapping page, and fall back to page 1."""
    service = _service()
    boundaries = [(0, 3), (5, 5), (5, 8)]

    assert service._pages_for_span(0, 8, boundaries) == [1, 2, 3]
    assert service._pages_for_span(2, 4, boundaries) == [1]
    assert service._pages_for_span(5, 9, boundaries) == [3]
    assert service._pages_for_span(9, 12, boundaries) == [1]
