"""HTTP-layer tests for the knowledge-base upload endpoints (IT-005..IT-008).

Route validation, multipart handling, and response mapping run as production
code; the OCR/indexing processor is the only replaced dependency.
"""

from __future__ import annotations

import httpx
import pytest

from tests.doubles import FakeDocumentProcessor
from tests.factories import make_upload

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


async def test_api_05_upload_ingests_valid_pdf(
    api_client: httpx.AsyncClient,
    documents_backend: FakeDocumentProcessor,
    sample_pdf_bytes: bytes,
) -> None:
    """A valid PDF is ingested and reported with its document metadata."""
    response = await api_client.post(
        "/upload",
        files=[make_upload("academic-policy.pdf", sample_pdf_bytes, "application/pdf")],
    )

    assert response.status_code == 201
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1

    result = body[0]
    assert result["success"] is True
    assert result["document_id"]
    assert result["filename"] == "academic-policy.pdf"
    assert result["document_type"] == "policy"
    assert result["status"] == "completed"
    assert result["error"] is None
    assert documents_backend.uploads == [("academic-policy.pdf", sample_pdf_bytes)]
    assert documents_backend.indexed_document_ids == [result["document_id"]]


async def test_api_06_upload_reports_ingestion_failure(
    api_client: httpx.AsyncClient,
    documents_backend: FakeDocumentProcessor,
) -> None:
    """A controlled OCR failure is reported as failed and never reaches indexing."""
    documents_backend.fail_with = RuntimeError("OCR extraction failed for corrupt PDF")

    response = await api_client.post(
        "/upload",
        files=[make_upload("corrupt.pdf", b"%PDF-1.4 corrupt payload", "application/pdf")],
    )

    assert response.status_code == 201
    result = response.json()[0]
    assert result["success"] is False
    assert result["filename"] == "corrupt.pdf"
    assert result["status"] == "failed"
    assert "OCR extraction failed for corrupt PDF" in result["error"]
    assert documents_backend.indexed_document_ids == []


async def test_api_07_upload_rejects_unsupported_formats(
    api_client: httpx.AsyncClient,
    documents_backend: FakeDocumentProcessor,
    sample_pdf_bytes: bytes,
) -> None:
    """DOCX, TXT, and unsupported image uploads are rejected; PDF stays accepted."""
    unsupported = [
        ("thesis.docx", make_upload("thesis.docx", b"PK\x03\x04docx-payload", DOCX_MIME)),
        ("notes.txt", make_upload("notes.txt", b"plain text notes", "text/plain")),
        ("scan.bmp", make_upload("scan.bmp", b"BM\x00\x00bitmap", "image/bmp")),
    ]

    for filename, part in unsupported:
        response = await api_client.post("/upload", files=[part])

        assert response.status_code == 415, filename
        assert "Unsupported file type" in response.json()["detail"]

    assert documents_backend.uploads == []

    accepted = await api_client.post(
        "/upload",
        files=[make_upload("academic-policy.pdf", sample_pdf_bytes, "application/pdf")],
    )
    assert accepted.status_code == 201


async def test_api_08_upload_rejects_oversized_and_empty_files(
    api_client: httpx.AsyncClient,
    documents_backend: FakeDocumentProcessor,
    sample_pdf_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized files fail with 413; empty files are reported failed, never indexed."""
    monkeypatch.setattr("src.utils.tools.ocr.GLMOCRTool.MAX_PDF_BYTES", 16)

    oversized = await api_client.post(
        "/upload",
        files=[make_upload("oversized.pdf", sample_pdf_bytes, "application/pdf")],
    )

    assert oversized.status_code == 413
    assert "too large" in oversized.json()["detail"]
    assert documents_backend.uploads == []

    documents_backend.fail_with = ValueError("Empty document payload")
    empty = await api_client.post(
        "/upload",
        files=[make_upload("empty.pdf", b"", "application/pdf")],
    )

    assert empty.status_code == 201
    assert empty.json()[0]["status"] == "failed"
    assert documents_backend.indexed_document_ids == []
