"""Reusable builders for test payloads, workflow states, and fixture documents."""

from __future__ import annotations

import uuid
from typing import Any

from src.utils.state import AgentState, Citation, StudentRecord

FormData = dict[str, Any]

# Minimal PDF-shaped payload for upload flows whose OCR/indexing is mocked.
# The real upload pipeline is covered by tests/system with generated text PDFs.
PDF_STUB_BYTES = b"%PDF-1.4\n% PASsistant upload fixture (OCR/indexing mocked)\n%%EOF\n"


def make_chat_payload(message: str, *, thread_id: str | None = None) -> FormData:
    """Build form data for a chat request with an optional thread identifier."""
    payload: FormData = {"message": message}
    if thread_id is not None:
        payload["thread_id"] = thread_id
    return payload


def make_upload(
    filename: str,
    content: bytes,
    content_type: str,
) -> tuple[str, tuple[str, bytes, str]]:
    """Build one ``httpx`` multipart part for the ``files`` upload field."""
    return ("files", (filename, content, content_type))


def make_agent_state(
    *,
    response: str = "The graduation requirement is 144 credits.",
    intent: str | None = "query_document",
    citations: list[Citation] | None = None,
    session_id: str = "test-session",
    error: str | None = None,
) -> AgentState:
    """Build a final workflow state as returned by a chat agent double."""
    return AgentState(
        session_id=session_id,
        current_intent=intent,
        draft_response=response,
        citations=list(citations or []),
        error=error,
    )


def make_citation(
    index: int = 1,
    *,
    filename: str = "academic-policy.pdf",
    document_id: str = "doc-0001",
    section: str | None = None,
    page: int | None = None,
    snippet: str | None = None,
) -> Citation:
    """Build a source citation attached to a document-backed response."""
    return Citation(
        id=index,
        document_id=document_id,
        filename=filename,
        section=section,
        page=page,
        snippet=snippet,
    )


def make_student_record(
    *,
    student_id: str | None = None,
    full_name: str = "Aditya Pratama",
    email: str | None = None,
    gpa: float | None = 3.75,
    program: str = "Informatics",
) -> StudentRecord:
    """Build a realistic student record for record-service scenarios."""
    return StudentRecord(
        student_id=student_id or f"STU_{uuid.uuid4().hex[:8].upper()}",
        full_name=full_name,
        email=email or "aditya.pratama@example.com",
        gpa=gpa,
        program=program,
    )


def make_record_query_by_id_message(student_id: str) -> str:
    """Build a direct student-record lookup message by student id."""
    return f"Show the student record for student id {student_id}."


def make_record_query_by_email_message(email: str) -> str:
    """Build a direct student-record lookup message by email address."""
    return f"Please show the student record for {email}."


def build_pdf_bytes(lines: list[str], *, font_size: int = 12) -> bytes:
    """Build a minimal single-page PDF whose text extractors can read.

    Used by system tests that need a self-generated document with a known,
    unique fact instead of a checked-in binary fixture.
    """
    text_ops = ["BT", f"/F1 {font_size} Tf", "72 720 Td", f"{font_size + 6} TL"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        text_ops.append(f"({escaped}) Tj T*")
    text_ops.append("ET")
    content_stream = "\n".join(text_ops).encode("latin-1", errors="replace")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(content_stream)).encode("ascii")
            + b" >>\nstream\n"
            + content_stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    buffer = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{index} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_offset = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode("ascii")
    buffer += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(buffer)


def seed_ocr_cache(
    cache: Any,
    *,
    document_id: str = "doc-0001",
    filename: str = "handbook.pdf",
    pages: list[Any] | None = None,
    num_pages: int | None = None,
) -> Any:
    """Write a complete OCR cache entry for reindex/deletion tests."""
    from src.services.ocr_cache import CachedPage, OCRCacheMeta

    if pages is None:
        pages = [
            CachedPage(
                page_index=0,
                page=1,
                status="success",
                layout_details=[_text_block("Halo")],
                extracted_text="Halo",
            ),
            CachedPage(
                page_index=1,
                page=2,
                status="success",
                layout_details=[_text_block("Dunia")],
                extracted_text="Dunia",
            ),
        ]

    page_status = [
        {
            "page_index": page.page_index,
            "status": page.status,
            "text_length": len(page.extracted_text),
            "layout_blocks": len(page.layout_details),
            "error": page.error,
        }
        for page in pages
    ]
    meta = OCRCacheMeta(
        document_id=document_id,
        filename=filename,
        doc_title="Handbook",
        document_type="policy",
        file_type="application/pdf",
        num_pages=num_pages if num_pages is not None else len(pages),
        raw_file=f"{document_id}.pdf",
        source_file_sha256="deadbeef",
        ocr_provider="zhipu",
        ocr_model="glm-ocr",
        ocr_timestamp="2026-01-01T00:00:00+00:00",
        ocr_page_status=page_status,
    )
    cache.save(meta, pages)
    return meta


def _text_block(content: str) -> dict[str, Any]:
    return {
        "index": 0,
        "label": "text",
        "bbox_2d": [0, 0, 100, 20],
        "content": content,
        "height": 20,
        "width": 100,
    }
