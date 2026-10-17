"""Tests for PDF OCR batching and retry page ordering.

The provider page limit is lowered so the batching/retry paths run fast; the
logic under test is the same one used for PDFs over 99 pages.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from src.utils.tools.ocr import (
    GLMOCRTool,
    OCRPageJob,
    OCRPageRange,
)


def _pdf_bytes(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _tool() -> GLMOCRTool:
    return GLMOCRTool.__new__(GLMOCRTool)


def test_batch_jobs_split_large_pdfs_into_page_ranges(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GLMOCRTool, "MAX_PDF_PAGES_PER_REQUEST", 3)
    path = tmp_path / "big.pdf"
    path.write_bytes(_pdf_bytes(7))

    jobs = _tool()._build_pdf_page_batch_jobs(path, 7)

    assert [(job.page_range.start_page, job.page_range.end_page) for job in jobs] == [
        (0, 3),
        (3, 6),
        (6, 7),
    ]
    assert {job.attempt_level for job in jobs} == {"batch"}


async def test_failed_batches_keep_every_page_in_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(GLMOCRTool, "MAX_PDF_PAGES_PER_REQUEST", 3)
    monkeypatch.setattr(GLMOCRTool, "FALLBACK_SUBRANGE_PAGES", 2)
    path = tmp_path / "big.pdf"
    path.write_bytes(_pdf_bytes(5))
    tool = _tool()

    async def fail(data_uri: str):
        raise ConnectionError("provider down")

    monkeypatch.setattr(tool, "_create_layout_parsing_with_retry", fail)
    jobs = tool._build_pdf_page_batch_jobs(path, 5)

    results = await tool._extract_layout_page_results(jobs=jobs)

    assert [result.page_index for result in results] == [0, 1, 2, 3, 4]
    assert all(result.status == "failed" for result in results)
    assert all("provider down" in (result.error or "") for result in results)


def test_batch_response_maps_back_to_original_page_indexes() -> None:
    class Detail:
        def __init__(self, index: int) -> None:
            self.index = index
            self.label = "text"
            self.bbox_2d = [0, 0, 10, 10]
            self.content = f"page-{index}"
            self.height = 10
            self.width = 10

    response = SimpleNamespace(
        layout_details=[[Detail(0)], [Detail(1)]],
        md_results="",
        data_info=SimpleNamespace(num_pages=2),
    )
    job = OCRPageJob(
        data_uri="",
        page_range=OCRPageRange(10, 12),
        attempt_level="batch",
    )

    results = _tool()._page_results_from_response(response, job)

    assert [result.page_index for result in results] == [10, 11]
    assert [result.text for result in results] == ["page-0", "page-1"]

    combined = GLMOCRTool._build_ocr_result_from_pages(
        page_results=results,
        total_pages=12,
    )
    assert combined.text == "page-0\n\npage-1"
    assert [page["page_index"] for page in combined.page_results or []] == list(range(12))
    assert combined.page_texts is not None
    assert combined.page_texts[10] == "page-0"
    assert combined.page_texts[11] == "page-1"
