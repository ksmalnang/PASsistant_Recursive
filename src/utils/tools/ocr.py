"""OCR tool integrations."""

import asyncio
import base64
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]
from pypdf import PdfReader, PdfWriter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from zai import ZaiClient
from zai.types.ocr.layout_parsing_resp import LayoutParsingResp

from src.config import get_settings
from src.utils.state import DocumentType, OCRResult

logger = logging.getLogger(__name__)

PDF_MIME_TYPE = "application/pdf"
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
SUPPORTED_MIME_TYPES = {PDF_MIME_TYPE, *SUPPORTED_IMAGE_MIME_TYPES}


@dataclass(frozen=True, slots=True)
class OCRPageRange:
    """Zero-based inclusive/exclusive page range for one OCR request."""

    start_page: int
    end_page: int

    @property
    def page_count(self) -> int:
        """Return the number of pages in this range."""
        return max(0, self.end_page - self.start_page)


@dataclass(frozen=True, slots=True)
class OCRPageJob:
    """Provider payload plus original document page range metadata."""

    data_uri: str
    page_range: OCRPageRange
    attempt_level: str
    source_path: Path | None = None


@dataclass(slots=True)
class OCRPageResult:
    """Per-page OCR parse status and content."""

    page_index: int
    status: str
    text: str = ""
    layout_details: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class GLMOCRTool:
    """
    Document parsing using GLM-OCR via layout_parsing (Zhipu AI).

    Handles PDF and image files by converting them to base64
    and using the GLM-OCR layout_parsing API for text extraction
    with layout metadata.
    """

    MAX_PDF_BYTES = 50 * 1024 * 1024
    MAX_IMAGE_BYTES = 10 * 1024 * 1024
    MAX_PDF_PAGES_PER_REQUEST = 99
    FALLBACK_SUBRANGE_PAGES = 10
    RETRY_ON: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, OSError)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: ZaiClient | None = None,
        max_retries: int = 3,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.ZHIPU_API_KEY
        self.model = model if model is not None else settings.GLM_LAYOUT_MODEL
        self.client = client
        self._max_retries = max_retries

        if self.client is None and self.api_key:
            self.client = ZaiClient(
                api_key=self.api_key,
                timeout=120.0,
            )

    async def extract_text(
        self,
        file_path: str,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> OCRResult:
        """
        Extract text from a document using GLM-OCR layout_parsing.

        Args:
            file_path: Path to the document file
            document_type: Type of document for context-aware parsing

        Returns:
            OCRResult with text content, heuristic quality score (0.0–1.0),
            num_pages from data_info, and serialized layout_details.
        """
        path = Path(file_path)
        mime_type = self._detect_mime_type(path)
        local_page_count = (
            self._estimate_pdf_page_count(path) if mime_type == PDF_MIME_TYPE else None
        )
        jobs = await self._build_layout_request_jobs(
            path=path,
            mime_type=mime_type,
            total_pages=local_page_count,
        )
        page_results = await self._extract_layout_page_results(jobs=jobs)
        result = self._build_ocr_result_from_pages(
            page_results=page_results,
            total_pages=local_page_count,
        )

        logger.info(
            "OCR extraction completed",
            extra={
                "file": str(path),
                "document_type": document_type.value,
                "quality_score": result.text_quality_score,
                "text_length": len(result.text),
                "num_pages": result.num_pages,
                "ocr_requests": len(jobs),
                "parsed_pages": result.parsed_pages,
                "failed_pages": result.failed_pages,
            },
        )
        return result

    def _get_client(self) -> ZaiClient:
        if self.client is None:
            raise ValueError("ZHIPU_API_KEY is required for OCR extraction")
        return self.client

    @staticmethod
    def _serialize_layout_details(
        layout_details: list[Any],
    ) -> list[list[dict[str, Any]]]:
        result: list[list[dict[str, Any]]] = []
        for page in layout_details:
            page_items: list[dict[str, Any]] = []
            for detail in page:
                page_items.append(
                    {
                        "index": detail.index,
                        "label": detail.label,
                        "bbox_2d": detail.bbox_2d,
                        "content": detail.content,
                        "height": detail.height,
                        "width": detail.width,
                    }
                )
            result.append(page_items)
        return result

    async def _create_layout_parsing_with_retry(
        self,
        data_uri: str,
    ) -> LayoutParsingResp:
        """Call the GLM-OCR layout_parsing API with retry on transient errors."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(self.RETRY_ON),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        def _call(
            client: ZaiClient,
            model: str,
        ) -> LayoutParsingResp:
            return client.layout_parsing.create(
                model=model,
                file=data_uri,
            )

        client = self._get_client()
        return await asyncio.to_thread(
            _call,
            client,
            self.model,
        )

    async def _build_layout_request_jobs(
        self,
        *,
        path: Path,
        mime_type: str,
        total_pages: int | None,
    ) -> list[OCRPageJob]:
        """Build provider-compatible OCR jobs with original page ranges."""
        if (
            mime_type == PDF_MIME_TYPE
            and total_pages is not None
            and total_pages > self.MAX_PDF_PAGES_PER_REQUEST
        ):
            return await asyncio.to_thread(
                self._build_pdf_page_batch_jobs,
                path,
                total_pages,
            )

        self._validate_file_size(path, mime_type)
        end_page = total_pages if mime_type == PDF_MIME_TYPE and total_pages else 1
        return [
            OCRPageJob(
                data_uri=await self._build_data_uri(path, mime_type),
                page_range=OCRPageRange(0, end_page),
                attempt_level="batch" if mime_type == PDF_MIME_TYPE else "page",
                source_path=path if mime_type == PDF_MIME_TYPE else None,
            )
        ]

    def _build_pdf_page_batch_jobs(self, path: Path, total_pages: int) -> list[OCRPageJob]:
        """Split large PDFs into page-bounded jobs accepted by the OCR provider."""
        try:
            reader = PdfReader(str(path))
            actual_total_pages = len(reader.pages)
        except Exception as exc:
            raise ValueError(f"Unable to split PDF for OCR: {path}") from exc

        if actual_total_pages <= 0:
            raise ValueError(f"OCR PDF has no pages: {path}")
        if actual_total_pages != total_pages:
            logger.debug(
                "Adjusted local PDF page count before OCR batching",
                extra={
                    "file_path": str(path),
                    "estimated_pages": total_pages,
                    "actual_pages": actual_total_pages,
                },
            )
            total_pages = actual_total_pages

        logger.info(
            "Splitting PDF into OCR page jobs",
            extra={
                "file_path": str(path),
                "num_pages": total_pages,
                "pages_per_request": self.MAX_PDF_PAGES_PER_REQUEST,
            },
        )

        jobs: list[OCRPageJob] = []
        for start_page in range(0, total_pages, self.MAX_PDF_PAGES_PER_REQUEST):
            end_page = min(
                start_page + self.MAX_PDF_PAGES_PER_REQUEST,
                total_pages,
            )
            jobs.append(
                self._build_pdf_page_job(
                    reader=reader,
                    source_path=path,
                    start_page=start_page,
                    end_page=end_page,
                    attempt_level="batch",
                )
            )

        logger.info(
            "OCR page jobs created",
            extra={
                "file_path": str(path),
                "pages": total_pages,
                "jobs": len(jobs),
                "max_pages_per_request": self.MAX_PDF_PAGES_PER_REQUEST,
            },
        )
        return jobs

    def _build_pdf_page_job(
        self,
        *,
        reader: PdfReader,
        source_path: Path,
        start_page: int,
        end_page: int,
        attempt_level: str,
    ) -> OCRPageJob:
        """Build one PDF OCR job for a zero-based page range."""
        writer = PdfWriter()
        for page_index in range(start_page, end_page):
            writer.add_page(reader.pages[page_index])

        buffer = BytesIO()
        writer.write(buffer)
        chunk_bytes = buffer.getvalue()
        self._validate_bytes_size(
            len(chunk_bytes),
            PDF_MIME_TYPE,
            label=f"OCR PDF page batch {start_page + 1}-{end_page}",
        )
        return OCRPageJob(
            data_uri=self._build_data_uri_from_bytes(chunk_bytes, PDF_MIME_TYPE),
            page_range=OCRPageRange(start_page, end_page),
            attempt_level=attempt_level,
            source_path=source_path,
        )

    async def _extract_layout_page_results(
        self,
        *,
        jobs: list[OCRPageJob],
    ) -> list[OCRPageResult]:
        """Run OCR jobs and return page-scoped parse results."""
        page_results: list[OCRPageResult] = []
        for job in jobs:
            page_results.extend(await self._extract_single_job_page_results(job))
        return page_results

    async def _extract_single_job_page_results(
        self,
        job: OCRPageJob,
    ) -> list[OCRPageResult]:
        """Run one job, falling back to smaller page ranges after final failure."""
        try:
            response = await self._create_layout_parsing_with_retry(job.data_uri)
        except Exception as exc:
            if job.page_range.page_count > 1 and job.source_path is not None:
                logger.warning(
                    "OCR page job failed; retrying smaller ranges",
                    extra={
                        "pages": self._format_page_range(job.page_range),
                        "attempt_level": job.attempt_level,
                        "error": str(exc),
                    },
                )
                return await self._retry_job_as_smaller_ranges(job)
            return [
                OCRPageResult(
                    page_index=page_index,
                    status="failed",
                    error=str(exc),
                )
                for page_index in range(job.page_range.start_page, job.page_range.end_page)
            ]

        results = self._page_results_from_response(response, job)
        logger.info(
            "OCR page job completed",
            extra={
                "pages": self._format_page_range(job.page_range),
                "attempt_level": job.attempt_level,
                "text_length": sum(len(result.text) for result in results),
            },
        )
        return results

    async def _retry_job_as_smaller_ranges(
        self,
        job: OCRPageJob,
    ) -> list[OCRPageResult]:
        """Retry failed PDF jobs as 10-page subranges, then individual pages."""
        if job.source_path is None:
            return [
                OCRPageResult(
                    page_index=page_index,
                    status="failed",
                    error="Unable to retry OCR job without source PDF path",
                )
                for page_index in range(job.page_range.start_page, job.page_range.end_page)
            ]

        try:
            reader = PdfReader(str(job.source_path))
        except Exception as exc:
            return [
                OCRPageResult(
                    page_index=page_index,
                    status="failed",
                    error=f"Unable to reopen PDF for OCR retry: {exc}",
                )
                for page_index in range(job.page_range.start_page, job.page_range.end_page)
            ]

        subrange_size = (
            1 if job.attempt_level == "subrange" else self.FALLBACK_SUBRANGE_PAGES
        )
        retry_results: list[OCRPageResult] = []
        for start_page in range(job.page_range.start_page, job.page_range.end_page, subrange_size):
            end_page = min(start_page + subrange_size, job.page_range.end_page)
            attempt_level = "page" if end_page - start_page == 1 else "subrange"
            retry_job = self._build_pdf_page_job(
                reader=reader,
                source_path=job.source_path,
                start_page=start_page,
                end_page=end_page,
                attempt_level=attempt_level,
            )
            retry_results.extend(await self._extract_single_job_page_results(retry_job))
        return retry_results

    def _page_results_from_response(
        self,
        response: LayoutParsingResp,
        job: OCRPageJob,
    ) -> list[OCRPageResult]:
        """Map a batch OCR response back to original zero-based page indexes."""
        serialized_pages = (
            self._serialize_layout_details(response.layout_details)
            if response.layout_details is not None
            else []
        )
        batch_text = str(response.md_results or "").strip()
        results: list[OCRPageResult] = []
        response_num_pages = (
            data_info.num_pages
            if (data_info := getattr(response, "data_info", None)) is not None
            else None
        )
        metadata_page_count = (
            int(response_num_pages or 0) if job.page_range.page_count <= 1 else 0
        )
        page_count = max(
            job.page_range.page_count,
            len(serialized_pages),
            metadata_page_count,
        )

        for relative_index in range(page_count):
            page_index = job.page_range.start_page + relative_index
            page_layout = (
                serialized_pages[relative_index]
                if relative_index < len(serialized_pages)
                else []
            )
            page_text = self._text_from_layout_details(page_layout)
            if not page_text and relative_index == 0:
                page_text = batch_text
            status = "success" if page_text or page_layout else "empty"
            results.append(
                OCRPageResult(
                    page_index=page_index,
                    status=status,
                    text=page_text,
                    layout_details=page_layout,
                )
            )
        return results

    def _build_ocr_result_from_pages(
        self,
        *,
        page_results: list[OCRPageResult],
        total_pages: int | None,
    ) -> OCRResult:
        """Merge page-scoped OCR results into the public OCRResult contract."""
        result_by_page = {result.page_index: result for result in page_results}
        inferred_total_pages = (
            max(result_by_page) + 1 if result_by_page else total_pages
        )
        num_pages = total_pages or inferred_total_pages
        ordered_indexes = range(num_pages or 0)

        text_parts: list[str] = []
        layout_details: list[list[dict[str, Any]]] = []
        serializable_page_results: list[dict[str, Any]] = []
        failed_pages: list[int] = []
        parsed_pages = 0

        for page_index in ordered_indexes:
            page_result = result_by_page.get(
                page_index,
                OCRPageResult(page_index=page_index, status="skipped"),
            )
            if page_result.status == "success" and (
                page_result.text or page_result.layout_details
            ):
                parsed_pages += 1
            if page_result.status == "failed":
                failed_pages.append(page_index)
            if page_result.text:
                text_parts.append(page_result.text)
            layout_details.append(page_result.layout_details)
            serializable_page_results.append(self._serialize_page_result(page_result))

        extracted_text = "\n\n".join(text_parts).strip()
        if parsed_pages == 0:
            raise ValueError("OCR extraction failed: no pages were parsed successfully")

        warnings = self._build_ocr_warnings(
            parsed_pages=parsed_pages,
            num_pages=num_pages,
            failed_pages=failed_pages,
        )
        return OCRResult(
            text=extracted_text,
            text_quality_score=self._estimate_text_quality(extracted_text),
            num_pages=num_pages,
            layout_details=layout_details or None,
            parsed_pages=parsed_pages,
            failed_pages=failed_pages,
            page_results=serializable_page_results,
            ocr_warnings=warnings,
        )

    def _serialize_page_result(self, page_result: OCRPageResult) -> dict[str, Any]:
        """Return a JSON-friendly page OCR status record."""
        return {
            "page_index": page_result.page_index,
            "status": page_result.status,
            "text_length": len(page_result.text),
            "layout_blocks": len(page_result.layout_details),
            "error": page_result.error,
        }

    def _build_ocr_warnings(
        self,
        *,
        parsed_pages: int,
        num_pages: int | None,
        failed_pages: list[int],
    ) -> list[str]:
        """Build compact warnings for partial OCR success."""
        warnings: list[str] = []
        if num_pages is not None and parsed_pages < num_pages:
            warnings.append(
                f"OCR parsed {parsed_pages} of {num_pages} pages successfully."
            )
        if failed_pages:
            warnings.append(f"OCR failed for zero-based pages: {failed_pages}.")
        return warnings

    def _text_from_layout_details(self, page_layout: list[dict[str, Any]]) -> str:
        """Build page text from serialized layout block content.

        Table-labeled blocks whose content contains HTML table markup are
        converted to Markdown tables so the extracted text is clean and
        useful for downstream chunking, retrieval, and LLM context.
        """
        parts: list[str] = []
        for detail in page_layout:
            raw = str(detail.get("content") or "").strip()
            if not raw:
                continue
            label = str(detail.get("label") or "").lower()
            if label == "table" and "<table" in raw.lower():
                converted = self._convert_html_table_to_markdown(raw)
                if converted:
                    parts.append(converted)
            else:
                parts.append(raw)
        return "\n".join(parts).strip()

    @staticmethod
    def _convert_html_table_to_markdown(html: str) -> str:
        """Convert an HTML ``<table>`` fragment to a Markdown table.

        Uses Python's built-in :class:`html.parser.HTMLParser` so no
        extra dependencies are required.  Falls back to stripping all
        HTML tags and returning the plain cell text when the table
        structure cannot be reliably reconstructed.
        """
        parser = _HTMLTableParser()
        try:
            parser.feed(html)
        except Exception:
            return re.sub(r"<[^>]+>", " ", html).strip()

        rows = parser.rows
        if not rows:
            return re.sub(r"<[^>]+>", " ", html).strip()

        # Expand colspan / rowspan into a regular grid
        grid = _expand_spans(rows)
        if not grid:
            return re.sub(r"<[^>]+>", " ", html).strip()

        # Build Markdown table
        col_count = max(len(row) for row in grid)
        normalised = [
            [row[c] if c < len(row) else "" for c in range(col_count)]
            for row in grid
        ]

        lines: list[str] = []
        header = normalised[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in normalised[1:]:
            if row == header:
                continue  # Skip repeated header rows (e.g. multi-page tables)
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _format_page_range(self, page_range: OCRPageRange) -> str:
        """Format a zero-based range as a one-based inclusive display range."""
        return f"{page_range.start_page + 1}-{page_range.end_page}"

    def _detect_mime_type(self, path: Path) -> str:
        """Detect supported document MIME types using file signatures first."""
        with path.open("rb") as file:
            header = file.read(16)

        if not header:
            raise ValueError(f"OCR input is empty: {path}")

        if header.startswith(b"%PDF"):
            return PDF_MIME_TYPE
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            return "image/gif"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"

        guessed_type, _ = mimetypes.guess_type(path.name)
        if guessed_type in {"application/x-pdf", "application/acrobat"}:
            return PDF_MIME_TYPE
        if guessed_type in SUPPORTED_MIME_TYPES:
            return guessed_type

        raise ValueError(f"Unsupported OCR document type: {guessed_type or 'unknown'}")

    def _validate_file_size(self, path: Path, mime_type: str) -> None:
        """Reject oversized OCR inputs before base64 encoding."""
        self._validate_bytes_size(path.stat().st_size, mime_type, label="OCR input")

    def _validate_bytes_size(self, byte_count: int, mime_type: str, *, label: str) -> None:
        """Reject oversized OCR request payloads before base64 encoding."""
        max_bytes = self.MAX_PDF_BYTES if mime_type == PDF_MIME_TYPE else self.MAX_IMAGE_BYTES
        if byte_count > max_bytes:
            max_mb = max_bytes // (1024 * 1024)
            actual_mb = byte_count / (1024 * 1024)
            raise ValueError(
                f"{label} is too large: {actual_mb:.1f} MB exceeds {max_mb} MB limit"
            )

    @staticmethod
    def _estimate_pdf_page_count(path: Path) -> int | None:
        """Estimate PDF pages, falling back to a raw marker count for malformed files."""
        try:
            page_count = len(PdfReader(str(path)).pages)
            return page_count or None
        except Exception as exc:
            logger.debug(
                "Falling back to raw PDF page marker count",
                extra={"file_path": str(path), "ocr_error": str(exc)},
            )

        try:
            raw = path.read_bytes()
        except OSError:
            return None
        matches = re.findall(rb"/Type\s*/Page\b", raw)
        return len(matches) or None

    @staticmethod
    async def _build_data_uri(path: Path, mime_type: str) -> str:
        """Read file asynchronously and build a base64 data URI."""
        async with aiofiles.open(path, "rb") as f:
            file_bytes = await f.read()
        return GLMOCRTool._build_data_uri_from_bytes(file_bytes, mime_type)

    @staticmethod
    def _build_data_uri_from_bytes(file_bytes: bytes, mime_type: str) -> str:
        """Build a base64 data URI from an in-memory provider payload."""
        base64_content = base64.b64encode(file_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base64_content}"

    @staticmethod
    def _estimate_text_quality(text: str) -> float:
        """Return a heuristic text-quality score (0.0–1.0).

        This is a fallback for when the provider does not expose confidence.
        """
        if not text:
            return 0.0

        score = 1.0
        replacement_ratio = text.count("\ufffd") / len(text)
        control_ratio = sum(1 for char in text if ord(char) < 32 and char not in "\t\n\r") / len(
            text
        )

        score -= replacement_ratio * 0.8
        score -= control_ratio * 0.4

        if len(text.strip()) < 10:
            score -= 0.4
        elif len(text.strip()) < 50:
            score -= 0.15

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Module-level helpers for HTML table → Markdown conversion
# ---------------------------------------------------------------------------

class _CellInfo:
    """Parsed data for a single ``<td>`` / ``<th>`` cell."""

    __slots__ = ("text", "colspan", "rowspan", "is_header")

    def __init__(self, *, colspan: int = 1, rowspan: int = 1, is_header: bool = False):
        self.text: str = ""
        self.colspan = colspan
        self.rowspan = rowspan
        self.is_header = is_header


class _HTMLTableParser(HTMLParser):
    """Minimal HTML parser that extracts table rows and cells.

    Each element of :attr:`rows` is a list of :class:`_CellInfo` objects
    representing one ``<tr>``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[_CellInfo]] = []
        self._current_row: list[_CellInfo] | None = None
        self._current_cell: _CellInfo | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "tr":
            self._current_row = []
        elif tag_lower in ("td", "th"):
            attr_map = dict(attrs)
            colspan = max(1, int(attr_map.get("colspan", "1") or "1"))
            rowspan = max(1, int(attr_map.get("rowspan", "1") or "1"))
            self._current_cell = _CellInfo(
                colspan=colspan,
                rowspan=rowspan,
                is_header=tag_lower == "th",
            )

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("td", "th") and self._current_cell is not None:
            self._current_cell.text = re.sub(
                r"\s+", " ", self._current_cell.text,
            ).strip()
            if self._current_row is not None:
                self._current_row.append(self._current_cell)
            self._current_cell = None
        elif tag_lower == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.text += data


def _expand_spans(rows: list[list[_CellInfo]]) -> list[list[str]]:
    """Expand ``colspan`` / ``rowspan`` into a flat grid of string cells."""
    if not rows:
        return []

    # First pass: determine the grid width
    max_cols = 0
    for row in rows:
        cols = sum(cell.colspan for cell in row)
        if cols > max_cols:
            max_cols = cols
    if max_cols == 0:
        return []

    num_rows = len(rows)
    grid: list[list[str | None]] = [[None] * max_cols for _ in range(num_rows)]

    for r_idx, row in enumerate(rows):
        c_target = 0
        for cell in row:
            # Advance past cells already filled by prior rowspans
            while c_target < max_cols and grid[r_idx][c_target] is not None:
                c_target += 1
            if c_target >= max_cols:
                break
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    tr = r_idx + dr
                    tc = c_target + dc
                    if tr < num_rows and tc < max_cols:
                        grid[tr][tc] = cell.text
            c_target += cell.colspan

    # Replace any remaining None with empty string
    return [
        [cell if cell is not None else "" for cell in row]
        for row in grid
    ]
