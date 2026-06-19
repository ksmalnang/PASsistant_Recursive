"""Post-ingestion health checks for structured document indexing."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.utils.state import DocumentUpload

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionIssue:
    """One validation finding from the ingestion health check."""

    severity: str
    code: str
    message: str


@dataclass(slots=True)
class IngestionStats:
    """Structured stats recorded for one ingested document."""

    context_record_count: int
    vector_record_count: int
    text_coverage: float
    pages: int | None
    extracted_chars: int
    indexed_chars: int


@dataclass(slots=True)
class IngestionReport:
    """Machine-readable ingestion diagnostics."""

    issues: list[IngestionIssue] = field(default_factory=list)
    stats: IngestionStats | None = None
    unrecognized_headings: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return not any(issue.severity == "ERROR" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return bool(self.issues or self.unrecognized_headings)

    def to_dict(self) -> dict:
        """Serialize the report into plain Python containers."""
        return asdict(self)


class IngestionHealthCheck:
    """Validate that document indexing produced a credible retrieval index."""

    MIN_RECORDS_PER_PAGE = 0.3
    MIN_TEXT_COVERAGE_RATIO = 0.15
    MAX_ORPHAN_TEXT_RATIO = 0.70
    MIN_VECTOR_RECORDS = 1
    _HEADING_SUSPECT_PATTERN = re.compile(
        r"^(?:#+\s*)?(?:[IVXLCDM]+(?:\.\d+)*|\d+(?:\.\d+)*)\.?\s+\S+"
    )

    def validate(
        self,
        document: DocumentUpload,
        structured_document: Any,
    ) -> IngestionReport:
        """Return a report describing indexing quality for one document."""
        issues: list[IngestionIssue] = []

        total_extracted = len(document.extracted_text or "")
        total_indexed = sum(len(chunk.text) for chunk in structured_document.vector_records)
        coverage = (total_indexed / total_extracted) if total_extracted > 0 else 0.0
        stats = IngestionStats(
            context_record_count=len(structured_document.context_records),
            vector_record_count=len(structured_document.vector_records),
            text_coverage=coverage,
            pages=document.num_pages,
            extracted_chars=total_extracted,
            indexed_chars=total_indexed,
        )

        if document.num_pages and document.num_pages > 2:
            ratio = len(structured_document.context_records) / document.num_pages
            if ratio < self.MIN_RECORDS_PER_PAGE:
                issues.append(
                    IngestionIssue(
                        severity="ERROR",
                        code="LOW_CHUNK_DENSITY",
                        message=(
                            f"Only {len(structured_document.context_records)} context records "
                            f"for {document.num_pages} pages (ratio={ratio:.2f}). "
                            "The indexing method may be too coarse for this document."
                        ),
                    )
                )

        if total_extracted > 0 and coverage < self.MIN_TEXT_COVERAGE_RATIO:
            issues.append(
                IngestionIssue(
                    severity="WARNING",
                    code="LOW_TEXT_COVERAGE",
                    message=f"Only {coverage:.0%} of extracted text was indexed.",
                )
            )

        if structured_document.context_records and total_indexed > 0:
            largest = max(len(parent.text) for parent in structured_document.context_records)
            if (largest / total_indexed) > self.MAX_ORPHAN_TEXT_RATIO:
                issues.append(
                    IngestionIssue(
                        severity="WARNING",
                        code="ORPHAN_CONCENTRATION",
                        message="Most indexed text is concentrated in a single context record.",
                    )
                )

        if len(structured_document.vector_records) < self.MIN_VECTOR_RECORDS:
            issues.append(
                IngestionIssue(
                    severity="ERROR",
                    code="TOO_FEW_VECTOR_RECORDS",
                    message=(
                        f"Only {len(structured_document.vector_records)} vector records were produced."
                    ),
                )
            )

        unrecognized_headings = self.detect_unrecognized_headings(
            text=document.extracted_text or "",
            recognized_headings=structured_document.recognized_headings,
        )
        report = IngestionReport(
            issues=issues,
            stats=stats,
            unrecognized_headings=unrecognized_headings,
        )
        self._log_report(document, report)
        return report

    def detect_unrecognized_headings(
        self,
        text: str,
        recognized_headings: list[str],
    ) -> list[str]:
        """Find structural-looking lines that were not parsed as headings."""
        normalized_recognized = {
            self._normalize_heading_text(heading) for heading in recognized_headings if heading
        }
        suspects: list[str] = []
        seen: set[str] = set()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or len(line) > 120:
                continue
            if not self._HEADING_SUSPECT_PATTERN.match(line):
                continue

            normalized = self._normalize_heading_text(line)
            if normalized in normalized_recognized or normalized in seen:
                continue

            suspects.append(line)
            seen.add(normalized)

        if suspects:
            logger.warning(
                "Detected potential unrecognized headings",
                extra={"count": len(suspects), "samples": suspects[:5]},
            )
        return suspects

    def _normalize_heading_text(self, text: str) -> str:
        """Normalize a heading-like line for equality checks."""
        normalized = text.strip()
        normalized = re.sub(r"^#+\s*", "", normalized)
        normalized = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip().lower()

    def _log_report(self, document: DocumentUpload, report: IngestionReport) -> None:
        """Emit structured logs for visibility into ingestion quality."""
        payload = {
            "document_id": document.document_id,
            "document_name": document.filename,
            "stats": asdict(report.stats) if report.stats else None,
            "issues": [asdict(issue) for issue in report.issues],
            "unrecognized_headings": report.unrecognized_headings[:5],
        }
        logger.info("Ingestion health report", extra=payload)
        for issue in report.issues:
            log_fn = logger.error if issue.severity == "ERROR" else logger.warning
            log_fn(
                "Ingestion health issue [%s] %s",
                issue.code,
                issue.message,
                extra={"document_id": document.document_id, "document_name": document.filename},
            )
