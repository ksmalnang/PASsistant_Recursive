"""Persistent per-page OCR cache.

Each document gets one directory, {OCR_CACHE_DIR}/{document_id}/, holding:

- pages.json - the raw per-page OCR output (layout blocks + extracted text)
- meta.json - document/OCR metadata, written LAST

meta.json existing is what makes an entry valid: a crash between the two
writes leaves a directory without metadata, which every reader ignores. All
writes are atomic (temp file + rename) so a partially written JSON file never
becomes visible.

The backend is deliberately small and swappable: nothing outside OCRCache
does OCR-cache file I/O.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)

OCR_CACHE_SCHEMA_VERSION = 1
PAGES_FILENAME = "pages.json"
META_FILENAME = "meta.json"

_SAFE_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(slots=True)
class CachedPage:
    """One page of persisted OCR output."""

    page_index: int
    page: int
    status: str
    layout_details: list[dict[str, Any]] = field(default_factory=list)
    extracted_text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page": self.page,
            "status": self.status,
            "error": self.error,
            "layout_details": self.layout_details,
            "extracted_text": self.extracted_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedPage:
        page_index = int(data["page_index"])
        return cls(
            page_index=page_index,
            page=int(data.get("page", page_index + 1)),
            status=str(data.get("status", "unknown")),
            layout_details=list(data.get("layout_details") or []),
            extracted_text=str(data.get("extracted_text") or ""),
            error=data.get("error"),
        )


@dataclass(slots=True)
class OCRCacheMeta:
    """Document and OCR metadata for one cache entry."""

    document_id: str
    filename: str
    doc_title: str | None
    document_type: str
    file_type: str
    num_pages: int | None
    raw_file: str
    source_file_sha256: str
    ocr_provider: str
    ocr_model: str
    ocr_timestamp: str
    ocr_page_status: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = OCR_CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "filename": self.filename,
            "doc_title": self.doc_title,
            "document_type": self.document_type,
            "file_type": self.file_type,
            "num_pages": self.num_pages,
            "raw_file": self.raw_file,
            "source_file_sha256": self.source_file_sha256,
            "ocr_provider": self.ocr_provider,
            "ocr_model": self.ocr_model,
            "ocr_timestamp": self.ocr_timestamp,
            "ocr_page_status": self.ocr_page_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRCacheMeta:
        return cls(
            document_id=str(data["document_id"]),
            filename=str(data.get("filename", "")),
            doc_title=data.get("doc_title"),
            document_type=str(data.get("document_type", "other")),
            file_type=str(data.get("file_type", "application/octet-stream")),
            num_pages=data.get("num_pages"),
            raw_file=str(data.get("raw_file", "")),
            source_file_sha256=str(data.get("source_file_sha256", "")),
            ocr_provider=str(data.get("ocr_provider", "")),
            ocr_model=str(data.get("ocr_model", "")),
            ocr_timestamp=str(data.get("ocr_timestamp", "")),
            ocr_page_status=list(data.get("ocr_page_status") or []),
            schema_version=int(data.get("schema_version", OCR_CACHE_SCHEMA_VERSION)),
        )


@dataclass(slots=True)
class OCRCacheEntry:
    """A loaded cache entry: metadata plus its ordered pages."""

    meta: OCRCacheMeta
    pages: list[CachedPage]


class OCRCache:
    """File-backed store for raw per-page OCR output."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        resolved = cache_dir if cache_dir is not None else get_settings().ocr_cache_dir
        self.cache_dir = Path(resolved)

    def exists(self, document_id: str) -> bool:
        """Whether a complete cache entry (metadata present) exists."""
        path = self._safe_meta_path(document_id)
        return path is not None and path.is_file()

    def save(self, meta: OCRCacheMeta, pages: list[CachedPage]) -> None:
        """Write pages first and metadata last; metadata marks the entry complete."""
        entry_dir = self._entry_dir(meta.document_id)
        entry_dir.mkdir(parents=True, exist_ok=True)

        ordered = sorted(pages, key=lambda page: page.page_index)
        self._atomic_write_json(
            entry_dir / PAGES_FILENAME,
            {
                "schema_version": OCR_CACHE_SCHEMA_VERSION,
                "pages": [page.to_dict() for page in ordered],
            },
        )
        self._atomic_write_json(entry_dir / META_FILENAME, meta.to_dict())

    def load(self, document_id: str) -> OCRCacheEntry | None:
        """Load a complete entry, or None when it is missing or invalid."""
        meta_path = self._safe_meta_path(document_id)
        if meta_path is None:
            return None
        meta = self._read_json(meta_path)
        if meta is None:
            return None

        pages_path = meta_path.parent / PAGES_FILENAME
        pages_payload = self._read_json(pages_path)
        if pages_payload is None:
            logger.warning(
                "OCR cache metadata found without readable pages",
                extra={"document_id": document_id, "path": str(pages_path)},
            )
            return None

        try:
            meta_obj = OCRCacheMeta.from_dict(meta)
            pages = [CachedPage.from_dict(page) for page in pages_payload.get("pages", [])]
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Ignoring malformed OCR cache entry",
                extra={"document_id": document_id, "path": str(meta_path)},
            )
            return None
        return OCRCacheEntry(meta=meta_obj, pages=pages)

    def delete(self, document_id: str) -> None:
        """Remove a cache entry directory, ignoring missing entries."""
        path = self._safe_entry_dir(document_id)
        if path is None:
            return
        shutil.rmtree(path, ignore_errors=True)

    def find_by_filename(self, filename: str) -> list[str]:
        """Return document IDs whose cached metadata carries this exact filename."""
        matches = [meta.document_id for meta in self.list() if meta.filename == filename]
        return sorted(matches)

    def list(self) -> list[OCRCacheMeta]:
        """Return metadata for every complete cache entry."""
        if not self.cache_dir.is_dir():
            return []

        metas: list[OCRCacheMeta] = []
        for entry in sorted(self.cache_dir.iterdir()):
            if not entry.is_dir() or not _SAFE_DOCUMENT_ID.match(entry.name):
                continue
            meta = self._read_json(entry / META_FILENAME)
            if meta is None:
                continue
            try:
                metas.append(OCRCacheMeta.from_dict(meta))
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Ignoring malformed OCR cache metadata",
                    extra={"path": str(entry / META_FILENAME)},
                )
        return metas

    def _entry_dir(self, document_id: str) -> Path:
        path = self._safe_entry_dir(document_id)
        if path is None:
            raise ValueError(f"Unsafe OCR cache document id: {document_id!r}")
        return path

    def _safe_entry_dir(self, document_id: str) -> Path | None:
        if not _SAFE_DOCUMENT_ID.match(document_id):
            return None
        return self.cache_dir / document_id

    def _safe_meta_path(self, document_id: str) -> Path | None:
        entry_dir = self._safe_entry_dir(document_id)
        return None if entry_dir is None else entry_dir / META_FILENAME

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        """Write JSON through a temp file in the same directory, then rename."""
        tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable OCR cache file", extra={"path": str(path)})
            return None
        return data if isinstance(data, dict) else None
