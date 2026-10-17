"""Tests for the persistent per-page OCR cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.ocr_cache import (
    META_FILENAME,
    PAGES_FILENAME,
    CachedPage,
    OCRCache,
    OCRCacheMeta,
)

MERGED_TABLE = (
    '<table><tr><th colspan="2">Identitas</th></tr>'
    '<tr><td rowspan="2">Adit</td><td>NIM</td></tr></table>'
)


def _meta(document_id: str, filename: str = "handbook.pdf") -> OCRCacheMeta:
    return OCRCacheMeta(
        document_id=document_id,
        filename=filename,
        doc_title="Handbook",
        document_type="policy",
        file_type="application/pdf",
        num_pages=2,
        raw_file=f"{document_id}.pdf",
        source_file_sha256="abc123",
        ocr_provider="zhipu",
        ocr_model="glm-ocr",
        ocr_timestamp="2026-01-01T00:00:00+00:00",
        ocr_page_status=[
            {"page_index": 0, "status": "success", "text_length": 4, "layout_blocks": 1},
            {"page_index": 1, "status": "failed", "text_length": 0, "layout_blocks": 0},
        ],
    )


def _pages() -> list[CachedPage]:
    return [
        CachedPage(
            page_index=0,
            page=1,
            status="success",
            layout_details=[
                {
                    "index": 0,
                    "label": "table",
                    "bbox_2d": [1, 2, 3, 4],
                    "content": MERGED_TABLE,
                    "height": 10,
                    "width": 20,
                }
            ],
            extracted_text="| Identitas | Identitas |\n| --- | --- |",
        ),
        CachedPage(
            page_index=1,
            page=2,
            status="failed",
            layout_details=[],
            extracted_text="",
            error="provider timeout",
        ),
    ]


def test_roundtrip_preserves_raw_layout_details(tmp_path: Path) -> None:
    cache = OCRCache(cache_dir=tmp_path)
    cache.save(_meta("doc-1"), _pages())

    entry = cache.load("doc-1")

    assert entry is not None
    assert entry.meta.filename == "handbook.pdf"
    assert entry.meta.ocr_page_status[1]["status"] == "failed"
    assert [page.page_index for page in entry.pages] == [0, 1]
    assert entry.pages[0].layout_details[0]["content"] == MERGED_TABLE
    assert entry.pages[0].layout_details[0]["bbox_2d"] == [1, 2, 3, 4]
    assert entry.pages[0].extracted_text.startswith("| Identitas")
    assert entry.pages[1].status == "failed"
    assert entry.pages[1].error == "provider timeout"


def test_save_sorts_pages_by_index(tmp_path: Path) -> None:
    cache = OCRCache(cache_dir=tmp_path)
    cache.save(_meta("doc-1"), list(reversed(_pages())))

    payload = json.loads((tmp_path / "doc-1" / PAGES_FILENAME).read_text(encoding="utf-8"))

    assert [page["page_index"] for page in payload["pages"]] == [0, 1]


def test_meta_written_last_marks_entry_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = OCRCache(cache_dir=tmp_path)
    real_write = OCRCache._atomic_write_json

    def flaky(path: Path, payload: object) -> None:
        if path.name == META_FILENAME:
            raise OSError("disk full")
        real_write(path, payload)

    monkeypatch.setattr(OCRCache, "_atomic_write_json", staticmethod(flaky))

    with pytest.raises(OSError):
        cache.save(_meta("doc-1"), _pages())

    assert cache.exists("doc-1") is False
    assert cache.load("doc-1") is None
    assert not list((tmp_path / "doc-1").glob("*.tmp"))


def test_delete_removes_entry(tmp_path: Path) -> None:
    cache = OCRCache(cache_dir=tmp_path)
    cache.save(_meta("doc-1"), _pages())

    cache.delete("doc-1")

    assert cache.exists("doc-1") is False
    assert cache.load("doc-1") is None


def test_find_by_filename_matches_exact_names(tmp_path: Path) -> None:
    cache = OCRCache(cache_dir=tmp_path)
    cache.save(_meta("doc-1", "shared.pdf"), _pages())
    cache.save(_meta("doc-2", "shared.pdf"), _pages())
    cache.save(_meta("doc-3", "other.pdf"), _pages())

    assert cache.find_by_filename("shared.pdf") == ["doc-1", "doc-2"]
    assert cache.find_by_filename("missing.pdf") == []


def test_list_returns_all_metadata(tmp_path: Path) -> None:
    cache = OCRCache(cache_dir=tmp_path)
    cache.save(_meta("doc-1"), _pages())
    cache.save(_meta("doc-2"), _pages())

    assert {meta.document_id for meta in cache.list()} == {"doc-1", "doc-2"}


def test_unsafe_document_ids_are_ignored(tmp_path: Path) -> None:
    cache = OCRCache(cache_dir=tmp_path)

    assert cache.exists("../../etc/passwd") is False
    assert cache.load("..") is None
    cache.delete("..")
