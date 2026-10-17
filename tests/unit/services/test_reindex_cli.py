"""Tests for the reindex CLI, which shares the service layer with the API."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from src.services.document_locks import DocumentLockRegistry
from src.services.ocr_cache import OCRCache
from src.services.reindex import OCRReindexService
from tests.doubles import FakeIndexer
from tests.factories import seed_ocr_cache

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "reindex_from_ocr_cache.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reindex_from_ocr_cache", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides: object) -> object:
    values = {"document_id": None, "filename": None, "all": False, "rebuild_text": False}
    values.update(overrides)
    return type("Args", (), values)()


async def test_cli_reindexes_all_cached_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    cache = OCRCache(tmp_path / "cache")
    seed_ocr_cache(cache, document_id="doc-1")
    seed_ocr_cache(cache, document_id="doc-2")
    service = OCRReindexService(
        cache=cache,
        indexer=FakeIndexer(),
        locks=DocumentLockRegistry(),
    )
    monkeypatch.setattr("src.services.reindex.OCRReindexService", lambda: service)

    code = await cli.run(_args(all=True))

    output = capsys.readouterr().out
    assert code == 0
    assert "reindexed=2 skipped=0 failed=0" in output


async def test_cli_reports_skipped_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    cache = OCRCache(tmp_path / "cache")
    service = OCRReindexService(
        cache=cache,
        indexer=FakeIndexer(),
        locks=DocumentLockRegistry(),
    )
    monkeypatch.setattr("src.services.reindex.OCRReindexService", lambda: service)

    code = await cli.run(_args(document_id=["missing"]))

    output = capsys.readouterr().out
    assert code == 0
    assert "skipped=1" in output
    assert "no OCR cache for missing, re-upload required" in output


async def test_cli_unknown_filename_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    service = OCRReindexService(
        cache=OCRCache(tmp_path / "cache"),
        indexer=FakeIndexer(),
        locks=DocumentLockRegistry(),
    )
    monkeypatch.setattr("src.services.reindex.OCRReindexService", lambda: service)

    code = await cli.run(_args(filename="absent.pdf"))

    output = capsys.readouterr().out
    assert code == 1
    assert "no OCR cache for absent.pdf, re-upload required" in output
