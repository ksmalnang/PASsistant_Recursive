"""Unit tests for dataset generation helpers."""

from __future__ import annotations

from pathlib import Path

from src.eval.generate import _resolve_sample_size, default_responses_path


def test_sample_size_scales_with_testset_size() -> None:
    assert _resolve_sample_size(None, 2) == 12
    assert _resolve_sample_size(None, 20) == 80
    assert _resolve_sample_size(None, 100) == 200


def test_explicit_sample_size_wins() -> None:
    assert _resolve_sample_size(5, 100) == 5


def test_default_responses_path() -> None:
    assert default_responses_path(Path("testset.jsonl")) == Path("testset.responses.json")
