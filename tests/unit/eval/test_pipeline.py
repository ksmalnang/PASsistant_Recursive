"""Unit tests for fixture responses and citation stripping."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.models import DatasetSample
from src.eval.pipeline import apply_fixtures, load_fixtures, strip_citations


def _sample(sample_id: str = "q_001") -> DatasetSample:
    return DatasetSample(id=sample_id, user_input="Question?", reference="Reference.")


def test_strip_citations_removes_markers_and_sources_block() -> None:
    cleaned = strip_citations("Jawabannya 144 SKS [1].\n\nSources:\n- kurikulum.pdf")

    assert cleaned == "Jawabannya 144 SKS ."


def test_fixture_mode_uses_response_and_contexts_from_file() -> None:
    results = apply_fixtures(
        [_sample()],
        {"q_001": {"response": "Fresh answer [1].", "retrieved_contexts": ["a", "b"]}},
    )

    assert results[0].response == "Fresh answer [1]."
    assert results[0].retrieved_contexts == ["a", "b"]


def test_fixture_mode_requires_every_sample() -> None:
    with pytest.raises(ValueError, match="q_001"):
        apply_fixtures([_sample()], {})


def test_load_fixtures_reads_json_object(tmp_path: Path) -> None:
    path = tmp_path / "responses.json"
    path.write_text('{"q_001": {"response": "a"}}', encoding="utf-8")

    assert load_fixtures(path) == {"q_001": {"response": "a"}}
