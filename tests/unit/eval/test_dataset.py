"""Unit tests for testset reading, validation, and statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.dataset import DatasetError, dataset_stats, load_dataset, write_dataset
from src.eval.models import DatasetSample


def _sample(sample_id: str = "q_001") -> DatasetSample:
    return DatasetSample(
        id=sample_id,
        user_input="syarat daftar sidang skripsi apa aja?",
        reference="Lulus minimal 138 SKS dan sudah menyelesaikan seminar proposal.",
        reference_contexts=("Mahasiswa dapat mendaftar sidang skripsi apabila lulus 138 SKS.",),
        metadata={
            "topic": "skripsi",
            "question_type": "multi_doc",
            "answerable": True,
            "source_doc_ids": ["doc_12"],
            "source_doc_version": "2025-05-12",
            "author": "ksma",
            "dataset_version": "v1",
        },
    )


def _write(path: Path, rows: list[dict]) -> Path:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text, encoding="utf-8")
    return path


def test_round_trip_preserves_contexts_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "testset.jsonl"
    write_dataset(path, [_sample()])

    loaded = load_dataset(path)

    assert len(loaded) == 1
    assert loaded[0].user_input.startswith("syarat")
    assert loaded[0].reference_contexts == (
        "Mahasiswa dapat mendaftar sidang skripsi apabila lulus 138 SKS.",
    )
    assert loaded[0].metadata["question_type"] == "multi_doc"


def test_missing_required_field_is_reported_with_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.jsonl", [{"id": "q_001", "user_input": "Halo?"}])

    with pytest.raises(DatasetError) as excinfo:
        load_dataset(path)

    assert "reference" in excinfo.value.errors[0]
    assert "line 1" in excinfo.value.errors[0]


def test_invalid_context_and_metadata_are_reported(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.jsonl",
        [
            {"id": "q_001", "user_input": "Q?", "reference": "A.", "reference_contexts": [1]},
            {
                "id": "q_002",
                "user_input": "Q?",
                "reference": "A.",
                "metadata": {"question_type": "essay"},
            },
            {
                "id": "q_003",
                "user_input": "Q?",
                "reference": "A.",
                "metadata": {"answerable": "yes"},
            },
        ],
    )

    with pytest.raises(DatasetError) as excinfo:
        load_dataset(path)

    errors = " ".join(excinfo.value.errors)
    assert "reference_contexts[0]" in errors
    assert "question_type" in errors
    assert "answerable" in errors


def test_invalid_json_line_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "q_001", "user_input": "Q?"', encoding="utf-8")

    with pytest.raises(DatasetError) as excinfo:
        load_dataset(path)

    assert "invalid JSON" in excinfo.value.errors[0]


def test_stats_count_answerability_and_metadata() -> None:
    unanswerable = DatasetSample(
        id="q_003",
        user_input="Berapa biaya legalisir ijazah?",
        reference="Informasi tidak tersedia dalam dokumen.",
        metadata={"topic": "alumni", "answerable": False},
    )

    stats = dataset_stats([_sample("q_001"), _sample("q_002"), unanswerable])

    assert stats["samples"] == 3
    assert stats["answerable"] == {"True": 2, "False": 1}
    assert stats["question_type"] == {"multi_doc": 2}
    assert stats["with_reference_contexts"] == 2
