"""Unit tests for testset citation verification."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.citations import (
    VerificationConfig,
    check_quote,
    clip,
    load_corpus,
    load_rows,
    normalize,
    row_to_dict,
    verify_rows,
)


class FakePoint:
    def __init__(self, point_id: str, payload: dict) -> None:
        self.id = point_id
        self.payload = payload


class FakeClient:
    """Serves one scroll page per call, so the pagination path is exercised."""

    def __init__(self, pages: list[list[FakePoint]]) -> None:
        self._pages = pages
        self.calls = 0

    def scroll(self, **kwargs: object) -> tuple[list[FakePoint], int | None]:
        page = self._pages[self.calls]
        self.calls += 1
        offset = self.calls if self.calls < len(self._pages) else None
        return page, offset


def _point(doc_id: str, text: str, index: int, filename: str = "pedoman.pdf") -> FakePoint:
    return FakePoint(
        f"{doc_id}-{index}",
        {
            "document_id": doc_id,
            "chunk_id": f"{doc_id}-{index}",
            "chunk_index": index,
            "filename": filename,
            "text": text,
        },
    )


def _corpus(*points: FakePoint):
    pages = [[point] for point in points] or [[]]
    return load_corpus(FakeClient(pages), "test")


def test_normalize_collapses_whitespace_and_unicode_noise() -> None:
    assert normalize("  Mahasiswa\u00a0wajib\n\nmengisi   formulir  ") == (
        "Mahasiswa wajib mengisi formulir"
    )


def test_clip_truncates_long_text() -> None:
    assert clip("kata " * 60).endswith("...")
    assert clip("kalimat pendek") == "kalimat pendek"


def test_exact_quote_matches_across_scroll_pages() -> None:
    corpus = _corpus(
        _point("doc_a", "Mahasiswa wajib mengisi formulir cuti akademik.", 0),
        _point("doc_b", "Cuti diberikan paling lama dua semester.", 0),
    )
    check = check_quote("Mahasiswa wajib mengisi formulir cuti akademik.", corpus)

    assert check.status == "match"
    assert check.doc_ids == ("doc_a",)


def test_quote_spanning_two_chunks_matches() -> None:
    corpus = _corpus(
        _point("doc_a", "Mahasiswa wajib mengisi formulir", 0),
        _point("doc_a", "cuti akademik paling lambat dua minggu sebelum perwalian.", 1),
    )
    check = check_quote(
        "Mahasiswa wajib mengisi formulir cuti akademik paling lambat dua minggu "
        "sebelum perwalian.",
        corpus,
    )

    assert check.status == "match"
    assert check.doc_ids == ("doc_a",)
    assert check.detail


def test_paraphrased_quote_is_reported_as_near_miss() -> None:
    corpus = _corpus(
        _point(
            "doc_a",
            "Mahasiswa yang akan mengajukan cuti akademik wajib mengisi formulir "
            "permohonan dan mendapat persetujuan dosen wali.",
            0,
        )
    )
    check = check_quote(
        "Mahasiswa yang berencana mengambil cuti akademik harus mengisi formulir "
        "permohonan dan mendapatkan izin dari dosen wali.",
        corpus,
    )

    assert check.status == "near"
    assert check.doc_ids == ("doc_a",)
    assert "closest in doc_a" in check.detail


def test_unrelated_quote_is_missing() -> None:
    corpus = _corpus(_point("doc_a", "Besaran UKT ditetapkan oleh rektor.", 0))

    assert check_quote("Jadwal ujian skripsi dimulai pada bulan Agustus.", corpus).status == (
        "missing"
    )


def test_empty_corpus_marks_quotes_missing() -> None:
    check = check_quote("Mahasiswa wajib mengisi formulir cuti akademik.", _corpus())

    assert check.status == "missing"
    assert check.detail == "collection has no indexed text"


def test_verify_rows_fills_doc_ids_and_flags_problems(tmp_path: Path) -> None:
    corpus = _corpus(_point("doc_a", "Mahasiswa wajib mengisi formulir cuti akademik.", 0))
    rows = [
        {
            "id": "q_001",
            "user_input": "Apa syarat cuti?",
            "reference": "Mengisi formulir.",
            "reference_contexts": ["Mahasiswa wajib mengisi formulir cuti akademik."],
            "metadata": {"answerable": True, "source_refs": [{"file": "pedoman.pdf", "page": 3}]},
        },
        {
            "id": "q_001",
            "user_input": "Pertanyaan duplikat?",
            "reference": "Jawaban.",
            "reference_contexts": ["Mahasiswa wajib mengisi formulir cuti akademik."],
            "metadata": {"answerable": True},
        },
        {
            "id": "q_002",
            "user_input": "Berapa biaya legalisir?",
            "reference": "Informasi tersebut tidak tersedia dalam dokumen yang tersedia.",
            "reference_contexts": [],
            "metadata": {"answerable": False},
        },
        {
            "id": "q_003",
            "user_input": "Kapan cuti diajukan?",
            "reference": "Jawaban.",
            "reference_contexts": [
                "Pengajuan cuti akademik harus dilakukan sebelum masa perwalian dimulai."
            ],
            "metadata": {
                "answerable": True,
                "source_refs": [{"file": "file_hantu.pdf", "page": 1}],
            },
        },
    ]
    path = _dataset(tmp_path, rows)
    result = verify_rows(load_rows(path), corpus, VerificationConfig(fuzz_threshold=100.0))

    assert result.counts["match"] == 2
    assert result.rows[0].metadata["source_doc_ids"] == ["doc_a"]
    assert "reference_context_verified" not in result.rows[0].metadata
    assert result.rows[2].metadata["answerable"] is False
    assert "source_doc_ids" not in result.rows[3].metadata
    assert result.rows[3].metadata["reference_context_verified"] is False
    assert any("appears 2 times" in problem for problem in result.problems["q_001"])
    assert any("reference_contexts[0]" in problem for problem in result.problems.get("q_003", []))
    assert any("not an indexed file" in problem for problem in result.problems.get("q_003", []))


def test_row_without_contexts_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    corpus = _corpus(_point("doc_a", "Mahasiswa wajib mengisi formulir cuti akademik.", 0))
    path = _dataset(
        tmp_path,
        [
            {
                "id": "q_001",
                "user_input": "Apa syarat cuti?",
                "reference": "Jawaban.",
                "reference_contexts": [],
                "metadata": {"answerable": True},
            }
        ],
    )
    result = verify_rows(load_rows(path), corpus)

    assert "reference_contexts is empty" in result.problems["q_001"][0]


def test_row_with_uncheckable_quote_is_reported(tmp_path: Path) -> None:
    corpus = _corpus(_point("doc_a", "Mahasiswa wajib mengisi formulir cuti akademik.", 0))
    path = _dataset(
        tmp_path,
        [
            {
                "id": "q_001",
                "user_input": "Apa syarat cuti?",
                "reference": "Jawaban.",
                "reference_contexts": ["Lihat pedoman."],
                "metadata": {"answerable": True},
            }
        ],
    )
    result = verify_rows(load_rows(path), corpus)

    assert "not verifiable" in result.problems["q_001"][0]
    assert result.counts["match"] == 0


def test_short_quote_does_not_hide_the_other_quotes_in_the_row(tmp_path: Path) -> None:
    corpus = _corpus(_point("doc_a", "Mahasiswa wajib mengisi formulir cuti akademik.", 0))
    path = _dataset(
        tmp_path,
        [
            {
                "id": "q_001",
                "user_input": "Apa syarat cuti?",
                "reference": "Jawaban.",
                "reference_contexts": [
                    "Mahasiswa wajib mengisi formulir cuti akademik.",
                    "Lihat pedoman.",
                ],
                "metadata": {"answerable": True},
            }
        ],
    )
    result = verify_rows(load_rows(path), corpus)

    assert result.counts["match"] == 1
    assert result.rows[0].metadata["source_doc_ids"] == ["doc_a"]
    assert "not verifiable" in result.problems["q_001"][0]


def test_fuzz_threshold_controls_near_miss_classification() -> None:
    corpus = _corpus(_point("doc_a", "Besaran UKT ditetapkan oleh rektor setiap semester.", 0))
    strict = VerificationConfig(fuzz_threshold=99.9)

    assert check_quote("Besaran UKT ditetapkan oleh rektor setiap tahun.", corpus).status == "near"
    assert check_quote(
        "Besaran UKT ditetapkan oleh rektor setiap tahun.", corpus, strict
    ).status == ("missing")


def test_row_to_dict_keeps_the_original_schema(tmp_path: Path) -> None:
    path = _dataset(
        tmp_path,
        [
            {
                "id": "q_001",
                "user_input": "Apa syarat cuti?",
                "reference": "Jawaban.",
                "reference_contexts": ["Kutipan."],
                "metadata": {"answerable": True, "source_refs": [{"file": "a.pdf"}]},
            }
        ],
    )
    rows = load_rows(path)

    assert row_to_dict(rows[0]) == {
        "id": "q_001",
        "user_input": "Apa syarat cuti?",
        "reference": "Jawaban.",
        "reference_contexts": ["Kutipan."],
        "metadata": {"answerable": True, "source_refs": [{"file": "a.pdf"}]},
    }


def _dataset(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "testset.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path
