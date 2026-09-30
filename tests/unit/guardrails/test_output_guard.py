"""Unit tests for the output guard (UT-010)."""

import pytest

from src.guardrails.output_guard import OutputGuard


@pytest.fixture
def guard() -> OutputGuard:
    """Provide a fresh output guard."""
    return OutputGuard()


def test_unit_010_output_guard_masks_pii_values(guard: OutputGuard) -> None:
    """Student email, NIM, and phone values must be masked; academic text untouched."""
    draft = (
        "Hubungi Budi Santoso di budi.santoso@example.com, "
        "NIM 163.4001.001, atau telepon 081234567890 untuk detail jadwal."
    )

    filtered = guard.filter_response(draft)

    assert "budi.santoso@example.com" not in filtered
    assert "[email disamarkan]" in filtered
    assert "163.4001.001" not in filtered
    assert "[NIM disamarkan]" in filtered
    assert "081234567890" not in filtered
    assert "[nomor telepon disamarkan]" in filtered

    academic_text = (
        "Syarat kelulusan mencakup 144 SKS dengan nilai minimal C pada mata kuliah wajib."
    )
    assert guard.filter_response(academic_text) == academic_text
