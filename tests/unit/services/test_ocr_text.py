"""Characterization tests for the current OCR text and table-conversion behavior.

These lock in the exact output the ingestion pipeline produces today so the
reindex-from-cache work cannot silently change extracted text. The table
conversion is intentionally lossy: it flattens rowspan/colspan cells into
repeated values, which is why the raw layout blocks must be preserved.
"""

from __future__ import annotations

from typing import Any

from src.utils.tools.ocr import GLMOCRTool

SIMPLE_TABLE = "<table><tr><th>Nama</th><th>NIM</th></tr><tr><td>Adit</td><td>123</td></tr></table>"

MERGED_TABLE = (
    "<table>"
    '<tr><th colspan="2">Identitas</th><th>Nilai</th></tr>'
    '<tr><td rowspan="2">Adit</td><td>NIM</td><td>90</td></tr>'
    "<tr><td>Prodi</td><td>85</td></tr>"
    "</table>"
)


def _tool_without_client() -> GLMOCRTool:
    """Build a GLM-OCR tool without constructing an API client."""
    return GLMOCRTool.__new__(GLMOCRTool)


def _block(label: str, content: str) -> dict[str, Any]:
    return {
        "index": 0,
        "label": label,
        "bbox_2d": [0, 0, 100, 100],
        "content": content,
        "height": 10,
        "width": 10,
    }


def test_convert_simple_html_table_to_markdown() -> None:
    """A header + body table becomes a Markdown table."""
    assert GLMOCRTool._convert_html_table_to_markdown(SIMPLE_TABLE) == (
        "| Nama | NIM |\n| --- | --- |\n| Adit | 123 |"
    )


def test_convert_merged_table_flattens_spans() -> None:
    """rowspan/colspan are expanded by repeating the merged cell value."""
    assert GLMOCRTool._convert_html_table_to_markdown(MERGED_TABLE) == (
        "| Identitas | Identitas | Nilai |\n"
        "| --- | --- | --- |\n"
        "| Adit | NIM | 90 |\n"
        "| Adit | Prodi | 85 |"
    )


def test_convert_non_html_content_passes_through() -> None:
    """Content without a table is returned unchanged."""
    assert GLMOCRTool._convert_html_table_to_markdown("plain text") == "plain text"


def test_convert_broken_table_falls_back_to_text() -> None:
    """A table fragment without complete rows falls back to stripped text."""
    assert GLMOCRTool._convert_html_table_to_markdown("<table><tr><td>x") == "x"


def test_text_from_layout_details_converts_table_blocks_and_keeps_order() -> None:
    """Only table-labeled HTML blocks are converted; other blocks pass through."""
    page_layout = [
        _block("text", "Halo dunia"),
        _block("table", SIMPLE_TABLE),
        _block("text", ""),
        _block("text", "Penutup"),
    ]

    assert _tool_without_client()._text_from_layout_details(page_layout) == (
        "Halo dunia\n| Nama | NIM |\n| --- | --- |\n| Adit | 123 |\nPenutup"
    )


def test_text_from_layout_details_leaves_non_table_html_untouched() -> None:
    """HTML in a non-table block is not converted."""
    page_layout = [_block("text", "<b>Bold</b>")]

    assert _tool_without_client()._text_from_layout_details(page_layout) == "<b>Bold</b>"
