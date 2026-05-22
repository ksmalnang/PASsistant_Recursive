"""Tests for hierarchical chunking and parent-child retrieval."""

from types import SimpleNamespace

import pytest
from qdrant_client.http.models import SparseVector

from src.services.ingestion_health import IngestionHealthCheck
from src.services.response_generation import CitationBuilder, ResponseContextBuilder
from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import DocumentType, DocumentUpload
from src.utils.tools.hierarchical_chunking import HierarchicalChunker
from src.utils.tools.parent_store import ParentChunkStore
from src.utils.tools.vector_store import VectorStoreTools
from src.utils.vector_store.indexing import IndexingOperations

SAMPLE_POLICY_TEXT = """
Tata Tertib Siswa 2025

BAB I Ketentuan Umum

Pasal 1 Definisi

Ayat (1) Sekolah adalah lingkungan pembelajaran resmi.

Ayat (2) Tata tertib ini merujuk pada Lampiran A.

BAB II Kehadiran Siswa

Pasal 3 Absensi

Ayat (1) Siswa wajib hadir tepat waktu.

Ayat (2) Izin tidak hadir maksimal 3 hari dan mengikuti Pasal 4.

1. Surat izin orang tua
2. Surat dokter

Tabel Sanksi
| Hari | Sanksi |
| 1 | Teguran |

Lampiran A Formulir Izin

Isi formulir izin sebelum absen.
""".strip()


def _build_document(extracted_text: str = SAMPLE_POLICY_TEXT) -> DocumentUpload:
    return DocumentUpload(
        document_id="doc-001",
        filename="tata_tertib_siswa.pdf",
        file_path="C:/tmp/tata_tertib_siswa.pdf",
        document_type=DocumentType.OTHER,
        mime_type="application/pdf",
        file_size=1024,
        extracted_text=extracted_text,
    )


class FakeCache:
    """In-memory stand-in for Redis-backed cache operations."""

    def __init__(self):
        self.json_values: dict[str, object] = {}
        self.set_values: dict[str, set[str]] = {}
        self.deleted_prefixes: list[str] = []

    def get_json(self, key: str):
        return self.json_values.get(key)

    def set_json(self, key: str, value, ttl_seconds: int | None = None) -> None:
        self.json_values[key] = value

    def add_to_set(
        self,
        key: str,
        *values: str,
        ttl_seconds: int | None = None,
    ) -> None:
        self.set_values.setdefault(key, set()).update(values)

    def get_set_members(self, key: str) -> set[str]:
        return set(self.set_values.get(key, set()))

    def delete_many(self, keys: list[str]) -> None:
        for key in keys:
            self.json_values.pop(key, None)
            self.set_values.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        self.deleted_prefixes.append(prefix)
        for key in list(self.json_values):
            if key.startswith(prefix):
                self.json_values.pop(key, None)


def test_hierarchical_chunker_parses_sections_and_atomic_blocks():
    """Chunker should create section parents and clause/list/table children."""
    chunker = HierarchicalChunker(parent_max_chars=5000, child_max_chars=5000)

    structured = chunker.chunk_document(_build_document())

    parent_section_ids = {parent.section_id for parent in structured.parent_chunks}
    assert "bab_i.pasal_1" in parent_section_ids
    assert "bab_ii.pasal_3" in parent_section_ids
    assert "lampiran_a" in parent_section_ids

    absensi_children = [
        chunk for chunk in structured.child_chunks if chunk.parent_id.endswith("bab_ii.pasal_3")
    ]
    assert any(
        chunk.metadata["chunk_type"] == "clause" and "Pasal 4" in chunk.text
        for chunk in absensi_children
    )
    assert any(
        chunk.metadata["chunk_type"] == "list" and chunk.metadata["is_atomic"]
        for chunk in absensi_children
    )
    assert any("pasal_4" in chunk.metadata["cross_refs"] for chunk in absensi_children)


def test_chunker_normalizes_ocr_heading_and_keeps_html_table_under_section():
    """OCR headings like ``6. 2`` should form their own section with table children."""
    chunker = HierarchicalChunker(parent_max_chars=5000, child_max_chars=5000)
    document = _build_document(
        """
        6.1 Matrik Kurikulum

        <table><tr><td>MK</td><td>IF21W0001</td></tr></table>

        6. 2 Peta Kurikulum Berdasarkan CPL PRODI

        <table border="1"><tr><td>CPL01</td><td>IF21W0101</td></tr></table>
        """.strip()
    )

    structured = chunker.chunk_document(document)

    parent_by_section = {parent.section_id: parent for parent in structured.parent_chunks}
    assert "pasal_6_1" in parent_by_section
    assert "pasal_6_2" in parent_by_section
    assert "CPL01" in parent_by_section["pasal_6_2"].text
    assert "CPL01" not in parent_by_section["pasal_6_1"].text

    cpl_children = [
        chunk for chunk in structured.child_chunks if chunk.parent_id.endswith("pasal_6_2")
    ]
    assert any(
        chunk.metadata["chunk_type"] == "table"
        and chunk.metadata["is_atomic"]
        and "CPL01" in chunk.text
        for chunk in cpl_children
    )


def test_chunker_splits_inline_ocr_headings_into_distinct_sections():
    """Section headings embedded mid-block should not collapse into one chapter parent."""
    chunker = HierarchicalChunker(parent_max_chars=5000, child_max_chars=5000)
    document = _build_document(
        """
        3 Profil Lulusan & Rumusan Capaian Pembelajaran Lulusan (CPL)
        Profil Lulusan & Rumusan Capaian Pembelajaran Lulusan (CPL)
        ## 3.1 Profil Lulusan
        Profil Lulusan terdiri dari dua hal.
        Tabel 1. Profil Lulusan dan deskripsinya
        <table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>
        ## 3.2 Perumusan CPL
        Tabel 2. Capaian Pembelajaran Lulusan Program Studi
        <table><tr><td>CPL01</td><td>Sikap</td></tr></table>
        """.strip()
    )

    structured = chunker.chunk_document(document)
    parent_by_section = {parent.section_id: parent for parent in structured.parent_chunks}

    assert "bab_3.pasal_3_1" in parent_by_section
    assert "bab_3.pasal_3_2" in parent_by_section
    assert "CPL01" not in parent_by_section["bab_3.pasal_3_1"].text
    assert "PL1" not in parent_by_section["bab_3.pasal_3_2"].text


def test_chunker_cleans_toc_leader_suffix_from_heading_title():
    """Heading detection should tolerate markdown markers and dotted TOC suffixes."""
    chunker = HierarchicalChunker()

    assert chunker._detect_heading("# 6. 2 Peta Kurikulum") == (
        "section",
        "6_2",
        "Peta Kurikulum",
        2,
    )
    assert chunker._detect_heading("6.2 PETA KURIKULUM BERDASARKAN CPL PRODI ...1") == (
        "section",
        "6_2",
        "PETA KURIKULUM BERDASARKAN CPL PRODI",
        2,
    )
    assert chunker._detect_heading("6.2 Peta Kurikulum Berdasarkan CPL PRODI") == (
        "section",
        "6_2",
        "Peta Kurikulum Berdasarkan CPL PRODI",
        2,
    )


def test_roman_decimal_heading_detection():
    """Roman numeral academic headings should map to the right structural depth."""
    chunker = HierarchicalChunker()
    test_cases = [
        ("IV.9.3. Evaluasi Kinerja akademik dan Eksekusi Akhir Masa Studi", "subsection"),
        ("III.2.1 Ketentuan Cuti Akademik", "subsection"),
        ("IV.9 Evaluasi Akademik", "section"),
        ("II.4 Registrasi Mahasiswa", "section"),
        ("IV Evaluasi Akademik", "chapter"),
        ("I Pendahuluan", "chapter"),
    ]

    for text, expected_type in test_cases:
        result = chunker._detect_heading(text)
        assert result is not None, f"Failed to detect heading: {text}"
        assert result[0] == expected_type


def test_wadek_style_document_produces_chunks():
    """Roman-numbered academic-guide structures should not collapse into one appendix chunk."""
    chunker = HierarchicalChunker(parent_max_chars=5000, child_max_chars=5000)
    document = _build_document(
        """
        I Pendahuluan

        Penjelasan umum.

        I.1 Tujuan

        Dokumen ini menjelaskan ketentuan akademik.

        II Status Mahasiswa

        II.4 Registrasi Mahasiswa

        Heregistrasi dilakukan setiap semester.

        III Ketentuan Akademik

        III.2 Cuti Akademik

        Mahasiswa wajib mengajukan cuti resmi.

        III.2.1 Persyaratan Cuti Akademik

        Persyaratan cuti akademik dijelaskan pada bagian ini.

        IV Evaluasi Akademik

        IV.9 Evaluasi Kinerja Akademik

        Evaluasi dilakukan secara berkala.

        IV.9.3. Evaluasi Kinerja akademik dan Eksekusi Akhir Masa Studi

        Mahasiswa yang tidak aktif selama 4 semester berturut-turut tanpa cuti resmi
        dapat dikenakan eksekusi akhir masa studi.

        IV.9.4 Sanksi Akademik Lainnya

        Ketentuan sanksi lain.
        """.strip()
    )

    structured = chunker.chunk_document(document)

    assert len(structured.parent_chunks) >= 5
    assert any(
        parent.section_id.endswith("subbagian_iv_9_3") for parent in structured.parent_chunks
    )


def test_ingestion_health_flags_sparse_indexing():
    """Health check should surface obviously broken low-density indexing output."""
    chunker = HierarchicalChunker()
    document = _build_document("Paragraf umum tanpa struktur heading yang jelas.")
    document.num_pages = 20

    structured = chunker.chunk_document(document)
    report = IngestionHealthCheck().validate(document, structured)

    issue_codes = {issue.code for issue in report.issues}
    assert "LOW_CHUNK_DENSITY" in issue_codes
    assert "TOO_FEW_CHILD_CHUNKS" in issue_codes


def test_html_table_splits_into_table_child_parts():
    """Large HTML tables should remain table chunks split on row boundaries."""
    chunker = HierarchicalChunker(parent_max_chars=10000, child_max_chars=180)
    rows = "".join(
        f"<tr><td>CPL{i:02d}</td><td>IF21W{i:04d}</td><td>Mata Kuliah {i}</td></tr>"
        for i in range(1, 8)
    )
    document = _build_document(
        f"""
        6.2 Peta Kurikulum Berdasarkan CPL PRODI

        <table border="1"><tr><th>CPL</th><th>Kode</th><th>MK</th></tr>{rows}</table>
        """.strip()
    )

    structured = chunker.chunk_document(document)
    table_children = [
        chunk for chunk in structured.child_chunks if chunk.metadata["chunk_type"] == "table"
    ]

    assert len(table_children) > 1
    assert all(chunk.metadata["is_atomic"] for chunk in table_children)
    assert {chunk.metadata["table_part"] for chunk in table_children} == set(
        range(1, len(table_children) + 1)
    )
    assert any("CPL07" in chunk.text for chunk in table_children)


def test_response_context_renders_child_evidence_before_heading_only_parent():
    """Useful child table rows should lead context when parent text is heading-only."""
    builder = ResponseContextBuilder()

    context_lines = builder._build_retrieval_context(
        [
            {
                "filename": "kurikulum.pdf",
                "breadcrumb": "6.2 Peta Kurikulum Berdasarkan CPL PRODI",
                "score": 0.59,
                "text": (
                    "6.2 PETA KURIKULUM BERDASARKAN CPL PRODI\n\n"
                    "PETA KURIKULUM BERDASARKAN CPL PRODI"
                ),
                "matched_children": [
                    {
                        "chunk_id": "child-table",
                        "chunk_type": "table",
                        "score": 0.59,
                        "text": "<tr><td>CPL01</td><td>IF21W0101</td></tr>",
                        "source_locations": [{"page": 85, "bbox_2d": [1, 2, 3, 4]}],
                    }
                ],
            }
        ]
    )

    context = "\n".join(context_lines)
    assert "Matched child evidence" in context
    assert "CPL01" in context.split("Matched child evidence", 1)[1]
    assert "Parent section context" not in context


def test_response_context_omits_broad_parent_when_precise_child_table_exists():
    """Precise table matches should not be diluted by neighboring tables in the parent."""
    builder = ResponseContextBuilder()

    context_lines = builder._build_retrieval_context(
        [
            {
                "filename": "kurikulum.pdf",
                "breadcrumb": "3.1 Profil Lulusan",
                "score": 0.93,
                "text": (
                    "3.1 Profil Lulusan\n\n"
                    "Tabel 1. Profil Lulusan dan deskripsinya\n"
                    "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>\n\n"
                    "<table><tr><td>PL4</td><td>Memiliki wawasan global</td></tr></table>"
                ),
                "matched_children": [
                    {
                        "chunk_id": "table-1",
                        "chunk_type": "table",
                        "score": 0.93,
                        "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>",
                    }
                ],
            }
        ]
    )

    context = "\n".join(context_lines)
    assert "PL1" in context
    assert "PL4" not in context


def test_response_context_prefers_real_table_rows_over_caption_and_intro():
    """When table rows exist, they should outrank intro paragraphs and captions."""
    builder = ResponseContextBuilder()

    context_lines = builder._build_retrieval_context(
        [
            {
                "filename": "kurikulum.pdf",
                "breadcrumb": "3.1 Profil Lulusan",
                "score": 0.93,
                "text": "unused parent",
                "matched_children": [
                    {
                        "chunk_type": "paragraph",
                        "score": 0.94,
                        "text": "Profil Lulusan terdiri dari dua hal yaitu Profil Pekerjaan Lulusan Teknik Informatika.",
                    },
                    {
                        "chunk_type": "table",
                        "score": 0.93,
                        "chunk_id": "bab_3.pasal_3_1.table_1.part_1",
                        "text": "Tabel 1. Profil Lulusan dan deskripsinya",
                    },
                    {
                        "chunk_type": "table",
                        "score": 0.92,
                        "chunk_id": "bab_3.pasal_3_1.table_2.part_1",
                        "text": (
                            "<table><tr><td>No</td><td>Profil Lulusan(PL)</td></tr>"
                            "<tr><td>PL1</td><td>Professional bidang Informatika</td></tr>"
                            "<tr><td>PL2</td><td>Technopreneur</td></tr></table>"
                        ),
                    },
                ],
            }
        ]
    )

    matched_section = "\n".join(context_lines).split("Matched child evidence:\n", 1)[1]
    assert "PL1" in matched_section
    assert "PL2" in matched_section
    assert matched_section.index("PL1") < matched_section.index("Tabel 1. Profil Lulusan")


def test_response_context_prefers_earlier_table_parts_over_later_ones():
    """Earlier table chunks should lead when multiple parts from different tables match."""
    builder = ResponseContextBuilder()

    context_lines = builder._build_retrieval_context(
        [
            {
                "filename": "kurikulum.pdf",
                "breadcrumb": "3.1 Profil Lulusan",
                "score": 0.93,
                "text": "unused parent",
                "matched_children": [
                    {
                        "chunk_type": "table",
                        "score": 0.9301,
                        "chunk_id": "bab_3.pasal_3_1.table_3.part_2",
                        "text": "<table><tr><td>PL3</td><td>Akademisi</td></tr></table>",
                    },
                    {
                        "chunk_type": "table",
                        "score": 0.9299,
                        "chunk_id": "bab_3.pasal_3_1.table_2.part_1",
                        "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>",
                    },
                ],
            }
        ]
    )

    matched_section = "\n".join(context_lines).split("Matched child evidence:\n", 1)[1]
    assert matched_section.index("PL1") < matched_section.index("PL3")


def test_response_context_prefers_distinct_table_groups_over_second_part_of_same_table():
    """A split table should not consume all evidence slots ahead of the next table group."""
    builder = ResponseContextBuilder()

    context_lines = builder._build_retrieval_context(
        [
            {
                "filename": "kurikulum.pdf",
                "breadcrumb": "3.1 Profil Lulusan",
                "score": 0.93,
                "text": "unused parent",
                "matched_children": [
                    {
                        "chunk_type": "table",
                        "score": 0.95,
                        "chunk_id": "bab_3.pasal_3_1.table_2.part_1",
                        "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>",
                    },
                    {
                        "chunk_type": "table",
                        "score": 0.94,
                        "chunk_id": "bab_3.pasal_3_1.table_2.part_2",
                        "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika lanjutan</td></tr></table>",
                    },
                    {
                        "chunk_type": "table",
                        "score": 0.93,
                        "chunk_id": "bab_3.pasal_3_1.table_3.part_1",
                        "text": "<table><tr><td>PL2</td><td>Technopreneur</td></tr></table>",
                    },
                ],
            }
        ]
    )

    matched_section = "\n".join(context_lines).split("Matched child evidence:\n", 1)[1]
    assert "PL2" in matched_section
    assert "lanjutan" not in matched_section


def test_build_reranker_document_prefers_matched_child_evidence_over_broad_parent():
    """Reranker payloads should emphasize matched child tables instead of adjacent parent text."""
    vector_tools = VectorStoreTools()

    document = vector_tools._build_reranker_document(
        {
            "breadcrumb": "3.1 Profil Lulusan",
            "parent_text": (
                "3.1 Profil Lulusan\n\n"
                "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>\n\n"
                "<table><tr><td>PL4</td><td>Memiliki wawasan global</td></tr></table>"
            ),
            "matched_children": [
                {
                    "chunk_type": "table",
                    "score": 0.91,
                    "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>",
                }
            ],
        }
    )

    assert "Matched child evidence" in document
    assert "PL1" in document
    assert "PL4" not in document


def test_build_reranker_document_prefers_real_table_rows_over_caption_and_intro():
    """Reranker payload should lead with row-bearing table chunks when available."""
    vector_tools = VectorStoreTools()

    document = vector_tools._build_reranker_document(
        {
            "breadcrumb": "3.1 Profil Lulusan",
            "matched_children": [
                {
                    "chunk_type": "paragraph",
                    "score": 0.94,
                    "text": "Profil Lulusan terdiri dari dua hal yaitu Profil Pekerjaan Lulusan Teknik Informatika.",
                },
                {
                    "chunk_type": "table",
                    "score": 0.93,
                    "chunk_id": "bab_3.pasal_3_1.table_1.part_1",
                    "text": "Tabel 1. Profil Lulusan dan deskripsinya",
                },
                {
                    "chunk_type": "table",
                    "score": 0.92,
                    "chunk_id": "bab_3.pasal_3_1.table_2.part_1",
                    "text": (
                        "<table><tr><td>No</td><td>Profil Lulusan(PL)</td></tr>"
                        "<tr><td>PL1</td><td>Professional bidang Informatika</td></tr>"
                        "<tr><td>PL2</td><td>Technopreneur</td></tr></table>"
                    ),
                },
            ],
        }
    )

    matched_section = document.split("Matched child evidence:\n", 1)[1]
    assert "PL1" in matched_section
    assert "PL2" in matched_section
    assert matched_section.index("PL1") < matched_section.index("Tabel 1. Profil Lulusan")


def test_build_reranker_document_prefers_earlier_table_parts_over_later_ones():
    """Reranker payload ordering should preserve earlier table parts before later matches."""
    vector_tools = VectorStoreTools()

    document = vector_tools._build_reranker_document(
        {
            "breadcrumb": "3.1 Profil Lulusan",
            "matched_children": [
                {
                    "chunk_type": "table",
                    "score": 0.9301,
                    "chunk_id": "bab_3.pasal_3_1.table_3.part_2",
                    "text": "<table><tr><td>PL3</td><td>Akademisi</td></tr></table>",
                },
                {
                    "chunk_type": "table",
                    "score": 0.9299,
                    "chunk_id": "bab_3.pasal_3_1.table_2.part_1",
                    "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>",
                },
            ],
        }
    )

    matched_section = document.split("Matched child evidence:\n", 1)[1]
    assert matched_section.index("PL1") < matched_section.index("PL3")


def test_build_reranker_document_prefers_distinct_table_groups_over_second_part_of_same_table():
    """Reranker payload should include the next table group before a second part of the same table."""
    vector_tools = VectorStoreTools()

    document = vector_tools._build_reranker_document(
        {
            "breadcrumb": "3.1 Profil Lulusan",
            "matched_children": [
                {
                    "chunk_type": "table",
                    "score": 0.95,
                    "chunk_id": "bab_3.pasal_3_1.table_2.part_1",
                    "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika</td></tr></table>",
                },
                {
                    "chunk_type": "table",
                    "score": 0.94,
                    "chunk_id": "bab_3.pasal_3_1.table_2.part_2",
                    "text": "<table><tr><td>PL1</td><td>Professional bidang Informatika lanjutan</td></tr></table>",
                },
                {
                    "chunk_type": "table",
                    "score": 0.93,
                    "chunk_id": "bab_3.pasal_3_1.table_3.part_1",
                    "text": "<table><tr><td>PL2</td><td>Technopreneur</td></tr></table>",
                },
            ],
        }
    )

    matched_section = document.split("Matched child evidence:\n", 1)[1]
    assert "PL2" in matched_section
    assert "lanjutan" not in matched_section


def test_build_reranker_document_prioritizes_policy_rule_before_schedule_and_sync_detail():
    """Policy reranker payloads should lead with the explicit rule before supporting details."""
    vector_tools = VectorStoreTools()

    document = vector_tools._build_reranker_document(
        {
            "breadcrumb": "IV.1.1. Tahap Pendaftaran dan Pembayaran DPP",
            "matched_children": [
                {
                    "chunk_type": "table",
                    "score": 0.93,
                    "chunk_id": "iv_1_1.table_1.part_1",
                    "text": (
                        "DPP dana pelaksanaan pendidikan : dibayarkan 4 kali cicilan "
                        "Cicilan I 25% : Juli"
                    ),
                },
                {
                    "chunk_type": "paragraph",
                    "score": 0.89,
                    "chunk_id": "iv_1_1.paragraph_1",
                    "text": (
                        "Perwalian dapat dilakukan setelah mahasiswa memenuhi "
                        "persyaratan administrasi pembayaran uang kuliah "
                        "yang disyaratkan dari DPP/SPP Tahun Akademik yang bersangkutan."
                    ),
                },
                {
                    "chunk_type": "paragraph",
                    "score": 0.88,
                    "chunk_id": "iv_1_1.paragraph_2",
                    "text": (
                        "Jika status pembayaran belum sinkron antara dpp.unpas.ac.id "
                        "dengan SITU 2.0 maka silahkan melaporkan status pembayaran."
                    ),
                },
            ],
        }
    )

    matched_section = document.split("Matched child evidence:\n", 1)[1]
    assert "Perwalian dapat dilakukan setelah" in matched_section
    assert "Cicilan I 25% : Juli" in matched_section
    assert "SITU 2.0" in matched_section
    assert matched_section.index("Perwalian dapat dilakukan setelah") < matched_section.index(
        "Cicilan I 25% : Juli"
    )


def test_citation_builder_uses_child_snippet_and_locations():
    """Citation previews should expose matched child rows, not heading-only parents."""
    citations = CitationBuilder().build(
        [
            {
                "document_id": "doc-001",
                "filename": "kurikulum.pdf",
                "doc_title": "Kurikulum",
                "breadcrumb": "6.2 Peta Kurikulum",
                "score": 0.59,
                "parent_id": "doc-001:pasal_6_2",
                "text": "6.2 PETA KURIKULUM",
                "matched_children": [
                    {
                        "chunk_id": "child-table",
                        "score": 0.59,
                        "text": "<tr><td>CPL01</td><td>IF21W0101</td></tr>",
                        "source_locations": [{"page": 85, "bbox_2d": [1, 2, 3, 4]}],
                    }
                ],
            }
        ]
    )

    assert citations[0].snippet == "<tr><td>CPL01</td><td>IF21W0101</td></tr>"
    assert citations[0].page == 86
    assert citations[0].source_locations == [{"page": 85, "bbox_2d": [1, 2, 3, 4]}]


def test_citation_builder_prefers_explicit_policy_child_over_higher_scored_schedule_table():
    """Policy citations should preview the normative rule, not just the schedule fragment."""
    citations = CitationBuilder().build(
        [
            {
                "document_id": "doc-001",
                "filename": "policy.pdf",
                "doc_title": "Panduan Akademik",
                "breadcrumb": "IV.1.1. Tahap Pendaftaran dan Pembayaran DPP",
                "score": 0.32,
                "parent_id": "doc-001:iv_1_1",
                "text": "Tahap Pendaftaran dan Pembayaran DPP",
                "matched_children": [
                    {
                        "chunk_id": "iv_1_1.table_1.part_1",
                        "chunk_type": "table",
                        "score": 0.93,
                        "text": (
                            "DPP dana pelaksanaan pendidikan : dibayarkan 4 kali cicilan "
                            "Cicilan I 25% : Juli"
                        ),
                        "source_locations": [{"page": 65, "bbox_2d": [1, 2, 3, 4]}],
                    },
                    {
                        "chunk_id": "iv_1_1.paragraph_1",
                        "chunk_type": "paragraph",
                        "score": 0.89,
                        "text": (
                            "Perwalian dapat dilakukan setelah mahasiswa memenuhi "
                            "persyaratan administrasi pembayaran uang kuliah "
                            "yang disyaratkan dari DPP/SPP Tahun Akademik yang bersangkutan."
                        ),
                        "source_locations": [{"page": 64, "bbox_2d": [5, 6, 7, 8]}],
                    },
                ],
            }
        ]
    )

    assert citations[0].snippet.startswith("Perwalian dapat dilakukan setelah")
    assert citations[0].page == 65


def test_source_locations_prefer_body_heading_over_toc_for_heading_only_text():
    """Heading-only chunks should avoid attaching table-of-contents locations."""
    chunker = HierarchicalChunker()
    locations = chunker._extract_source_locations(
        [
            [
                {
                    "label": "table_of_contents",
                    "bbox_2d": [1, 1, 2, 2],
                    "content": "6.2 Peta Kurikulum Berdasarkan CPL PRODI ...1",
                }
            ],
            [],
            [],
            [
                {
                    "label": "section_header",
                    "bbox_2d": [10, 10, 20, 20],
                    "content": "6.2 Peta Kurikulum Berdasarkan CPL PRODI",
                }
            ],
        ],
        "6.2 Peta Kurikulum Berdasarkan CPL PRODI",
    )

    assert locations == [{"page": 3, "bbox_2d": [10.0, 10.0, 20.0, 20.0]}]


def test_parent_chunk_store_persists_records(tmp_path):
    """Parent chunks should survive a store reload."""
    store_path = tmp_path / "parent_chunks.json"
    store = ParentChunkStore(store_path=store_path, cache=FakeCache())
    record = {
        "parent_id": "doc-001:bab_ii.pasal_3",
        "document_id": "doc-001",
        "filename": "tata_tertib_siswa.pdf",
        "section_id": "bab_ii.pasal_3",
        "text": "Pasal 3 Absensi",
        "metadata": {"breadcrumb": "BAB II > Pasal 3"},
    }

    store.put_many([record])

    reloaded = ParentChunkStore(store_path=store_path)
    assert reloaded.get(record["parent_id"]) == record


def test_parent_chunk_store_reads_from_cache(tmp_path):
    """Parent store should return cached parents without touching disk."""
    cache = FakeCache()
    store = ParentChunkStore(store_path=tmp_path / "parent_chunks.json", cache=cache)
    record = {
        "parent_id": "doc-001:bab_ii.pasal_3",
        "document_id": "doc-001",
        "filename": "tata_tertib_siswa.pdf",
        "section_id": "bab_ii.pasal_3",
        "text": "Pasal 3 Absensi",
        "metadata": {"breadcrumb": "BAB II > Pasal 3"},
    }

    store.put_many([record])

    assert cache.get_json("parent_chunk:doc-001:bab_ii.pasal_3") == record
    assert store.get(record["parent_id"]) == record


def test_parent_chunk_store_delete_document_clears_disk_loaded_cache(tmp_path):
    """Deleting a document should clear cached parents even if the cache index is missing."""
    cache = FakeCache()
    store = ParentChunkStore(store_path=tmp_path / "parent_chunks.json", cache=cache)
    record = {
        "parent_id": "doc-001:bab_ii.pasal_3",
        "document_id": "doc-001",
        "filename": "tata_tertib_siswa.pdf",
        "section_id": "bab_ii.pasal_3",
        "text": "Pasal 3 Absensi",
        "metadata": {"breadcrumb": "BAB II > Pasal 3"},
    }
    store.put_many([record])
    cache.set_values.clear()

    assert store.get(record["parent_id"]) == record

    store.delete_document("doc-001")

    assert store.get(record["parent_id"]) is None


@pytest.mark.asyncio
async def test_indexing_rolls_back_vectors_when_parent_store_fails():
    """Indexing should not leave vector chunks without parent retrieval context."""
    document = _build_document("Pasal 1 Definisi\n\nAyat (1) Sekolah resmi.")

    class FakeEmbeddings:
        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            assert any("Ayat (1) Sekolah resmi." in text for text in texts)
            return [[0.1, 0.2] for _ in texts]

    class FailingParentStore:
        def __init__(self):
            self.put_calls = 0
            self.deleted_documents: list[str] = []

        def put_many(self, parents: list[dict[str, object]]) -> None:
            self.put_calls += 1
            assert parents[0]["parent_id"] == "doc-001:pasal_1"
            raise RuntimeError("parent store unavailable")

        def delete_document(self, document_id: str) -> None:
            self.deleted_documents.append(document_id)

    class FakeCache:
        def __init__(self):
            self.deleted_prefixes: list[str] = []

        def delete_prefix(self, prefix: str) -> None:
            self.deleted_prefixes.append(prefix)

    class FakeClient:
        def __init__(self):
            self.upsert_calls = 0
            self.deleted_documents = 0

        def upsert(self, collection_name: str, points: list[object]) -> None:
            assert collection_name == "documents"
            assert points
            self.upsert_calls += 1

        def delete(self, collection_name: str, points_selector: object) -> None:
            assert collection_name == "documents"
            self.deleted_documents += 1

    class FakeIndexer(IndexingOperations):
        def __init__(self):
            self.chunker = HierarchicalChunker()
            self.embedding_model = "embedding-test"
            self.collection_name = "documents"
            self.client = FakeClient()
            self.parent_store = FailingParentStore()
            self.cache = FakeCache()

        def _get_embeddings(self) -> FakeEmbeddings:
            return FakeEmbeddings()

        def _ensure_collection_for_vector_size(self, vector_size: int) -> None:
            assert vector_size == 2

        def ensure_collection(self) -> None:
            return None

        def _supports_bm25_vectors(self) -> bool:
            return False

    indexer = FakeIndexer()

    with pytest.raises(RuntimeError, match="parent store unavailable"):
        await indexer.store_document_chunks(document)

    assert indexer.client.upsert_calls == 1
    assert indexer.client.deleted_documents == 2
    assert indexer.parent_store.put_calls == 1
    assert indexer.parent_store.deleted_documents == ["doc-001", "doc-001"]


@pytest.mark.asyncio
async def test_indexing_batches_large_upserts():
    """Large indexing jobs should be split into multiple Qdrant upsert requests."""
    document = _build_document(
        "\n\n".join(
            [f"Pasal {index} " + ("Ketentuan akademik " * 120) for index in range(1, 7)]
        )
    )

    class FakeEmbeddings:
        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

    class FakeParentStore:
        def put_many(self, parents: list[dict[str, object]]) -> None:
            assert parents

        def delete_document(self, document_id: str) -> None:
            assert document_id == "doc-001"

    class FakeCache:
        def delete_prefix(self, prefix: str) -> None:
            assert prefix == "search:"

    class FakeClient:
        def __init__(self):
            self.upsert_batches: list[int] = []

        def upsert(self, collection_name: str, points: list[object]) -> None:
            assert collection_name == "documents"
            self.upsert_batches.append(len(points))

        def delete(self, collection_name: str, points_selector: object) -> None:
            assert collection_name == "documents"

    class FakeIndexer(IndexingOperations):
        MAX_UPSERT_PAYLOAD_BYTES = 12000

        def __init__(self):
            self.chunker = HierarchicalChunker(child_max_chars=400, child_overlap_chars=0)
            self.embedding_model = "embedding-test"
            self.collection_name = "documents"
            self.client = FakeClient()
            self.parent_store = FakeParentStore()
            self.cache = FakeCache()

        def _get_embeddings(self) -> FakeEmbeddings:
            return FakeEmbeddings()

        def _ensure_collection_for_vector_size(self, vector_size: int) -> None:
            assert vector_size == 2

        def ensure_collection(self) -> None:
            return None

        def _supports_bm25_vectors(self) -> bool:
            return False

    indexer = FakeIndexer()

    chunk_ids = await indexer.store_document_chunks(document)

    assert chunk_ids
    assert len(indexer.client.upsert_batches) > 1
    assert sum(indexer.client.upsert_batches) == len(chunk_ids)


@pytest.mark.asyncio
async def test_search_similar_hydrates_parent_context_and_deduplicates(tmp_path, monkeypatch):
    """Child matches from the same parent should return one hydrated parent result."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()
    vector_tools.parent_store = ParentChunkStore(
        store_path=tmp_path / "parents.json",
        cache=vector_tools.cache,
    )
    vector_tools.parent_store.put_many(
        [
            {
                "parent_id": "doc-001:bab_ii.pasal_3",
                "document_id": "doc-001",
                "filename": "tata_tertib_siswa.pdf",
                "section_id": "bab_ii.pasal_3",
                "text": "Pasal 3 Absensi\n\nAyat (1) ...\n\nAyat (2) ...",
                "metadata": {
                    "breadcrumb": "BAB II Kehadiran Siswa > Pasal 3 Absensi",
                    "section_id": "bab_ii.pasal_3",
                    "chapter": "BAB II Kehadiran Siswa",
                    "section": "Pasal 3 Absensi",
                    "chunk_type": "section",
                    "cross_refs": ["pasal_4"],
                },
            }
        ]
    )

    class FakeEmbeddings:
        async def aembed_query(self, query: str) -> list[float]:
            assert query == "berapa batas izin tidak masuk sekolah?"
            return [0.1, 0.2]

    class FakeClient:
        def __init__(self):
            self.query_calls = 0

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            return SimpleNamespace(
                points_count=0,
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=2),
                        sparse_vectors={"bm25": object()},
                    )
                ),
            )

        def query_points(self, **_: object) -> SimpleNamespace:
            self.query_calls += 1
            return SimpleNamespace(points=[
                SimpleNamespace(
                    id="child-1",
                    score=0.93,
                    payload={
                        "text": "Ayat (2) Izin tidak hadir maksimal 3 hari.",
                        "parent_id": "doc-001:bab_ii.pasal_3",
                        "document_id": "doc-001",
                        "filename": "tata_tertib_siswa.pdf",
                        "document_type": "other",
                        "section_id": "bab_ii.pasal_3.ayat_2",
                        "breadcrumb": "BAB II > Pasal 3 > Ayat (2)",
                        "chunk_type": "clause",
                        "cross_refs": ["pasal_4"],
                    },
                ),
                SimpleNamespace(
                    id="child-2",
                    score=0.88,
                    payload={
                        "text": "1. Surat izin orang tua\n2. Surat dokter",
                        "parent_id": "doc-001:bab_ii.pasal_3",
                        "document_id": "doc-001",
                        "filename": "tata_tertib_siswa.pdf",
                        "document_type": "other",
                        "section_id": "bab_ii.pasal_3.list_3",
                        "breadcrumb": "BAB II > Pasal 3 > Persyaratan Izin",
                        "chunk_type": "list",
                        "cross_refs": [],
                    },
                ),
            ])

    monkeypatch.setattr(vector_tools, "ensure_collection", lambda: None)
    monkeypatch.setattr(vector_tools, "_get_embeddings", lambda: FakeEmbeddings())
    vector_tools.client = FakeClient()
    vector_tools.retrieval_strategy = "similarity"

    results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )

    assert len(results) == 1
    assert results[0]["section_id"] == "bab_ii.pasal_3"
    assert results[0]["text"].startswith("Pasal 3 Absensi")
    assert results[0]["score"] == pytest.approx(0.93)
    assert results[0]["final_score"] == pytest.approx(0.93)
    assert results[0]["retrieval_score"] == pytest.approx(0.93)
    assert len(results[0]["matched_children"]) == 2
    assert results[0]["matched_children"][0]["section_id"] == "bab_ii.pasal_3.ayat_2"
    assert results[0]["matched_children"][0]["retrieval_score"] == pytest.approx(0.93)

    cached_results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )
    assert cached_results == results
    assert vector_tools.client.query_calls == 1


@pytest.mark.asyncio
async def test_search_similar_supports_reranker_strategy(tmp_path, monkeypatch):
    """Reranker mode should reorder hydrated parent results and cache them."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()
    vector_tools.parent_store = ParentChunkStore(
        store_path=tmp_path / "parents.json",
        cache=vector_tools.cache,
    )
    vector_tools.parent_store.put_many(
        [
            {
                "parent_id": "doc-001:bab_i.pasal_1",
                "document_id": "doc-001",
                "filename": "tata_tertib_siswa.pdf",
                "section_id": "bab_i.pasal_1",
                "text": "Pasal 1 Definisi\n\nSekolah adalah lingkungan pembelajaran resmi.",
                "metadata": {
                    "breadcrumb": "BAB I Ketentuan Umum > Pasal 1 Definisi",
                    "section_id": "bab_i.pasal_1",
                    "chapter": "BAB I Ketentuan Umum",
                    "section": "Pasal 1 Definisi",
                    "chunk_type": "section",
                },
            },
            {
                "parent_id": "doc-001:bab_ii.pasal_3",
                "document_id": "doc-001",
                "filename": "tata_tertib_siswa.pdf",
                "section_id": "bab_ii.pasal_3",
                "text": "Pasal 3 Absensi\n\nIzin tidak hadir maksimal 3 hari.",
                "metadata": {
                    "breadcrumb": "BAB II Kehadiran Siswa > Pasal 3 Absensi",
                    "section_id": "bab_ii.pasal_3",
                    "chapter": "BAB II Kehadiran Siswa",
                    "section": "Pasal 3 Absensi",
                    "chunk_type": "section",
                },
            },
        ]
    )

    class FakeEmbeddings:
        async def aembed_query(self, query: str) -> list[float]:
            assert query == "berapa batas izin tidak masuk sekolah?"
            return [0.1, 0.2]

    class FakeClient:
        def __init__(self):
            self.query_calls = 0

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            return SimpleNamespace(
                points_count=0,
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=2),
                        sparse_vectors={"bm25": object()},
                    )
                ),
            )

        def query_points(self, **kwargs: object) -> SimpleNamespace:
            self.query_calls += 1
            return SimpleNamespace(points=[
                SimpleNamespace(
                    id="child-1",
                    score=0.93,
                    payload={
                        "text": "Sekolah adalah lingkungan pembelajaran resmi.",
                        "parent_id": "doc-001:bab_i.pasal_1",
                        "document_id": "doc-001",
                        "filename": "tata_tertib_siswa.pdf",
                        "document_type": "other",
                        "section_id": "bab_i.pasal_1.ayat_1",
                        "breadcrumb": "BAB I > Pasal 1 > Ayat (1)",
                        "chunk_type": "clause",
                        "cross_refs": [],
                    },
                ),
                SimpleNamespace(
                    id="child-2",
                    score=0.82,
                    payload={
                        "text": "Izin tidak hadir maksimal 3 hari.",
                        "parent_id": "doc-001:bab_ii.pasal_3",
                        "document_id": "doc-001",
                        "filename": "tata_tertib_siswa.pdf",
                        "document_type": "other",
                        "section_id": "bab_ii.pasal_3.ayat_2",
                        "breadcrumb": "BAB II > Pasal 3 > Ayat (2)",
                        "chunk_type": "clause",
                        "cross_refs": [],
                    },
                ),
            ])

    class FakeReranker:
        def __init__(self):
            self.calls = 0

        def rerank(self, *, query: str, documents: list[str]) -> list[float]:
            self.calls += 1
            assert query == "berapa batas izin tidak masuk sekolah?"
            assert len(documents) == 2
            return [0.08, 0.97]

    fake_reranker = FakeReranker()

    monkeypatch.setattr(vector_tools, "ensure_collection", lambda: None)
    monkeypatch.setattr(vector_tools, "_get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(vector_tools, "_get_reranker", lambda: fake_reranker)
    monkeypatch.setattr(vector_tools, "_supports_bm25_vectors", lambda: False)
    vector_tools.client = FakeClient()
    vector_tools.retrieval_strategy = "reranker"
    vector_tools.reranker_model = "jinaai/jina-reranker-v2-base-multilingual"

    results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )

    assert [result["section_id"] for result in results] == [
        "bab_ii.pasal_3",
        "bab_i.pasal_1",
    ]
    assert results[0]["reranker_score"] == pytest.approx(0.97)
    assert results[0]["final_score"] == pytest.approx(0.97)
    assert results[0]["retrieval_score"] == pytest.approx(0.82)
    assert results[0]["matched_children"][0]["retrieval_score"] == pytest.approx(0.82)
    assert results[0]["vector_score"] == pytest.approx(0.82)
    assert results[1]["reranker_score"] == pytest.approx(0.08)

    cached_results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )
    assert cached_results == results
    assert vector_tools.client.query_calls == 1
    assert fake_reranker.calls == 1


@pytest.mark.asyncio
async def test_reranker_uses_rrf_candidates_when_bm25_is_available(tmp_path, monkeypatch):
    """Reranker mode should rerank hybrid RRF candidates, not dense-only candidates."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()
    vector_tools.parent_store = ParentChunkStore(
        store_path=tmp_path / "parents.json",
        cache=vector_tools.cache,
    )
    vector_tools.parent_store.put_many(
        [
            {
                "parent_id": "doc-001:general",
                "document_id": "doc-001",
                "filename": "akademik.pdf",
                "section_id": "general",
                "text": "Pembayaran umum mahasiswa.",
                "metadata": {
                    "breadcrumb": "IV.1 Pembayaran Umum",
                    "section_id": "general",
                    "chunk_type": "section",
                },
            },
            {
                "parent_id": "doc-001:cuti",
                "document_id": "doc-001",
                "filename": "akademik.pdf",
                "section_id": "cuti",
                "text": "IV.9.1 Cuti Akademik\n\nBiaya administrasi cuti sebesar 5% dari DPP.",
                "metadata": {
                    "breadcrumb": "IV.9.1 Cuti Akademik",
                    "section_id": "cuti",
                    "chunk_type": "section",
                },
            },
        ]
    )

    class FakeEmbeddings:
        async def aembed_query(self, query: str) -> list[float]:
            assert query == "cuti akademik biaya DPP"
            return [0.1, 0.2]

    class FakeClient:
        def __init__(self):
            self.dense_calls = 0
            self.bm25_calls = 0

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            return SimpleNamespace(
                points_count=0,
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=2),
                        sparse_vectors={"bm25": object()},
                    )
                ),
            )

        def query_points(self, **kwargs: object) -> SimpleNamespace:
            if kwargs.get("using") == "bm25":
                self.bm25_calls += 1
                return SimpleNamespace(points=[
                    SimpleNamespace(
                        id="child-cuti",
                        score=12.0,
                        payload={
                            "text": "Biaya administrasi cuti sebesar 5% dari DPP.",
                            "parent_id": "doc-001:cuti",
                            "document_id": "doc-001",
                            "filename": "akademik.pdf",
                            "document_type": "policy",
                            "section_id": "cuti.p1",
                            "breadcrumb": "IV.9.1 Cuti Akademik",
                            "chunk_type": "paragraph",
                        },
                    )
                ])

            self.dense_calls += 1
            return SimpleNamespace(points=[
                SimpleNamespace(
                    id="child-general",
                    score=0.92,
                    payload={
                        "text": "Pembayaran umum mahasiswa.",
                        "parent_id": "doc-001:general",
                        "document_id": "doc-001",
                        "filename": "akademik.pdf",
                        "document_type": "policy",
                        "section_id": "general.p1",
                        "breadcrumb": "IV.1 Pembayaran Umum",
                        "chunk_type": "paragraph",
                    },
                )
            ])

    class FakeReranker:
        def rerank(self, *, query: str, documents: list[str]) -> list[float]:
            assert query == "cuti akademik biaya DPP"
            assert len(documents) == 2
            assert any("Cuti Akademik" in document for document in documents)
            return [0.91 if "Cuti Akademik" in document else 0.12 for document in documents]

    monkeypatch.setattr(vector_tools, "ensure_collection", lambda: None)
    monkeypatch.setattr(vector_tools, "_get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(vector_tools, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(
        vector_tools,
        "_build_bm25_vector",
        lambda _query, *, is_query: SparseVector(indices=[1], values=[1.0]),
    )
    vector_tools.client = FakeClient()
    vector_tools.retrieval_strategy = "reranker"
    vector_tools.reranker_model = "fake-reranker"
    vector_tools.bm25_vectors_enabled = True

    results = await vector_tools.search_similar(
        query="cuti akademik biaya DPP",
        top_k=2,
        score_threshold=0.4,
    )

    assert [result["section_id"] for result in results] == ["cuti", "general"]
    assert results[0]["reranker_score"] == pytest.approx(0.91)
    assert results[0]["bm25_score"] == pytest.approx(12.0)
    assert vector_tools.client.dense_calls == 1
    assert vector_tools.client.bm25_calls == 1


def test_score_reranker_documents_calls_model_with_keywords():
    """Reranker calls should match FastEmbed's query/documents API explicitly."""
    vector_tools = VectorStoreTools()

    class FakeReranker:
        def rerank(self, *, query: str, documents: list[str]) -> list[float]:
            assert query == "izin sakit"
            assert documents == ["doc a", "doc b"]
            return [0.2, 0.9]

    vector_tools.reranker = FakeReranker()

    assert vector_tools._score_reranker_documents(
        query="izin sakit",
        documents=["doc a", "doc b"],
    ) == [0.2, 0.9]


def test_score_reranker_documents_rejects_misaligned_scores():
    """Every reranker score must map back to exactly one candidate document."""
    vector_tools = VectorStoreTools()

    class FakeReranker:
        def rerank(self, *, query: str, documents: list[str]) -> list[float]:
            return [0.5]

    vector_tools.reranker = FakeReranker()

    with pytest.raises(RuntimeError, match="unexpected number of scores"):
        vector_tools._score_reranker_documents(
            query="izin sakit",
            documents=["doc a", "doc b"],
        )


def test_get_reranker_uses_remote_client_when_base_url_is_configured():
    """Remote reranking should use base URL, model, and API key settings."""
    vector_tools = VectorStoreTools()
    vector_tools.reranker_model = "jina-reranker"
    vector_tools.reranker_base_url = "https://rerank.example.test/v1"
    vector_tools.reranker_api_key = "secret"

    reranker = vector_tools._get_reranker()

    assert reranker.endpoint == "https://rerank.example.test/v1/rerank"
    assert reranker.model == "jina-reranker"
    assert reranker.api_key == "secret"


def test_remote_reranker_calls_endpoint_and_aligns_ranked_results(monkeypatch):
    """Remote reranker responses sorted by relevance should map back by index."""
    from src.utils.vector_store.reranker import RemoteReranker

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.12},
                ]
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.utils.vector_store.reranker.httpx.post", fake_post)

    reranker = RemoteReranker(
        base_url="https://rerank.example.test/v1",
        api_key="secret",
        model="jina-reranker",
    )

    scores = reranker.rerank(query="izin sakit", documents=["doc a", "doc b"])

    assert scores == [0.12, 0.91]
    assert captured["url"] == "https://rerank.example.test/v1/rerank"
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "model": "jina-reranker",
        "query": "izin sakit",
        "documents": ["doc a", "doc b"],
        "top_n": 2,
    }


@pytest.mark.asyncio
async def test_search_similar_supports_rrf_strategy(tmp_path, monkeypatch):
    """RRF mode should fuse dense and sparse rankings before parent hydration."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()
    vector_tools.parent_store = ParentChunkStore(
        store_path=tmp_path / "parents.json",
        cache=vector_tools.cache,
    )
    vector_tools.parent_store.put_many(
        [
            {
                "parent_id": "doc-001:bab_i.pasal_1",
                "document_id": "doc-001",
                "filename": "tata_tertib_siswa.pdf",
                "section_id": "bab_i.pasal_1",
                "text": "Pasal 1 Definisi\n\nSekolah adalah lingkungan pembelajaran resmi.",
                "metadata": {
                    "breadcrumb": "BAB I Ketentuan Umum > Pasal 1 Definisi",
                    "section_id": "bab_i.pasal_1",
                    "chapter": "BAB I Ketentuan Umum",
                    "section": "Pasal 1 Definisi",
                    "chunk_type": "section",
                },
            },
            {
                "parent_id": "doc-001:bab_ii.pasal_3",
                "document_id": "doc-001",
                "filename": "tata_tertib_siswa.pdf",
                "section_id": "bab_ii.pasal_3",
                "text": "Pasal 3 Absensi\n\nIzin tidak hadir maksimal 3 hari.",
                "metadata": {
                    "breadcrumb": "BAB II Kehadiran Siswa > Pasal 3 Absensi",
                    "section_id": "bab_ii.pasal_3",
                    "chapter": "BAB II Kehadiran Siswa",
                    "section": "Pasal 3 Absensi",
                    "chunk_type": "section",
                },
            },
            {
                "parent_id": "doc-001:lampiran_a",
                "document_id": "doc-001",
                "filename": "tata_tertib_siswa.pdf",
                "section_id": "lampiran_a",
                "text": "Lampiran A Formulir Izin",
                "metadata": {
                    "breadcrumb": "Lampiran A",
                    "section_id": "lampiran_a",
                    "chapter": "Lampiran A",
                    "section": "Formulir Izin",
                    "chunk_type": "section",
                },
            },
        ]
    )

    class FakeEmbeddings:
        async def aembed_query(self, query: str) -> list[float]:
            assert query == "berapa batas izin tidak masuk sekolah?"
            return [0.1, 0.2]

    class FakeClient:
        def __init__(self):
            self.dense_calls = 0
            self.bm25_calls = 0
            self.dense_kwargs: list[dict[str, object]] = []

        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            return SimpleNamespace(
                points_count=0,
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=2),
                        sparse_vectors={"bm25": object()},
                    )
                ),
            )

        def query_points(self, **kwargs: object) -> SimpleNamespace:
            if kwargs.get("using") == "bm25":
                self.bm25_calls += 1
                return SimpleNamespace(points=[
                    SimpleNamespace(
                        id="child-2",
                        score=11.0,
                        payload={
                            "text": "Izin tidak hadir maksimal 3 hari.",
                            "parent_id": "doc-001:bab_ii.pasal_3",
                            "document_id": "doc-001",
                            "filename": "tata_tertib_siswa.pdf",
                            "document_type": "other",
                            "section_id": "bab_ii.pasal_3.ayat_2",
                            "breadcrumb": "BAB II > Pasal 3 > Ayat (2)",
                            "chunk_type": "clause",
                            "cross_refs": [],
                        },
                    ),
                    SimpleNamespace(
                        id="child-3",
                        score=5.0,
                        payload={
                            "text": "Lampiran A Formulir Izin",
                            "parent_id": "doc-001:lampiran_a",
                            "document_id": "doc-001",
                            "filename": "tata_tertib_siswa.pdf",
                            "document_type": "other",
                            "section_id": "lampiran_a",
                            "breadcrumb": "Lampiran A",
                            "chunk_type": "section",
                            "cross_refs": [],
                        },
                    ),
                ])

            self.dense_calls += 1
            self.dense_kwargs.append(kwargs)
            return SimpleNamespace(points=[
                SimpleNamespace(
                    id="child-1",
                    score=0.93,
                    payload={
                        "text": "Sekolah adalah lingkungan pembelajaran resmi.",
                        "parent_id": "doc-001:bab_i.pasal_1",
                        "document_id": "doc-001",
                        "filename": "tata_tertib_siswa.pdf",
                        "document_type": "other",
                        "section_id": "bab_i.pasal_1.ayat_1",
                        "breadcrumb": "BAB I > Pasal 1 > Ayat (1)",
                        "chunk_type": "clause",
                        "cross_refs": [],
                    },
                ),
                SimpleNamespace(
                    id="child-3",
                    score=0.90,
                    payload={
                        "text": "Lampiran A Formulir Izin",
                        "parent_id": "doc-001:lampiran_a",
                        "document_id": "doc-001",
                        "filename": "tata_tertib_siswa.pdf",
                        "document_type": "other",
                        "section_id": "lampiran_a",
                        "breadcrumb": "Lampiran A",
                        "chunk_type": "section",
                        "cross_refs": [],
                    },
                ),
                SimpleNamespace(
                    id="child-2",
                    score=0.82,
                    payload={
                        "text": "Izin tidak hadir maksimal 3 hari.",
                        "parent_id": "doc-001:bab_ii.pasal_3",
                        "document_id": "doc-001",
                        "filename": "tata_tertib_siswa.pdf",
                        "document_type": "other",
                        "section_id": "bab_ii.pasal_3.ayat_2",
                        "breadcrumb": "BAB II > Pasal 3 > Ayat (2)",
                        "chunk_type": "clause",
                        "cross_refs": [],
                    },
                ),
            ])

    monkeypatch.setattr(vector_tools, "ensure_collection", lambda: None)
    monkeypatch.setattr(vector_tools, "_get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(
        vector_tools,
        "_build_bm25_vector",
        lambda _query, *, is_query: SparseVector(indices=[1, 2], values=[1.0, 1.0]),
    )
    vector_tools.client = FakeClient()
    vector_tools.retrieval_strategy = "rrf"
    vector_tools.bm25_vectors_enabled = True

    results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )

    assert [result["section_id"] for result in results] == [
        "bab_ii.pasal_3",
        "lampiran_a",
    ]
    assert results[0]["rrf_score"] == pytest.approx((1 / 63) + (1 / 61))
    assert results[0]["vector_score"] == pytest.approx(0.82)
    assert results[0]["bm25_score"] == pytest.approx(11.0)

    cached_results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )
    assert cached_results == results
    assert vector_tools.client.dense_calls == 1
    assert vector_tools.client.bm25_calls == 1
    assert "score_threshold" not in vector_tools.client.dense_kwargs[0]


@pytest.mark.asyncio
async def test_search_similar_filters_negative_reranker_results(tmp_path, monkeypatch):
    """Negative reranker scores should not be returned as context or citations."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()
    vector_tools.parent_store = ParentChunkStore(
        store_path=tmp_path / "parents.json",
        cache=vector_tools.cache,
    )
    vector_tools.parent_store.put_many(
        [
            {
                "parent_id": "doc-001:pasal_6_2",
                "document_id": "doc-001",
                "filename": "kurikulum.pdf",
                "section_id": "pasal_6_2",
                "text": "6.2 Peta Kurikulum\n\nCPL01 IF21W0101",
                "metadata": {
                    "breadcrumb": "6.2 Peta Kurikulum",
                    "section_id": "pasal_6_2",
                    "chunk_type": "section",
                },
            },
            {
                "parent_id": "doc-001:pasal_1_1",
                "document_id": "doc-001",
                "filename": "kurikulum.pdf",
                "section_id": "pasal_1_1",
                "text": "Unrelated text",
                "metadata": {
                    "breadcrumb": "1.1 Unrelated",
                    "section_id": "pasal_1_1",
                    "chunk_type": "section",
                },
            },
        ]
    )

    class FakeEmbeddings:
        async def aembed_query(self, query: str) -> list[float]:
            assert query == "apa isi peta kurikulum berdasarkan CPL?"
            return [0.1, 0.2]

    class FakeClient:
        def collection_exists(self, _name: str) -> bool:
            return True

        def get_collection(self, _name: str) -> SimpleNamespace:
            return SimpleNamespace(
                points_count=0,
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=SimpleNamespace(size=2),
                        sparse_vectors={"bm25": object()},
                    )
                ),
            )

        def query_points(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs.get("score_threshold") is None
            return SimpleNamespace(points=[
                SimpleNamespace(
                    id="child-good",
                    score=0.88,
                    payload={
                        "text": "CPL01 IF21W0101",
                        "parent_id": "doc-001:pasal_6_2",
                        "document_id": "doc-001",
                        "filename": "kurikulum.pdf",
                        "document_type": "curriculum",
                        "section_id": "pasal_6_2.table_1",
                        "breadcrumb": "6.2 Peta Kurikulum",
                        "chunk_type": "table",
                    },
                ),
                SimpleNamespace(
                    id="child-bad",
                    score=0.87,
                    payload={
                        "text": "Unrelated text",
                        "parent_id": "doc-001:pasal_1_1",
                        "document_id": "doc-001",
                        "filename": "kurikulum.pdf",
                        "document_type": "curriculum",
                        "section_id": "pasal_1_1",
                        "breadcrumb": "1.1 Unrelated",
                        "chunk_type": "paragraph",
                    },
                ),
            ])

    class FakeReranker:
        def rerank(self, *, query: str, documents: list[str]) -> list[float]:
            assert len(documents) == 2
            return [0.72, -0.02]

    monkeypatch.setattr(vector_tools, "ensure_collection", lambda: None)
    monkeypatch.setattr(vector_tools, "_get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(vector_tools, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(vector_tools, "_supports_bm25_vectors", lambda: False)
    vector_tools.client = FakeClient()
    vector_tools.retrieval_strategy = "reranker"
    vector_tools.reranker_model = "fake-reranker"

    results = await vector_tools.search_similar(
        query="apa isi peta kurikulum berdasarkan CPL?",
        top_k=2,
        score_threshold=0.4,
    )

    assert len(results) == 1
    assert results[0]["section_id"] == "pasal_6_2"
    assert results[0]["reranker_score"] == pytest.approx(0.72)


def test_rrf_final_filter_uses_relative_scores():
    """RRF filtering should keep only results close to the best fused score."""
    vector_tools = VectorStoreTools()
    vector_tools.retrieval_strategy = "rrf"

    filtered = vector_tools._filter_final_results(
        [
            {"section_id": "best", "rrf_score": 0.032},
            {"section_id": "close", "rrf_score": 0.020},
            {"section_id": "weak", "rrf_score": 0.010},
        ]
    )

    assert [result["section_id"] for result in filtered] == ["best", "close"]


def test_retrieval_confidence_keeps_clear_topical_match_without_warning():
    """A near-threshold score should not trigger fallback when the top hit is clearly topical."""
    node = RetrievalNode(retriever=SimpleNamespace())
    results = [
        {
            "score": 0.399,
            "breadcrumb": "IV.3. Penilaian > IV.3.3. Ujian Susulan",
            "text": "Ujian Susulan diberikan izin oleh Wakil Dekan I.",
            "matched_children": [
                {
                    "text": (
                        "Mahasiswa pada jadwal ujian sedang dirawat inap di rumah sakit "
                        "dapat mengikuti ujian susulan."
                    )
                }
            ],
        },
        {
            "score": 0.02,
            "breadcrumb": "IV.3.1. Ujian Tengah Semester",
            "text": "Ujian Tengah Semester.",
            "matched_children": [],
        },
    ]

    confidence, warning = node._score_retrieval_confidence(
        query="jadwal ujian UTS masuk rumah sakit bisa nyusul ujian batas lapor",
        results=results,
    )

    assert confidence < 0.45
    assert warning is None


def test_retrieval_confidence_warns_for_weak_off_topic_match():
    """Low-scoring context without enough topical overlap should still warn."""
    node = RetrievalNode(retriever=SimpleNamespace())
    results = [
        {
            "score": 0.399,
            "breadcrumb": "VIII.3. Prosedur Pengeluaran Dana",
            "text": "Dana kemahasiswaan digunakan untuk membiayai kegiatan.",
            "matched_children": [{"text": "Susunan pengurus lembaga kemahasiswaan."}],
        }
    ]

    _confidence, warning = node._score_retrieval_confidence(
        query="jadwal ujian UTS masuk rumah sakit bisa nyusul ujian batas lapor",
        results=results,
    )

    assert warning is not None
