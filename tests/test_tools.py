"""
Unit tests for utility tools.

Tests document storage, student record management,
and vector store operations.
"""

import base64
import logging
import uuid
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

import pytest
from pypdf import PdfReader, PdfWriter
from qdrant_client.http.models import SparseVector

from src.config import get_settings
from src.utils.state import DocumentType, DocumentUpload, StudentRecord
from src.utils.tools import DocumentTools, GLMOCRTool, StudentTools, VectorStoreTools
from src.utils.tools.hierarchical_chunking import HierarchicalChunker
from src.utils.tools.ocr import OCRPageJob, OCRPageRange


class TestDocumentTools:
    """Test document storage operations."""

    def test_detect_document_type(self):
        """Should correctly identify document types from filenames."""
        tools = DocumentTools()

        assert tools._detect_document_type("transcript.pdf") == "transcript"
        assert tools._detect_document_type("my_passport.jpg") == "id_card"
        assert tools._detect_document_type("application_form.pdf") == "application"
        assert tools._detect_document_type("recommendation_letter.docx") == "recommendation"
        assert tools._detect_document_type("Kurikulum Teknik Informatika 2021.pdf") == "curriculum"
        assert tools._detect_document_type("RPS Machine Learning.docx") == "syllabus"
        assert tools._detect_document_type("sertifikat_kelulusan.pdf") == "certificate"
        assert tools._detect_document_type("tuition_invoice_2026.pdf") == "invoice"
        assert tools._detect_document_type("pedoman_kebijakan_akademik.pdf") == "policy"
        assert tools._detect_document_type("random_file.txt") == "other"

    def test_save_upload_does_not_use_reserved_logrecord_fields(self, tmp_path, caplog):
        """Saving uploads should not fail because of reserved logging extra keys."""
        tools = DocumentTools()
        tools.raw_dir = tmp_path / "raw"
        tools.raw_dir.mkdir(parents=True, exist_ok=True)

        with caplog.at_level(logging.INFO):
            document = tools.save_upload(b"example pdf bytes", "sample.pdf")

        assert document.filename == "sample.pdf"
        assert (tools.raw_dir / f"{document.document_id}.pdf").exists()
        assert "Document saved" in caplog.text

    def test_detect_document_type_does_not_misclassify_informatika_as_application(self):
        tools = DocumentTools()

        assert (
            tools._detect_document_type(
                "Kurikulum Teknik Informatika Unpas 2021 (versi Publik - Tanpa RPS)-pages.pdf"
            )
            == "curriculum"
        )

    def test_extract_from_text(self):
        """Should extract student data from raw text."""
        tools = StudentTools()

        text = """
        Name: John Doe
        Student ID: STU_12345678
        Email: john@example.com
        GPA: 3.85
        Program: Computer Science
        Major: Artificial Intelligence
        """

        result = tools.extract_from_text(text)

        assert result["full_name"] == "John Doe"
        assert result["student_id"] == "STU_12345678"
        assert result["email"] == "john@example.com"
        assert result["gpa"] == 3.85
        assert result["program"] == "Computer Science"
        assert result["major"] == "Artificial Intelligence"


class TestGLMOCRTool:
    """Test OCR payload construction and text quality scoring."""

    @staticmethod
    def _write_blank_pdf(path, pages: int) -> None:
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=72, height=72)
        with path.open("wb") as file:
            writer.write(file)

    def test_init_uses_layout_model_from_settings(self):
        tool = GLMOCRTool()
        assert tool.model == get_settings().GLM_LAYOUT_MODEL

    def test_serialize_layout_details_extracts_bbox(self):
        from types import SimpleNamespace

        detail = SimpleNamespace(
            index=0,
            label="paragraph",
            bbox_2d=[10.0, 20.0, 100.0, 50.0],
            content="Hello world",
            height=30,
            width=90,
        )
        page = [detail]
        layout_details = [page]

        result = GLMOCRTool._serialize_layout_details(layout_details)

        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0]["index"] == 0
        assert result[0][0]["label"] == "paragraph"
        assert result[0][0]["bbox_2d"] == [10.0, 20.0, 100.0, 50.0]
        assert result[0][0]["content"] == "Hello world"
        assert result[0][0]["height"] == 30
        assert result[0][0]["width"] == 90

    def test_detect_mime_type_prefers_pdf_signature_over_extension(self, tmp_path):
        tool = GLMOCRTool()
        path = tmp_path / "not_really_an_image.png"
        path.write_bytes(b"%PDF-1.7\nexample")

        assert tool._detect_mime_type(path) == "application/pdf"

    def test_validate_file_size_rejects_oversized_images(self, tmp_path, monkeypatch):
        tool = GLMOCRTool()
        path = tmp_path / "large.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\nxx")

        monkeypatch.setattr(GLMOCRTool, "MAX_IMAGE_BYTES", 1)

        with pytest.raises(ValueError, match="too large"):
            tool._validate_file_size(path, "image/png")

    def test_estimate_text_quality_penalizes_replacement_character(self):
        tool = GLMOCRTool()

        clean_score = tool._estimate_text_quality("Valid Chinese text:  " * 3)
        garbled_score = tool._estimate_text_quality("Valid text \ufffd\ufffd\ufffd")

        assert clean_score > garbled_score

    def test_estimate_text_quality_returns_zero_for_empty_string(self):
        tool = GLMOCRTool()
        assert tool._estimate_text_quality("") == 0.0

    def test_detect_mime_type_raises_for_empty_file(self, tmp_path):
        tool = GLMOCRTool()
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")

        with pytest.raises(ValueError, match="empty"):
            tool._detect_mime_type(path)

    def test_estimate_pdf_page_count_counts_page_objects(self, tmp_path):
        path = tmp_path / "sample.pdf"
        path.write_bytes(
            b"%PDF-1.7\n"
            b"1 0 obj << /Type /Pages /Kids [2 0 R 3 0 R] >> endobj\n"
            b"2 0 obj << /Type /Page /Parent 1 0 R >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 1 0 R >> endobj\n"
        )

        assert GLMOCRTool._estimate_pdf_page_count(path) == 2

    def test_build_pdf_page_batch_jobs_split_large_pdfs(self, tmp_path):
        tool = GLMOCRTool()
        path = tmp_path / "large.pdf"
        self._write_blank_pdf(path, 100)

        jobs = tool._build_pdf_page_batch_jobs(path, 101)

        assert len(jobs) == 2
        page_counts = []
        for job in jobs:
            prefix, encoded = job.data_uri.split(",", 1)
            assert prefix == "data:application/pdf;base64"
            chunk_bytes = base64.b64decode(encoded)
            page_counts.append(len(PdfReader(BytesIO(chunk_bytes)).pages))
        assert page_counts == [99, 1]

    def test_build_pdf_page_batch_jobs_cover_original_page_ranges(self, tmp_path):
        tool = GLMOCRTool()
        path = tmp_path / "large.pdf"
        self._write_blank_pdf(path, 100)

        jobs = tool._build_pdf_page_batch_jobs(path, 100)

        assert [(job.page_range.start_page, job.page_range.end_page) for job in jobs] == [
            (0, 99),
            (99, 100),
        ]
        covered_pages = [
            page
            for job in jobs
            for page in range(job.page_range.start_page, job.page_range.end_page)
        ]
        assert covered_pages == list(range(100))

    @pytest.mark.asyncio
    async def test_extract_layout_page_results_calls_once_per_job(self, monkeypatch):
        tool = GLMOCRTool()
        calls: list[str] = []

        async def fake_create(data_uri):
            calls.append(data_uri)
            return SimpleNamespace(md_results="", data_info=None, layout_details=None)

        monkeypatch.setattr(tool, "_create_layout_parsing_with_retry", fake_create)

        results = await tool._extract_layout_page_results(
            jobs=[
                OCRPageJob(
                    data_uri="uri-1",
                    page_range=OCRPageRange(0, 1),
                    attempt_level="page",
                ),
                OCRPageJob(
                    data_uri="uri-2",
                    page_range=OCRPageRange(1, 2),
                    attempt_level="page",
                ),
            ]
        )

        assert len(results) == 2
        assert calls == ["uri-1", "uri-2"]

    @pytest.mark.asyncio
    async def test_extract_text_merges_batched_pdf_results(self, tmp_path, monkeypatch):
        tool = GLMOCRTool()
        path = tmp_path / "large.pdf"
        path.write_bytes(b"%PDF-1.7\n")

        monkeypatch.setattr(tool, "_validate_file_size", lambda *_args: None)
        monkeypatch.setattr(tool, "_estimate_pdf_page_count", lambda _path: 101)

        async def fake_build_layout_request_jobs(**_kwargs):
            return [
                OCRPageJob(
                    data_uri="data:application/pdf;base64,chunk-1",
                    page_range=OCRPageRange(0, 1),
                    attempt_level="batch",
                ),
                OCRPageJob(
                    data_uri="data:application/pdf;base64,chunk-2",
                    page_range=OCRPageRange(100, 101),
                    attempt_level="batch",
                ),
            ]

        monkeypatch.setattr(
            tool,
            "_build_layout_request_jobs",
            fake_build_layout_request_jobs,
        )

        async def fake_create(data_uri):
            if data_uri == "data:application/pdf;base64,chunk-1":
                return SimpleNamespace(
                    md_results="page one fallback text",
                    data_info=SimpleNamespace(num_pages=101),
                    layout_details=[
                        [
                            SimpleNamespace(
                                index=0,
                                label="paragraph",
                                bbox_2d=[1, 2, 3, 4],
                                content="page one text",
                                height=2,
                                width=2,
                            )
                        ]
                    ],
                )
            assert data_uri == "data:application/pdf;base64,chunk-2"
            return SimpleNamespace(
                md_results="page one hundred one fallback text",
                data_info=SimpleNamespace(num_pages=101),
                layout_details=[
                    [
                        SimpleNamespace(
                            index=0,
                            label="paragraph",
                            bbox_2d=[5, 6, 7, 8],
                            content="page one hundred one text",
                            height=2,
                            width=2,
                        )
                    ]
                ],
            )

        monkeypatch.setattr(tool, "_create_layout_parsing_with_retry", fake_create)

        result = await tool.extract_text(str(path))

        assert result.text == "page one text\n\npage one hundred one text"
        assert result.num_pages == 101
        assert result.parsed_pages == 2
        assert result.failed_pages == []
        assert result.ocr_warnings == ["OCR parsed 2 of 101 pages successfully."]
        assert result.layout_details is not None
        assert len(result.layout_details) == 101
        assert result.layout_details[100][0]["content"] == "page one hundred one text"

    @pytest.mark.asyncio
    async def test_failed_batch_falls_back_to_subranges_and_pages(self, tmp_path, monkeypatch):
        tool = GLMOCRTool()
        path = tmp_path / "sample.pdf"
        self._write_blank_pdf(path, 3)

        calls: list[tuple[str, str]] = []

        async def fake_create(data_uri):
            prefix, encoded = data_uri.split(",", 1)
            assert prefix == "data:application/pdf;base64"
            page_count = len(PdfReader(BytesIO(base64.b64decode(encoded))).pages)
            attempt_level = "batch" if page_count == 3 else "page"
            calls.append((attempt_level, str(page_count)))
            if page_count > 1:
                raise TimeoutError("batch timeout")
            page_attempts = sum(1 for _, count in calls if count == "1")
            if page_attempts == 1:
                return SimpleNamespace(
                    md_results="",
                    data_info=SimpleNamespace(num_pages=1),
                    layout_details=[
                        [
                            SimpleNamespace(
                                index=0,
                                label="paragraph",
                                bbox_2d=[1, 2, 3, 4],
                                content="page zero text",
                                height=2,
                                width=2,
                            )
                        ]
                    ],
                )
            raise TimeoutError("page timeout")

        monkeypatch.setattr(tool, "_create_layout_parsing_with_retry", fake_create)
        jobs = tool._build_pdf_page_batch_jobs(path, 3)

        page_results = await tool._extract_layout_page_results(jobs=jobs)
        result = tool._build_ocr_result_from_pages(
            page_results=page_results,
            total_pages=3,
        )

        assert result.text == "page zero text"
        assert result.parsed_pages == 1
        assert result.failed_pages == [1, 2]
        assert result.ocr_warnings
        assert [page["status"] for page in result.page_results] == [
            "success",
            "failed",
            "failed",
        ]

    @pytest.mark.asyncio
    async def test_ocr_result_raises_when_zero_pages_parse(self, tmp_path, monkeypatch):
        tool = GLMOCRTool()
        path = tmp_path / "sample.pdf"
        self._write_blank_pdf(path, 2)

        async def fake_create(_data_uri):
            raise TimeoutError("provider unavailable")

        monkeypatch.setattr(tool, "_create_layout_parsing_with_retry", fake_create)

        with pytest.raises(ValueError, match="no pages were parsed successfully"):
            await tool.extract_text(str(path))


class TestHTMLTableParser:
    """Tests for HTML table → Markdown conversion in GLMOCRTool."""

    def test_simple_table_to_markdown(self):
        html = (
            "<table>"
            "<tr><td>Name</td><td>Score</td></tr>"
            "<tr><td>Alice</td><td>90</td></tr>"
            "<tr><td>Bob</td><td>85</td></tr>"
            "</table>"
        )
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        assert "| Name | Score |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 90 |" in result
        assert "| Bob | 85 |" in result

    def test_table_with_th_headers(self):
        html = (
            "<table>"
            "<thead><tr><th>Mata Kuliah</th><th>SKS</th></tr></thead>"
            "<tbody>"
            "<tr><td>Algoritma</td><td>3</td></tr>"
            "<tr><td>Basis Data</td><td>3</td></tr>"
            "</tbody>"
            "</table>"
        )
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        assert "| Mata Kuliah | SKS |" in result
        assert "| Algoritma | 3 |" in result
        assert "| Basis Data | 3 |" in result

    def test_table_with_colspan(self):
        html = (
            "<table>"
            "<tr><td colspan='2'>Merged Header</td></tr>"
            "<tr><td>A</td><td>B</td></tr>"
            "</table>"
        )
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        lines = result.strip().split("\n")
        assert lines[0].count("|") >= 3
        assert "Merged Header" in lines[0]
        assert "| A | B |" in result

    def test_table_with_rowspan(self):
        html = (
            "<table>"
            "<tr><td>Header</td><td>Value</td></tr>"
            "<tr><td rowspan='2'>Semester 1</td><td>Algo</td></tr>"
            "<tr><td>DB</td></tr>"
            "</table>"
        )
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        lines = [l for l in result.strip().split("\n") if not l.startswith("| ---")]
        semester_lines = [l for l in lines if "Semester 1" in l]
        assert len(semester_lines) == 2

    def test_whitespace_and_special_chars_in_cells(self):
        html = (
            "<table>"
            "<tr><td>  Spaced  </td><td>Line\nBreak</td></tr>"
            "<tr><td>Normal</td><td>OK</td></tr>"
            "</table>"
        )
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        assert "| Spaced | Line Break |" in result

    def test_malformed_html_falls_back_to_stripped_text(self):
        html = "<table><tr><td>Only cell"
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        assert "Only cell" in result

    def test_non_table_html_returns_stripped_text(self):
        html = "<div>Not a table <b>at all</b></div>"
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        assert "Not a table" in result

    def test_empty_table_falls_back(self):
        html = "<table></table>"
        result = GLMOCRTool._convert_html_table_to_markdown(html)
        assert isinstance(result, str)

    def test_text_from_layout_details_routes_table_blocks(self):
        """Table-labeled blocks should be converted; others kept as-is."""
        tool = GLMOCRTool()
        page_layout = [
            {"label": "title", "content": "Document Title"},
            {
                "label": "table",
                "content": (
                    "<table>"
                    "<tr><td>Col1</td><td>Col2</td></tr>"
                    "<tr><td>A</td><td>B</td></tr>"
                    "</table>"
                ),
            },
            {"label": "paragraph", "content": "Some text after table."},
        ]
        result = tool._text_from_layout_details(page_layout)
        assert "Document Title" in result
        assert "| Col1 | Col2 |" in result
        assert "| A | B |" in result
        assert "Some text after table." in result
        assert "<table>" not in result
        assert "<td>" not in result

    def test_text_from_layout_details_preserves_non_html_table_content(self):
        """A table-labeled block without HTML should pass through unchanged."""
        tool = GLMOCRTool()
        page_layout = [
            {"label": "table", "content": "Plaintext table-like content"},
        ]
        result = tool._text_from_layout_details(page_layout)
        assert result == "Plaintext table-like content"


class TestHierarchicalChunker:
    """Test OCR text normalization and heading parsing."""

    def test_normalize_ocr_text_strips_fences(self):
        chunker = HierarchicalChunker()

        text = "```plaintext\n# 3 Profil Lulusan\n\nIsi\n```"

        assert chunker._normalize_ocr_text(text) == "# 3 Profil Lulusan\n\nIsi"

    def test_normalize_ocr_text_strips_layout_image_markers(self):
        chunker = HierarchicalChunker()

        text = "![](page=0,bbox=[281, 170, 396, 279])\n# 3 Profil Lulusan\n\nIsi"

        assert chunker._normalize_ocr_text(text) == "# 3 Profil Lulusan\n\nIsi"

    def test_resolve_title_skips_layout_artifacts_and_page_markers(self):
        chunker = HierarchicalChunker()
        document = DocumentUpload(
            document_id="doc-123",
            filename="Kurikulum Teknik Informatika.pdf",
            file_path="C:/tmp/Kurikulum Teknik Informatika.pdf",
            document_type=DocumentType.CURRICULUM,
            mime_type="application/pdf",
            file_size=1024,
            extracted_text=(
                "![](page=0,bbox=[281, 170, 396, 279])\n"
                "KPT 4.0 - 13\n"
                "# 3 Profil Lulusan & Rumusan Capaian Pembelajaran Lulusan (CPL)"
            ),
        )

        assert (
            chunker._resolve_title(document)
            == "3 Profil Lulusan & Rumusan Capaian Pembelajaran Lulusan (CPL)"
        )

    def test_numeric_markdown_heading_becomes_structural_node(self):
        chunker = HierarchicalChunker()

        heading = chunker._detect_heading("# 3 Profil Lulusan")

        assert heading == ("chapter", "3", "Profil Lulusan", 1)

    def test_chunk_metadata_uses_chunk_scoped_source_locations(self):
        chunker = HierarchicalChunker()
        document = DocumentUpload(
            document_id="doc-123",
            filename="sample.pdf",
            file_path="C:/tmp/sample.pdf",
            document_type=DocumentType.OTHER,
            mime_type="application/pdf",
            file_size=1024,
            extracted_text=(
                "![](page=0,bbox=[281, 170, 396, 279])\n"
                "# 3 Profil Lulusan\n\n"
                "## 3.1 Profil Lulusan\n\n"
                "Isi profil lulusan."
            ),
            layout_details=[
                [
                    {
                        "bbox_2d": [281.0, 170.0, 396.0, 279.0],
                        "content": "layout artifact",
                    },
                    {
                        "bbox_2d": [10.0, 20.0, 100.0, 50.0],
                        "content": "Isi profil lulusan.",
                    }
                ],
                [
                    {
                        "bbox_2d": [200.0, 220.0, 300.0, 250.0],
                        "content": "Unrelated page text",
                    }
                ]
            ],
        )

        structured = chunker.chunk_document(document)
        content_chunk = next(
            chunk for chunk in structured.child_chunks if chunk.text == "Isi profil lulusan."
        )

        assert "bbox_2d" not in structured.parent_chunks[0].metadata
        assert "bbox_2d" not in content_chunk.metadata
        assert content_chunk.metadata["source_locations"] == [
            {"page": 0, "bbox_2d": [10.0, 20.0, 100.0, 50.0]}
        ]


class TestVectorStoreTools:
    """Test vector-store helper behavior."""

    def test_build_bm25_vector_uses_document_and_query_paths(self, monkeypatch):
        tool = VectorStoreTools()

        class FakeSparseEmbedding:
            def __init__(self, indices: list[int], values: list[float]):
                self.indices = indices
                self.values = values

        class FakeBM25Embedder:
            def embed(self, *, documents):
                assert documents == ["dokumen kurikulum"]
                yield FakeSparseEmbedding([11, 22], [0.4, 0.7])

            def query_embed(self, *, query):
                assert query == "query kurikulum"
                yield FakeSparseEmbedding([22, 33], [1.0, 1.0])

        monkeypatch.setattr(tool, "_get_bm25_text_embedder", lambda: FakeBM25Embedder())

        document_vector = tool._build_bm25_vector("dokumen kurikulum", is_query=False)
        query_vector = tool._build_bm25_vector("query kurikulum", is_query=True)

        assert document_vector == SparseVector(indices=[11, 22], values=[0.4, 0.7])
        assert query_vector == SparseVector(indices=[22, 33], values=[1.0, 1.0])

    def test_build_point_id_returns_uuid(self):
        tool = VectorStoreTools()

        point_id = tool._build_point_id("0324ebdc-231c-441e-aaa0-8dfcb7184407", "root.p1_1")

        assert str(uuid.UUID(point_id)) == point_id

    def test_build_child_payload_preserves_logical_chunk_id(self):
        tool = VectorStoreTools()
        document = type(
            "Document",
            (),
            {
                "document_id": "doc-123",
                "filename": "sample.pdf",
                "document_type": DocumentType.OTHER,
                "uploaded_at": datetime.now(UTC),
            },
        )()
        chunk = type(
            "Chunk",
            (),
            {
                "text": "sample text",
                "metadata": {"parent_id": "parent-1"},
            },
        )()

        payload = tool._build_child_payload(
            document=document,
            chunk=chunk,
            chunk_id="root.p1_1",
            chunk_index=0,
            total_chunks=1,
        )

        assert payload["chunk_id"] == "root.p1_1"
        assert payload["document_id"] == "doc-123"

    def test_build_child_payload_drops_raw_layout_fields_and_keeps_source_locations(self):
        tool = VectorStoreTools()
        document = type(
            "Document",
            (),
            {
                "document_id": "doc-123",
                "filename": "sample.pdf",
                "document_type": DocumentType.OTHER,
                "uploaded_at": datetime.now(UTC),
            },
        )()
        chunk = type(
            "Chunk",
            (),
            {
                "text": "sample text",
                "metadata": {
                    "parent_id": "parent-1",
                    "bbox_2d": [[281.0, 170.0, 396.0, 279.0]],
                    "layout_details": [{"content": "raw layout"}],
                    "source_locations": [
                        {"page": 0, "bbox_2d": [281.0, 170.0, 396.0, 279.0]}
                    ],
                },
            },
        )()

        payload = tool._build_child_payload(
            document=document,
            chunk=chunk,
            chunk_id="root.p1_1",
            chunk_index=0,
            total_chunks=1,
        )

        assert "bbox_2d" not in payload
        assert "layout_details" not in payload
        assert payload["source_locations"] == [
            {"page": 0, "bbox_2d": [281.0, 170.0, 396.0, 279.0]}
        ]

    def test_ensure_collection_recreates_empty_mismatched_collection(self):
        tool = VectorStoreTools()
        calls: list[tuple] = []

        class FakeClient:
            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return SimpleNamespace(
                    points_count=0,
                    config=SimpleNamespace(
                        params=SimpleNamespace(
                            vectors=SimpleNamespace(size=1536),
                            sparse_vectors={"bm25": object()},
                        )
                    ),
                )

            def delete_collection(self, name):
                calls.append(("delete", name))

            def create_collection(self, collection_name, vectors_config, sparse_vectors_config):
                calls.append(("create", collection_name, vectors_config.size))

        tool.client = FakeClient()
        tool._ensure_collection_for_vector_size(4096)

        assert calls == [
            ("delete", tool.collection_name),
            ("create", tool.collection_name, 4096),
        ]
        assert tool.vector_size == 4096

    def test_ensure_collection_raises_for_non_empty_mismatched_collection(self):
        tool = VectorStoreTools()

        class FakeClient:
            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return SimpleNamespace(
                    points_count=3,
                    config=SimpleNamespace(
                        params=SimpleNamespace(
                            vectors=SimpleNamespace(size=1536),
                            sparse_vectors={"bm25": object()},
                        )
                    ),
                )

        tool.client = FakeClient()

        with pytest.raises(RuntimeError, match="vector size mismatch"):
            tool._ensure_collection_for_vector_size(4096)

    @pytest.mark.asyncio
    async def test_search_similar_uses_query_embedding_size_for_collection_check(self, monkeypatch):
        tool = VectorStoreTools()
        tool.cache = SimpleNamespace(
            get_json=lambda _key: None,
            set_json=lambda _key, _value: None,
        )
        tool.parent_store = SimpleNamespace(get=lambda _parent_id: None)
        calls: list[int] = []

        class FakeEmbeddings:
            async def aembed_query(self, _query: str) -> list[float]:
                return [0.1, 0.2, 0.3, 0.4]

        monkeypatch.setattr(tool, "_get_embeddings", lambda: FakeEmbeddings())
        monkeypatch.setattr(tool, "_ensure_collection_for_vector_size", calls.append)
        monkeypatch.setattr(tool, "_search_candidates", lambda **_kwargs: [])

        results = await tool.search_similar("sample question")

        assert results == []
        assert calls == [4]

    @pytest.mark.asyncio
    async def test_search_similar_honors_requested_top_k(self, monkeypatch):
        tool = VectorStoreTools()
        tool.cache = SimpleNamespace(
            get_json=lambda _key: None,
            set_json=lambda _key, _value: None,
        )

        class FakeEmbeddings:
            async def aembed_query(self, _query: str) -> list[float]:
                return [0.1, 0.2]

        monkeypatch.setattr(tool, "_get_embeddings", lambda: FakeEmbeddings())
        monkeypatch.setattr(tool, "_ensure_collection_for_vector_size", lambda _size: None)
        monkeypatch.setattr(
            tool,
            "_search_candidates",
            lambda **_kwargs: [object()],
        )
        monkeypatch.setattr(
            tool,
            "_hydrate_results",
            lambda _results: {
                f"parent-{index}": {"section_id": f"section-{index}", "score": float(10 - index)}
                for index in range(5)
            },
        )
        monkeypatch.setattr(tool, "_rerank_results", lambda _query, results: results)
        monkeypatch.setattr(tool, "_filter_final_results", lambda results: results)

        results = await tool.search_similar("sample question", top_k=5)

        assert len(results) == 5
        assert [result["section_id"] for result in results] == [
            "section-0",
            "section-1",
            "section-2",
            "section-3",
            "section-4",
        ]


class TestStudentTools:
    """Test student record CRUD operations."""

    def test_create_and_get_record(self):
        """Should create and retrieve a student record."""
        tools = StudentTools()

        record = StudentRecord(
            full_name="Jane Smith",
            email="jane@example.com",
            gpa=3.9,
        )

        created = tools.create_record(record)

        assert created.student_id is not None
        assert created.student_id.startswith("STU_")

        # Retrieve
        retrieved = tools.get_record(created.student_id)
        assert retrieved is not None
        assert retrieved.full_name == "Jane Smith"

    def test_find_by_email(self):
        """Should find record by email."""
        tools = StudentTools()

        record = StudentRecord(
            full_name="Bob Wilson",
            email="bob@example.com",
        )
        tools.create_record(record)

        found = tools.find_by_email("bob@example.com")
        assert found is not None
        assert found.full_name == "Bob Wilson"

    def test_update_record(self):
        """Should update existing record."""
        tools = StudentTools()

        record = StudentRecord(full_name="Alice Brown", gpa=3.5)
        created = tools.create_record(record)

        updated = tools.update_record(created.student_id, {"gpa": 3.7})

        assert updated is not None
        assert updated.gpa == 3.7

    def test_delete_record(self):
        """Should delete a record."""
        tools = StudentTools()

        record = StudentRecord(full_name="To Delete")
        created = tools.create_record(record)

        success = tools.delete_record(created.student_id)
        assert success is True

        # Verify deletion
        assert tools.get_record(created.student_id) is None
