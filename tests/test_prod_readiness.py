"""
Production readiness tests for the ingestion and retrieval pipeline.

These tests validate that the indexing and retrieval pipeline
correctly handles the academic document patterns found in the knowledge base
(Buku Panduan, Kurikulum IF, Buku Pedoman Kemahasiswaan).

Run with: uv run python -m pytest tests/test_prod_readiness.py -v
"""

from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import DocumentType, DocumentUpload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUKU_PANDUAN_PRODI_IF = """
## III.4. Program Studi Teknik Informatika

Program Studi Teknik Informatika

Semester I

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100101</td><td>Agama</td><td>MKU</td><td>2</td></tr><tr><td colspan="4">Jumlah SKS</td><td>19</td></tr></table>

Semester II

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100201</td><td>Pendidikan Pancasila</td><td>MKU</td><td>2</td></tr><tr><td colspan="4">Jumlah SKS</td><td>19</td></tr></table>

Semester V

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100501</td><td>Bahasa Indonesia</td><td>MKU</td><td>2</td></tr><tr><td>2</td><td>IF2100502</td><td>Internet of Things</td><td>MKK</td><td>2</td></tr><tr><td>3</td><td>IF2100503</td><td>Intelegensia Buatan</td><td>MKK</td><td>3</td></tr><tr><td colspan="4">Jumlah SKS</td><td>20</td></tr></table>

Semester VIII

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>Kelompok</td><td>SKS</td></tr><tr><td>1</td><td>IF2100801</td><td>Islam Dasar Ilmu</td><td>MKU</td><td>2</td></tr><tr><td colspan="4">Jumlah SKS</td><td>12</td></tr></table>

Pilihan 1

<table border="1"><tr><td>No</td><td>Kode</td><td>Matakuliah</td><td>SKS</td></tr><tr><td>1</td><td>IF2110501</td><td>Algoritma Optimasi</td><td>3</td></tr><tr><td>2</td><td>IF2110502</td><td>Data Warehouse</td><td>3</td></tr></table>
""".strip()


def _build_doc(text: str, filename: str = "test.pdf") -> DocumentUpload:
    return DocumentUpload(
        document_id="test-doc-001",
        filename=filename,
        file_path="C:/tmp/test.pdf",
        document_type=DocumentType.OTHER,
        mime_type="application/pdf",
        file_size=1024,
        extracted_text=text,
    )


# ---------------------------------------------------------------------------
# 1. Contextual Embedding Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Query Normalization Tests
# ---------------------------------------------------------------------------


class TestQueryNormalization:
    """Verify semester numeral normalization and query expansion."""

    def setup_method(self):
        self.node = RetrievalNode.__new__(RetrievalNode)

    def test_arabic_to_roman_semester_5(self):
        result = self.node._normalize_semester_numerals("matakuliah semester 5 teknik informatika")
        assert "semester V" in result

    def test_arabic_to_roman_semester_8(self):
        result = self.node._normalize_semester_numerals("jadwal semester 8")
        assert "semester VIII" in result

    def test_roman_to_arabic_semester_v(self):
        result = self.node._normalize_semester_numerals("matakuliah semester V teknik informatika")
        assert "semester 5" in result

    def test_roman_to_arabic_semester_iii(self):
        result = self.node._normalize_semester_numerals("kuliah semester III")
        assert "semester 3" in result

    def test_no_semester_unchanged(self):
        result = self.node._normalize_semester_numerals("syarat perwalian DPP")
        assert result == "syarat perwalian DPP"

    def test_expand_query_normalizes_semester(self):
        """_expand_query should produce a semester-normalized variant."""
        result = self.node._expand_query("matakuliah semester 5 teknik informatika")
        assert "V" in result or "semester V" in result.lower()

    def test_expand_query_does_not_add_semester_berturut(self):
        """Removed expansion should not appear."""
        result = self.node._expand_query("jadwal semester 5")
        assert "berturut" not in result


# ---------------------------------------------------------------------------
# 3. Retrieval Pipeline Integration Tests
# ---------------------------------------------------------------------------


class TestRetrievalPipeline:
    """Verify retrieval node behavior and confidence scoring."""

    def test_policy_scope_terms_are_specific(self):
        """Generic terms like 'bisa', 'boleh' should NOT be in policy scope."""
        node = RetrievalNode.__new__(RetrievalNode)
        assert "bisa" not in node._POLICY_SCOPE_TERMS
        assert "boleh" not in node._POLICY_SCOPE_TERMS
        assert "kapan" not in node._POLICY_SCOPE_TERMS
        assert "bulan" not in node._POLICY_SCOPE_TERMS

    def test_policy_scope_terms_include_financial(self):
        """Financial/policy terms should be in scope."""
        node = RetrievalNode.__new__(RetrievalNode)
        assert "dpp" in node._POLICY_SCOPE_TERMS
        assert "cicilan" in node._POLICY_SCOPE_TERMS
        assert "perwalian" in node._POLICY_SCOPE_TERMS
        assert "pembayaran" in node._POLICY_SCOPE_TERMS

    def test_looks_like_policy_scope_detects_dpp(self):
        node = RetrievalNode.__new__(RetrievalNode)
        assert node._looks_like_policy_scope("telat bayar DPP cicilan pertama")

    def test_looks_like_policy_scope_rejects_generic(self):
        node = RetrievalNode.__new__(RetrievalNode)
        assert not node._looks_like_policy_scope("apa saja matakuliah semester 5")

    def test_extract_keywords_removes_stopwords(self):
        node = RetrievalNode.__new__(RetrievalNode)
        result = node._extract_keywords("Kak, kalau aku telat bayar DPP bisa ikut perwalian nggak?")
        assert "kak" not in result
        assert "aku" not in result
        assert "nggak" not in result
        assert "telat" in result
        assert "perwalian" in result

    def test_score_threshold_is_0_2(self):
        """Score threshold should be 0.2 (not the old 0.4)."""
        # Verify the hardcoded threshold in the run method
        import inspect

        source = inspect.getsource(RetrievalNode.run)
        assert "score_threshold=0.2" in source


# ---------------------------------------------------------------------------
# 4. BM25 Configuration Tests
# ---------------------------------------------------------------------------


class TestBM25Configuration:
    """Verify BM25 stemmer is enabled for Indonesian text."""

    def test_bm25_stemmer_enabled(self):
        """BM25_DISABLE_STEMMER should be False for Indonesian morphology support."""
        from src.utils.vector_store.bm25 import BM25_DISABLE_STEMMER

        assert BM25_DISABLE_STEMMER is False


# ---------------------------------------------------------------------------
# 5. Document Management API Tests
# ---------------------------------------------------------------------------


class TestDocumentManagementAPI:
    """Verify document list and delete endpoints exist and are configured."""

    def test_list_documents_endpoint_exists(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.api import app
        from src.utils.vector_store import VectorStoreTools

        monkeypatch.setattr(VectorStoreTools, "list_documents", lambda self: [])

        client = TestClient(app)
        response = client.get("/documents")
        # Should not 404 — may return empty list or actual docs
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_delete_endpoint_returns_404_for_missing_file(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.api import app
        from src.utils.vector_store import VectorStoreTools

        monkeypatch.setattr(
            VectorStoreTools,
            "find_document_ids_by_filename",
            lambda self, filename: set(),
        )

        client = TestClient(app)
        response = client.delete("/documents/by-filename/nonexistent_file.pdf")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 6. Ingestion Health Check Tests
# ---------------------------------------------------------------------------
