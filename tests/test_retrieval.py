"""Tests for the retrieval pipeline (flat recursive chunks)."""

from types import SimpleNamespace

import pytest
from qdrant_client.http.models import SparseVector

from src.utils.nodes.retrieval import RetrievalNode
from src.utils.state import DocumentType, DocumentUpload
from src.utils.tools.vector_store import VectorStoreTools

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


@pytest.mark.asyncio
async def test_search_similar_returns_flat_chunk_results(monkeypatch):
    """Each retrieved Qdrant point should become a separate flat result."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()

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
            return SimpleNamespace(
                points=[
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
                ]
            )

    monkeypatch.setattr(vector_tools, "ensure_collection", lambda: None)
    monkeypatch.setattr(vector_tools, "_get_embeddings", lambda: FakeEmbeddings())
    vector_tools.client = FakeClient()
    vector_tools.retrieval_strategy = "similarity"

    results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )

    assert len(results) == 2
    assert results[0]["section_id"] == "bab_ii.pasal_3.ayat_2"
    assert results[0]["text"].startswith("Ayat (2)")
    assert results[0]["score"] == pytest.approx(0.93)
    assert results[0]["final_score"] == pytest.approx(0.93)
    assert results[0]["retrieval_score"] == pytest.approx(0.93)
    assert results[1]["section_id"] == "bab_ii.pasal_3.list_3"

    cached_results = await vector_tools.search_similar(
        query="berapa batas izin tidak masuk sekolah?",
        top_k=2,
        score_threshold=0.0,
    )
    assert cached_results == results
    assert vector_tools.client.query_calls == 1


@pytest.mark.asyncio
async def test_search_similar_supports_reranker_strategy(monkeypatch):
    """Reranker mode should reorder hydrated parent results and cache them."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()

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
            return SimpleNamespace(
                points=[
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
                ]
            )

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
        "bab_ii.pasal_3.ayat_2",
        "bab_i.pasal_1.ayat_1",
    ]
    assert results[0]["reranker_score"] == pytest.approx(0.97)
    assert results[0]["final_score"] == pytest.approx(0.97)
    assert results[0]["retrieval_score"] == pytest.approx(0.82)
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
async def test_reranker_uses_rrf_candidates_when_bm25_is_available(monkeypatch):
    """Reranker mode should rerank hybrid RRF candidates, not dense-only candidates."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()

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
                return SimpleNamespace(
                    points=[
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
                    ]
                )

            self.dense_calls += 1
            return SimpleNamespace(
                points=[
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
                ]
            )

    class FakeReranker:
        def rerank(self, *, query: str, documents: list[str]) -> list[float]:
            assert query == "cuti akademik biaya DPP"
            assert len(documents) == 2
            # Prefer the document that mentions DPP (the cuti chunk)
            return [0.91 if "DPP" in document else 0.12 for document in documents]

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

    assert [result["section_id"] for result in results] == ["cuti.p1", "general.p1"]
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
async def test_search_similar_supports_rrf_strategy(monkeypatch):
    """RRF mode should fuse dense and sparse rankings before parent hydration."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()

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
                return SimpleNamespace(
                    points=[
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
                    ]
                )

            self.dense_calls += 1
            self.dense_kwargs.append(kwargs)
            return SimpleNamespace(
                points=[
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
                ]
            )

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
        "bab_ii.pasal_3.ayat_2",
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
async def test_search_similar_filters_negative_reranker_results(monkeypatch):
    """Negative reranker scores should not be returned as context or citations."""
    vector_tools = VectorStoreTools()
    vector_tools.cache = FakeCache()

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
            return SimpleNamespace(
                points=[
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
                ]
            )

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
    assert results[0]["section_id"] == "pasal_6_2.table_1"
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
