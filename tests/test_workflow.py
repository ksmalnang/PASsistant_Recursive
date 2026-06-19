"""
Unit tests for the LangGraph workflow.

Tests core graph functionality including routing,
document processing, and response generation.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.graphs.workflow import check_student_resolution
from src.services.document_processing import DocumentProcessingService
from src.services.response_generation import CitationBuilder, ResponseContextBuilder
from src.utils.nodes import ResponseNode, RetrievalNode, RouterNode
from src.utils.state import AgentState, DocumentType, DocumentUpload, OCRResult, ProcessingStatus


class TestRouterNode:
    """Test intent classification routing."""

    def test_router_detects_upload_intent(self):
        """Router should classify upload-related messages."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="I want to upload a transcript PDF")]
        )

        result = router.run(state)

        assert result["current_intent"] == "upload_document"
        assert result["requires_upload"] is True

    def test_router_detects_query_intent(self):
        """Router should classify student query messages."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="What is my GPA?")]
        )

        result = router.run(state)

        assert result["current_intent"] == "query_student"
        assert result["requires_retrieval"] is True

    def test_query_student_record_lookup_uses_transcript_scope(self):
        """Pure record lookups should stay scoped to transcript documents."""
        retrieval = RetrievalNode()

        assert (
            retrieval._resolve_document_type("query_student", "What is my GPA?")
            == DocumentType.TRANSCRIPT
        )

    def test_query_student_policy_question_uses_all_documents(self):
        """Student wording about SKS/DPP rules should search policy docs too."""
        retrieval = RetrievalNode()
        query = (
            "Kak, IPS ku kemaren dapet 3.12, brarti bisa ambil brp sks nih? "
            "trus tagihan cicilan 1 DPP tuh bulan apa ya bayarnya?"
        )

        assert retrieval._resolve_document_type("query_student", query) is None

    def test_extract_keywords_drops_conversational_filler_for_policy_queries(self):
        """Keyword extraction should retain policy terms and drop Indonesian filler words."""
        retrieval = RetrievalNode()

        extracted = retrieval._extract_keywords(
            "Kak, kalau aku telat bayar DPP cicilan pertama, masih bisa ikut perwalian nggak?"
        )

        assert extracted == "telat bayar dpp cicilan pertama perwalian"

    def test_rrf_policy_retrieval_with_explicit_rule_is_not_marked_low_confidence(self):
        """Low absolute RRF scores should not trigger warnings when the rule is explicit."""
        retrieval = RetrievalNode()
        retrieval.vector_tools.retrieval_strategy = "rrf"
        results = [
            {
                "score": 0.028,
                "rrf_score": 0.028,
                "breadcrumb": (
                    "IV.1. Registrasi dan Perwalian > "
                    "IV.1.1. Tahap Pendaftaran dan Pembayaran DPP"
                ),
                "text": "Tahap Pendaftaran dan Pembayaran DPP",
                "matched_children": [
                    {
                        "chunk_type": "paragraph",
                        "score": 0.016,
                        "text": (
                            "Perwalian dapat dilakukan setelah mahasiswa memenuhi "
                            "persyaratan administrasi pembayaran uang kuliah "
                            "yang disyaratkan dari DPP/SPP Tahun Akademik yang bersangkutan."
                        ),
                    },
                    {
                        "chunk_type": "table",
                        "score": 0.015,
                        "text": (
                            "DPP dana pelaksanaan pendidikan : dibayarkan 4 kali cicilan "
                            "Cicilan I 25% : Juli"
                        ),
                    },
                ],
            },
            {
                "score": 0.018,
                "rrf_score": 0.018,
                "breadcrumb": "Lampiran",
                "text": "Lampiran susunan pengurus",
                "matched_children": [
                    {"chunk_type": "paragraph", "score": 0.011, "text": "Susunan pengurus."}
                ],
            },
        ]

        confidence, warning = retrieval._score_retrieval_confidence(
            "Kak, kalau aku telat bayar DPP cicilan pertama, masih bisa ikut perwalian nggak?",
            results,
        )

        assert confidence >= 0.45
        assert warning is None
        assert results[0]["query_overlap_ratio"] >= 0.60

    def test_reranker_policy_retrieval_with_explicit_rule_is_not_marked_low_confidence(self):
        """Moderate reranker scores should still count as high-confidence policy support."""
        retrieval = RetrievalNode()
        retrieval.vector_tools.retrieval_strategy = "reranker"
        results = [
            {
                "score": 0.32,
                "reranker_score": 0.32,
                "breadcrumb": (
                    "IV.1. Registrasi dan Perwalian > "
                    "IV.1.1. Tahap Pendaftaran dan Pembayaran DPP"
                ),
                "text": "Tahap Pendaftaran dan Pembayaran DPP",
                "matched_children": [
                    {
                        "chunk_type": "paragraph",
                        "score": 0.016,
                        "text": (
                            "Perwalian dapat dilakukan setelah mahasiswa memenuhi "
                            "persyaratan administrasi pembayaran uang kuliah "
                            "yang disyaratkan dari DPP/SPP Tahun Akademik yang bersangkutan."
                        ),
                    },
                    {
                        "chunk_type": "table",
                        "score": 0.015,
                        "text": (
                            "DPP dana pelaksanaan pendidikan : dibayarkan 4 kali cicilan "
                            "Cicilan I 25% : Juli"
                        ),
                    },
                ],
            },
            {
                "score": 0.21,
                "reranker_score": 0.21,
                "breadcrumb": "VI.4. Pelayanan Administrasi Akademik",
                "text": "Prosedur Bebas Mata Kuliah hanya tinggal Tugas Akhir",
                "matched_children": [
                    {
                        "chunk_type": "paragraph",
                        "score": 0.015,
                        "text": "Mahasiswa membayar tagihan cicilan I sebelum perwalian tugas akhir.",
                    }
                ],
            },
        ]

        confidence, warning = retrieval._score_retrieval_confidence(
            "Kak, kalau aku telat bayar DPP cicilan pertama, masih bisa ikut perwalian nggak?",
            results,
        )

        assert confidence >= 0.45
        assert warning is None

    def test_router_detects_academic_service_query(self):
        """Router should classify academic service questions for retrieval."""
        router = RouterNode()
        state = AgentState(
            messages=[
                HumanMessage(
                    content="How do I request an enrollment verification letter?"
                )
            ]
        )

        result = router.run(state)

        assert result["current_intent"] == "query_document"
        assert result["requires_retrieval"] is True

    def test_router_detects_curriculum_document_query_in_indonesian(self):
        """Router should classify curriculum-document questions for retrieval."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="apa saja Profil Lulusan dan deskripsinya")]
        )

        result = router.run(state)

        assert result["current_intent"] == "query_document"
        assert result["requires_retrieval"] is True

    def test_router_detects_indonesian_academic_policy_question(self):
        """Policy questions with Indonesian academic terms should skip LLM fallback."""
        router = RouterNode()
        state = AgentState(
            messages=[
                HumanMessage(
                    content=(
                        "Apa yang akan terjadi jika saya absen atau tidak aktif mengikuti "
                        "perkuliahan selama 4 semester berturut-turut tanpa mengajukan cuti resmi?"
                    )
                )
            ]
        )

        result = router.run(state)

        assert result["current_intent"] == "query_document"
        assert result["requires_retrieval"] is True

    def test_router_detects_general_chat(self):
        """Router should classify general conversation."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="Hello, how are you?")]
        )

        result = router.run(state)

        assert result["current_intent"] == "general_chat"


class TestResponseNode:
    """Test response generation."""

    def test_citation_builder_uses_child_source_locations(self):
        """Citation metadata should prefer the matched child source location."""
        builder = CitationBuilder()

        citations = builder.build(
            [
                {
                    "document_id": "doc-123",
                    "filename": "kurikulum.pdf",
                    "doc_title": "Kurikulum Teknik Informatika",
                    "breadcrumb": "3.1 Profil Lulusan",
                    "score": 0.97,
                    "chunk_id": "parent-chunk",
                    "parent_id": "doc-123:bab_3.pasal_3_1",
                    "text": "Parent text",
                    "source_locations": [
                        {"page": 9, "bbox_2d": [1.0, 2.0, 3.0, 4.0]}
                    ],
                    "matched_children": [
                        {
                            "chunk_id": "child-chunk",
                            "text": "Profil lulusan dan deskripsinya",
                            "source_locations": [
                                {"page": 13, "bbox_2d": [10.0, 20.0, 100.0, 50.0]}
                            ],
                        }
                    ],
                }
            ]
        )

        assert len(citations) == 1
        assert citations[0].id == 1
        assert citations[0].filename == "kurikulum.pdf"
        assert citations[0].title == "Kurikulum Teknik Informatika"
        assert citations[0].section == "3.1 Profil Lulusan"
        assert citations[0].page == 14
        assert citations[0].chunk_id == "child-chunk"
        assert citations[0].source_locations == [
            {"page": 13, "bbox_2d": [10.0, 20.0, 100.0, 50.0]}
        ]

    def test_citation_builder_deduplicates_context_records(self):
        """Multiple child hits under one parent should produce one citation."""
        builder = CitationBuilder()

        citations = builder.build(
            [
                {
                    "filename": "policy.pdf",
                    "parent_id": "doc-1:section-1",
                    "section_id": "section-1",
                    "score": 0.9,
                    "text": "First match",
                },
                {
                    "filename": "policy.pdf",
                    "parent_id": "doc-1:section-1",
                    "section_id": "section-1",
                    "score": 0.8,
                    "text": "Duplicate parent",
                },
            ]
        )

        assert len(citations) == 1

    def test_response_with_context(self):
        """Response should incorporate retrieved context."""
        response_node = ResponseNode()
        state = AgentState(
            messages=[
                HumanMessage(content="What courses did John take?")
            ],
            retrieved_chunks=[
                {
                    "text": "John enrolled in CS101, MATH201, and PHYS101 in Fall 2024.",
                    "score": 0.95,
                    "filename": "transcript.pdf",
                }
            ],
            current_intent="query_document",
        )

        result = response_node.run(state)

        assert "draft_response" in result
        assert len(result["messages"]) > 0
        assert result["citations"][0].filename == "transcript.pdf"
        assert "Sources:" in result["draft_response"]

    def test_response_without_context(self):
        """Response should handle missing context gracefully."""
        response_node = ResponseNode()
        state = AgentState(
            messages=[HumanMessage(content="Hello!")],
            current_intent="general_chat",
        )

        result = response_node.run(state)

        assert "draft_response" in result
        assert len(result["messages"]) > 0

    def test_response_context_preserves_table_rows_beyond_short_preview(self):
        """Table-heavy retrieval context should keep later rows such as PL2/PL3."""
        builder = ResponseContextBuilder()
        large_table_text = (
            "3.1 Profil Lulusan\n\n"
            + ("Pendahuluan. " * 80)
            + "| PL1 | Professional bidang Informatika | Deskripsi PL1 |\n"
            + "| PL2 | Technopreneur | Deskripsi PL2 |\n"
            + "| PL3 | Akademisi | Deskripsi PL3 |\n"
            + "| PL4 | Global citizen | Deskripsi PL4 |\n"
        )

        context_lines = builder._build_retrieval_context(
            [
                {
                    "filename": "kurikulum.pdf",
                    "breadcrumb": "3.1 Profil Lulusan",
                    "score": 0.97,
                    "text": large_table_text,
                    "matched_children": [{"text": "Profil lulusan dan deskripsinya"}],
                }
            ]
        )

        assert "PL3" in "\n".join(context_lines)

    def test_response_context_prioritizes_policy_rule_before_schedule_and_sync_detail(self):
        """Policy context should lead with the explicit rule and still retain follow-up details."""
        builder = ResponseContextBuilder()

        context_lines = builder._build_retrieval_context(
            [
                {
                    "filename": "Wadek1-Buku Panduan Akademik FT Unpas 2023-2024 Publish.pdf",
                    "breadcrumb": "IV.1.1. Tahap Pendaftaran dan Pembayaran DPP",
                    "score": 0.32,
                    "text": "Tahap Pendaftaran dan Pembayaran DPP",
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
            ]
        )

        matched_section = "\n".join(context_lines).split("Matched child evidence:\n", 1)[1]
        assert "Perwalian dapat dilakukan setelah" in matched_section
        assert "Cicilan I 25% : Juli" in matched_section
        assert "SITU 2.0" in matched_section
        assert matched_section.index("Perwalian dapat dilakukan setelah") < matched_section.index(
            "Cicilan I 25% : Juli"
        )


class TestAgentState:
    """Test state schema operations."""

    def test_state_initialization(self):
        """State should initialize with defaults."""
        state = AgentState()

        assert state.turn_count == 0
        assert state.current_intent is None
        assert len(state.pending_documents) == 0
        assert len(state.processed_documents) == 0

    def test_state_with_document(self):
        """State should track pending documents."""
        doc = DocumentUpload(
            document_id="test-doc-001",
            filename="transcript.pdf",
            file_path="/tmp/transcript.pdf",
            mime_type="application/pdf",
            file_size=1024,
        )

        state = AgentState(
            pending_documents=[doc],
            current_intent="upload_document",
        )

        assert len(state.pending_documents) == 1
        assert state.pending_documents[0].document_id == "test-doc-001"


class TestDocumentProcessingService:
    """Test document processing metadata behavior."""

    @pytest.mark.asyncio
    async def test_partial_ocr_completes_with_warnings(self):
        """Partial OCR should complete while preserving page status metadata."""
        document = DocumentUpload(
            document_id="doc-partial",
            filename="partial.pdf",
            file_path="/tmp/partial.pdf",
            mime_type="application/pdf",
            file_size=1024,
        )

        class FakeExtractor:
            async def extract_text(self, file_path, document_type):
                assert file_path == "/tmp/partial.pdf"
                return OCRResult(
                    text="page zero text",
                    text_quality_score=0.95,
                    num_pages=2,
                    layout_details=[
                        [{"content": "page zero text", "bbox_2d": [1, 2, 3, 4]}],
                        [],
                    ],
                    parsed_pages=1,
                    failed_pages=[1],
                    page_results=[
                        {"page_index": 0, "status": "success", "text_length": 14},
                        {"page_index": 1, "status": "failed", "error": "timeout"},
                    ],
                    ocr_warnings=["OCR parsed 1 of 2 pages successfully."],
                )

        class FakeIndexer:
            async def store_document_chunks(self, indexed_document):
                assert indexed_document.parsed_pages == 1
                assert indexed_document.failed_pages == [1]
                return ["chunk-1"]

        service = DocumentProcessingService(
            text_extractor=FakeExtractor(),
            chunk_indexer=FakeIndexer(),
        )

        await service.process_document(document)

        assert document.processing_status == ProcessingStatus.COMPLETED
        assert document.processing_error == "OCR parsed 1 of 2 pages successfully."
        assert document.parsed_pages == 1
        assert document.failed_pages == [1]
        assert document.ocr_page_status[1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_processing_fails_when_no_vector_chunks_are_stored(self):
        """Completed OCR without vector chunks should not look like successful ingest."""
        document = DocumentUpload(
            document_id="doc-empty-index",
            filename="empty-index.pdf",
            file_path="/tmp/empty-index.pdf",
            mime_type="application/pdf",
            file_size=1024,
        )

        class FakeExtractor:
            async def extract_text(self, file_path, document_type):
                return OCRResult(
                    text="valid OCR text",
                    text_quality_score=0.95,
                    num_pages=1,
                    layout_details=[[{"content": "valid OCR text"}]],
                    parsed_pages=1,
                )

        class EmptyIndexer:
            async def store_document_chunks(self, indexed_document):
                assert indexed_document.extracted_text == "valid OCR text"
                return []

        service = DocumentProcessingService(
            text_extractor=FakeExtractor(),
            chunk_indexer=EmptyIndexer(),
        )

        with pytest.raises(RuntimeError, match="no vector chunks were stored"):
            await service.process_document(document)

        assert document.processing_status == ProcessingStatus.FAILED
        assert document.processing_error == (
            "Document ingestion completed OCR but no vector chunks were stored"
        )


class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""

    def test_student_resolution_routes_to_retrieval_when_needed(self):
        """Student handler should forward unresolved record questions to retrieval."""
        state = AgentState(
            current_intent="query_student",
            requires_retrieval=True,
            retrieval_query="What is my GPA?",
        )

        assert check_student_resolution(state) == "retrieve"

    @pytest.mark.asyncio
    async def test_full_chat_flow(self, compiled_test_app):
        """Test complete message processing through the graph."""
        initial_state = AgentState(
            messages=[HumanMessage(content="Hi there!")],
            session_id="test-integration",
        )

        config = {"configurable": {"thread_id": "test-integration"}}

        result = await compiled_test_app.ainvoke(
            initial_state.model_dump(),
            config=config,
        )

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1  # User + assistant messages
