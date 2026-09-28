"""Unit tests for citation filtering and label sanitization."""


from langchain_core.messages import AIMessage, HumanMessage

from src.services.response_generation import CitationBuilder, ResponseGenerationService
from src.utils.state import AgentState, Citation


class TestCitationSanitization:
    """Test sanitization of section labels and page formatting."""

    def test_meaningful_sections_retained(self):
        """Structural sections like 'BAB II > Pasal 3' and '3.1 Profil Lulusan' should be retained."""
        builder = CitationBuilder()
        assert builder._is_internal_id("BAB II > Pasal 3") is False
        assert builder._is_internal_id("3.1 Profil Lulusan") is False
        assert (
            builder._is_internal_id(
                "IV.1. Registrasi dan Perwalian > IV.1.1. Tahap Pendaftaran dan Pembayaran DPP"
            )
            is False
        )
        assert builder._is_internal_id("Pasal 5 Ketentuan Kelulusan") is False
        assert builder._is_internal_id("Lampiran A Formulir") is False

    def test_internal_uuids_and_chunk_patterns_omitted(self):
        """UUIDs, :chunk: patterns, and chunk identifiers must not be treated as meaningful."""
        # UUIDs
        builder = CitationBuilder()
        assert builder._is_internal_id("123e4567-e89b-12d3-a456-426614174000") is True
        assert builder._is_internal_id("4e73b71e-09a2-4a0b-a0db-91b539a2f1ab") is True
        assert builder._is_internal_id("doc-4e73b71e-09a2-4a0b-a0db-91b539a2f1ab") is True

        # :chunk: patterns
        assert builder._is_internal_id("doc-123:chunk:0") is True
        assert builder._is_internal_id("guidelines.pdf:chunk:14") is True
        assert builder._is_internal_id(":chunk:5") is True

        # Chunk IDs
        assert builder._is_internal_id("parent-chunk") is True
        assert builder._is_internal_id("child-chunk") is True
        assert builder._is_internal_id("chunk_0") is True
        assert builder._is_internal_id("chunk-1") is True

        # Empty or generic values are not internal IDs, but are skipped by the
        # section-label selector because they are falsy or unavailable.
        assert builder._is_internal_id("") is False
        assert builder._is_internal_id("unknown") is False

    def test_section_label_selection_omits_chunk_and_uuid(self):
        """_section_label should ignore internal IDs and pick meaningful structural names."""
        builder = CitationBuilder()

        # Chunk with breadcrumb and internal chunk_id
        chunk_with_breadcrumb = {
            "breadcrumb": "BAB II > Pasal 3",
            "chunk_id": "parent-chunk",
            "section_id": "doc-123:chunk:0",
        }
        assert builder._section_label(chunk_with_breadcrumb) == "BAB II > Pasal 3"

        # Chunk with only UUID/chunk IDs
        chunk_internal_only = {
            "chunk_id": "4e73b71e-09a2-4a0b-a0db-91b539a2f1ab",
            "section_id": "doc-4e73b71e:chunk:2",
        }
        assert builder._section_label(chunk_internal_only) is None

    def test_format_citation_label_omits_internal_sections(self):
        """_format_citation_label should never expose UUIDs or chunk patterns."""
        service = ResponseGenerationService()

        cit_uuid = Citation(
            id=1,
            filename="pedoman.pdf",
            section="4e73b71e-09a2-4a0b-a0db-91b539a2f1ab",
            page=5,
        )
        assert service._format_citation_label(cit_uuid) == "pedoman.pdf (p. 5)"

        cit_chunk = Citation(
            id=2,
            filename="pedoman.pdf",
            section="doc-1:chunk:12",
            page=8,
        )
        assert service._format_citation_label(cit_chunk) == "pedoman.pdf (p. 8)"

        cit_structural = Citation(
            id=3,
            filename="kurikulum.pdf",
            section="3.1 Profil Lulusan",
            page=14,
        )
        assert (
            service._format_citation_label(cit_structural)
            == "kurikulum.pdf :: 3.1 Profil Lulusan (p. 14)"
        )


class TestOrphanCitationFiltering:
    """Test filtering orphan/phantom citations in footer and synchronization in generate()."""

    def test_append_citation_footer_only_includes_cited_sources(self):
        """Only citations present as [N] markers in content should appear in the footer."""
        service = ResponseGenerationService()
        citations = [
            Citation(id=1, filename="doc1.pdf", section="BAB I", page=2),
            Citation(id=2, filename="doc2.pdf", section="BAB II", page=5),
            Citation(id=3, filename="doc3.pdf", section="BAB III", page=10),
        ]

        # Content only cites [1] and [3]
        content = "Berdasarkan aturan kurikulum [1], dan ketentuan kelulusan [3]."
        footer, active_citations = service._append_citation_footer(content, citations)

        assert "Sources:" in footer
        assert "[1] doc1.pdf :: BAB I (p. 2)" in footer
        assert "[3] doc3.pdf :: BAB III (p. 10)" in footer
        assert "[2] doc2.pdf" not in footer
        assert [citation.id for citation in active_citations] == [1, 3]

    def test_append_citation_footer_omits_orphan_citations_when_no_markers(self):
        """When content has no citation markers, no footer or orphan citations are added."""
        service = ResponseGenerationService()
        citations = [
            Citation(id=1, filename="doc1.pdf", page=2),
            Citation(id=2, filename="doc2.pdf", page=5),
        ]

        content = "Halo, ada yang bisa saya bantu terkait perkuliahan?"
        result, active_citations = service._append_citation_footer(content, citations)

        assert result == content
        assert active_citations == []
        assert "Sources:" not in result

    def test_append_citation_footer_handles_empty_citations(self):
        """When citations list is empty, content is returned unchanged."""
        service = ResponseGenerationService()
        content = "Informasi ini diambil dari dokumen [1]."
        result, active_citations = service._append_citation_footer(content, [])

        assert result == content
        assert active_citations == []
        assert "Sources:" not in result

    def test_append_citation_footer_gracefully_handles_unmatched_numbers(self):
        """When citation numbers in text do not match retrieved citation IDs, fall back gracefully."""
        service = ResponseGenerationService()
        citations = [
            Citation(id=1, filename="doc1.pdf", page=2),
        ]

        content = "Referensi menunjukkan aturan ini [99]."
        result, active_citations = service._append_citation_footer(content, citations)

        assert result == content
        assert active_citations == []
        assert "Sources:" not in result

    def test_generate_syncs_citations_list_with_footer(self):
        """generate() should return active citations matching the footer so frontend expanders match."""
        citations = [
            Citation(id=1, filename="pedoman.pdf", section="BAB I", page=3),
            Citation(id=2, filename="orphan.pdf", section="BAB II", page=9),
        ]

        class FakeCitationBuilder:
            def build(self, chunks):
                return citations

        class FakeLLM:
            def invoke(self, messages):
                # The LLM only cites [1]
                return AIMessage(
                    content="Sesuai pedoman akademik [1], mahasiswa wajib registrasi."
                )

        service = ResponseGenerationService(
            citation_builder=FakeCitationBuilder(),
            llm_provider=lambda: FakeLLM(),
        )

        state = AgentState(
            messages=[HumanMessage(content="Bagaimana aturan registrasi?")],
            retrieved_chunks=[{"dummy": "chunk"}],
            current_intent="query_document",
        )

        result = service.generate(state)

        # Footer should only have citation [1]
        assert "[1] pedoman.pdf :: BAB I (p. 3)" in result["draft_response"]
        assert "orphan.pdf" not in result["draft_response"]

        # Returned citations list must be synced with footer (only [1], not [2])
        assert len(result["citations"]) == 1
        assert result["citations"][0].id == 1
        assert result["citations"][0].filename == "pedoman.pdf"

    def test_generate_returns_empty_citations_when_llm_does_not_cite(self):
        """When LLM does not cite any documents, returned citations should be empty."""
        citations = [
            Citation(id=1, filename="pedoman.pdf", page=3),
        ]

        class FakeCitationBuilder:
            def build(self, chunks):
                return citations

        class FakeLLM:
            def invoke(self, messages):
                return AIMessage(content="Saya tidak menemukan informasi tersebut.")

        service = ResponseGenerationService(
            citation_builder=FakeCitationBuilder(),
            llm_provider=lambda: FakeLLM(),
        )

        state = AgentState(
            messages=[HumanMessage(content="Apakah ada beasiswa?")],
            retrieved_chunks=[{"dummy": "chunk"}],
            current_intent="query_document",
        )

        result = service.generate(state)

        assert "Sources:" not in result["draft_response"]
        assert result["citations"] == []
