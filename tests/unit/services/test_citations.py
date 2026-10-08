"""Unit tests for CitationBuilder and ResponseGenerationService citation processing."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.services.response_generation import (
    CitationBuilder,
    ResponseContextBuilder,
    ResponseGenerationService,
)
from src.utils.state import AgentState, Citation


def test_unit_citation_builder_sequential_ids_and_metadata() -> None:
    """CitationBuilder creates 1-based sequential citations with rich metadata."""
    builder = CitationBuilder(limit=5)
    chunks = [
        {
            "document_id": "doc-01",
            "chunk_id": "chunk-01",
            "filename": "Pedoman_Akademik.pdf",
            "doc_title": "Pedoman Akademik 2024",
            "breadcrumb": "BAB III > Persyaratan Kelulusan",
            "text": "Mahasiswa wajib menyelesaikan 148 SKS.",
            "final_score": 0.95,
            "source_locations": [{"page": 4}],  # 0-based index -> page 5
        },
        {
            "document_id": "doc-02",
            "chunk_id": "chunk-02",
            "filename": "Kurikulum_IF_2021.pdf",
            "title": "Kurikulum IF",
            "section_id": "Semester 5",
            "text": "Mata kuliah pilihan semester 5.",
            "final_score": 0.82,
            "source_locations": [{"page": 11}],  # 0-based index -> page 12
        },
    ]

    citations = builder.build(chunks)
    assert len(citations) == 2

    c1 = citations[0]
    assert c1.id == 1
    assert c1.filename == "Pedoman_Akademik.pdf"
    assert c1.title == "Pedoman Akademik 2024"
    assert c1.section == "BAB III > Persyaratan Kelulusan"
    assert c1.page == 5
    assert c1.score == 0.95
    assert c1.snippet == "Mahasiswa wajib menyelesaikan 148 SKS."
    assert c1.is_cited is False

    c2 = citations[1]
    assert c2.id == 2
    assert c2.filename == "Kurikulum_IF_2021.pdf"
    assert c2.title == "Kurikulum IF"
    assert c2.section == "Semester 5"
    assert c2.page == 12
    assert c2.score == 0.82
    assert c2.is_cited is False


def test_unit_citation_builder_filters_weak_negative_scores() -> None:
    """Chunks with negative relevance scores are filtered out."""
    builder = CitationBuilder(limit=5)
    chunks = [
        {"filename": "Good.pdf", "text": "Valid content", "final_score": 0.50},
        {"filename": "Bad.pdf", "text": "Irrelevant content", "final_score": -0.15},
    ]

    citations = builder.build(chunks)
    assert len(citations) == 1
    assert citations[0].filename == "Good.pdf"
    assert citations[0].id == 1


def test_unit_process_citations_marks_inline_and_strips_trailing_sources() -> None:
    """_process_citations sets is_cited on inline references and strips text footers."""
    service = ResponseGenerationService()
    citations = [
        Citation(id=1, filename="Doc1.pdf"),
        Citation(id=2, filename="Doc2.pdf"),
        Citation(id=3, filename="Doc3.pdf"),
    ]

    raw_response = (
        "Berdasarkan dokumen yang tersedia, batas waktu pengisian KRS adalah akhir minggu ini [1]. "
        "Selain itu, syarat kelulusan adalah 148 SKS [3].\n\n"
        "Sources:\n"
        "[1] Doc1.pdf\n"
        "[3] Doc3.pdf"
    )

    cleaned_text, processed_citations = service._process_citations(raw_response, citations)

    # Must strip the text Sources: footer
    assert "Sources:" not in cleaned_text
    assert "Doc1.pdf" not in cleaned_text
    # Must preserve the clean text and inline numbers
    assert "akhir minggu ini [1]." in cleaned_text
    assert "148 SKS [3]." in cleaned_text

    # All citations are retained for the frontend
    assert len(processed_citations) == 3
    assert processed_citations[0].id == 1
    assert processed_citations[0].is_cited is True
    assert processed_citations[1].id == 2
    assert processed_citations[1].is_cited is False
    assert processed_citations[2].id == 3
    assert processed_citations[2].is_cited is True


def test_unit_response_generation_with_llm_returns_numeric_citations() -> None:
    """generate() returns clean response with inline numeric markers and all citations."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Berdasarkan pedoman akademik, mahasiswa wajib menempuh minimal 148 SKS [1]."
    )

    service = ResponseGenerationService(llm_provider=lambda: mock_llm)
    state = AgentState(
        session_id="test-session",
        messages=[HumanMessage(content="Berapa SKS untuk lulus?")],
        retrieved_chunks=[
            {"document_id": "d1", "filename": "Pedoman.pdf", "text": "Minimal 148 SKS.", "score": 0.90},
            {"document_id": "d2", "filename": "Kurikulum.pdf", "text": "Struktur kurikulum.", "score": 0.70},
        ],
    )

    result = service.generate(state)

    assert result["draft_response"] == (
        "Berdasarkan pedoman akademik, mahasiswa wajib menempuh minimal 148 SKS [1]."
    )
    assert "Sources:" not in result["draft_response"]
    assert "Pedoman.pdf" not in result["draft_response"]

    # Frontend receives all citations
    citations = result["citations"]
    assert len(citations) == 2
    assert citations[0].id == 1
    assert citations[0].is_cited is True
    assert citations[1].id == 2
    assert citations[1].is_cited is False


def test_unit_fallback_response_shows_only_numeric_marker() -> None:
    """Deterministic fallback output uses numeric marker [1] when chunks exist."""
    service = ResponseGenerationService(llm_provider=lambda: None)
    state = AgentState(
        session_id="test-session",
        messages=[HumanMessage(content="Halo")],
        retrieved_chunks=[
            {"filename": "Doc.pdf", "text": "Aturan akademik semester genap.", "score": 0.80}
        ],
    )

    result = service.generate(state)
    assert result["draft_response"].startswith("Based on available documents, I found:")
    assert "[1]" in result["draft_response"]
    assert "Doc.pdf" not in result["draft_response"]
    assert len(result["citations"]) == 1
    assert result["citations"][0].is_cited is True


def test_unit_context_builder_matches_citation_ids() -> None:
    """ResponseContextBuilder excerpt numbers match CitationBuilder ids."""
    context_builder = ResponseContextBuilder(top_k=2)
    citation_builder = CitationBuilder(limit=2)

    chunks = [
        {"filename": "A.pdf", "text": "Content A", "final_score": 0.9},
        {"filename": "B.pdf", "text": "Content B", "final_score": 0.8},
    ]

    context = context_builder._build_retrieval_context(chunks)
    citations = citation_builder.build(chunks)

    assert "[1] A.pdf" in context[1]
    assert citations[0].id == 1
    assert citations[0].filename == "A.pdf"

    assert "[2] B.pdf" in context[2]
    assert citations[1].id == 2
    assert citations[1].filename == "B.pdf"
