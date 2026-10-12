"""Unit tests for intent-based workflow routing."""

from src.graphs.workflow import route_by_intent
from src.utils.state import AgentState, DocumentType, DocumentUpload


def _pending_document() -> DocumentUpload:
    return DocumentUpload(
        document_id="doc-1",
        filename="policy.pdf",
        file_path="data/raw/policy.pdf",
        document_type=DocumentType.POLICY,
        mime_type="application/pdf",
        file_size=1,
    )


def test_attached_files_route_to_document_processing() -> None:
    """An attached file is processed even when the message classifies as a question."""
    state = AgentState(
        current_intent="query_document",
        pending_documents=[_pending_document()],
    )

    assert route_by_intent(state) == "process_document"


def test_document_question_without_files_retrieves() -> None:
    """A document question with no attachment keeps its retrieval route."""
    assert route_by_intent(AgentState(current_intent="query_document")) == "retrieve"
