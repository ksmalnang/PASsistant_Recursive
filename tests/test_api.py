"""API endpoint tests."""

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api import app
from src.api.models import ChatResponse, ChatStreamEvent, DependencyHealthResponse
from src.api.services import ChatRouteService
from src.services.contracts import AgentStreamUpdate
from src.utils.state import Citation, DocumentType, DocumentUpload, ProcessingStatus


def test_upload_documents_returns_created(monkeypatch):
    """Upload route should ingest documents and return HTTP 201."""

    async def fake_handle_knowledge_base_ingestion(files):
        assert len(files) == 1
        return [
            {
                "document_id": "doc-123",
                "filename": "knowledge-base.pdf",
                "document_type": "other",
                "status": "completed",
                "success": True,
                "vectors_stored": 2,
                "context_records_stored": 1,
                "document_title": "Knowledge Base",
                "parsed_pages": None,
                "failed_pages": [],
                "ocr_warnings": [],
                "quality_warning": None,
                "error": None,
            }
        ]

    monkeypatch.setattr(
        "src.api.routes.documents.handle_knowledge_base_ingestion",
        fake_handle_knowledge_base_ingestion,
    )

    client = TestClient(app)
    response = client.post(
        "/upload",
        files={
            "files": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")
        },
    )

    assert response.status_code == 201
    assert response.json() == [
        {
            "success": True,
            "document_id": "doc-123",
            "filename": "knowledge-base.pdf",
            "document_type": "other",
            "status": "completed",
            "document_title": "Knowledge Base",
            "parsed_pages": None,
            "failed_pages": [],
            "ocr_warnings": [],
            "quality_warning": None,
            "error": None,
        }
    ]


def test_upload_documents_rejects_unsupported_mime_type():
    """Document upload route should reject unsupported MIME types."""

    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"files": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported file type for 'notes.txt'. Only PDF, PNG, JPEG, GIF, and WEBP uploads are supported."
    }


def test_upload_documents_rejects_oversized_pdf(monkeypatch):
    """Document upload route should reject files above the configured PDF size limit."""

    monkeypatch.setattr("src.api.routes.documents.GLMOCRTool.MAX_PDF_BYTES", 4)

    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"files": ("large.pdf", b"12345", "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Uploaded file 'large.pdf' is too large. Maximum size for application/pdf is 0.0 MB."
    }


def test_ingest_knowledge_base_documents_success(monkeypatch):
    """Knowledge-base ingestion should return processing metadata."""

    class FakeProcessor:
        async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
            assert file_bytes == b"sample knowledge base content"
            assert filename == "knowledge-base.pdf"
            return DocumentUpload(
                document_id="doc-123",
                filename=filename,
                file_path="data/raw/doc-123.pdf",
                document_type=DocumentType.OTHER,
                mime_type="application/pdf",
                file_size=len(file_bytes),
                processing_status=ProcessingStatus.COMPLETED,
                document_title="Knowledge Base",
            )

    monkeypatch.setattr("src.api.helpers.DocumentProcessingNode", FakeProcessor)

    client = TestClient(app)
    response = client.post(
        "/knowledge-base/ingest",
        files={
            "files": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")
        },
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "success": True,
            "document_id": "doc-123",
            "filename": "knowledge-base.pdf",
            "document_type": "other",
            "status": "completed",
            "document_title": "Knowledge Base",
            "parsed_pages": None,
            "failed_pages": [],
            "ocr_warnings": [],
            "quality_warning": None,
            "error": None,
        }
    ]


def test_ingest_knowledge_base_documents_failure(monkeypatch):
    """Knowledge-base ingestion should surface per-file processing failures."""

    class FakeProcessor:
        async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
            raise RuntimeError("vector store unavailable")

    monkeypatch.setattr("src.api.helpers.DocumentProcessingNode", FakeProcessor)

    client = TestClient(app)
    response = client.post(
        "/knowledge-base/ingest",
        files={
            "files": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")
        },
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "success": False,
            "document_id": "error",
            "filename": "knowledge-base.pdf",
            "document_type": "other",
            "status": "failed",
            "document_title": None,
            "parsed_pages": None,
            "failed_pages": [],
            "ocr_warnings": [],
            "quality_warning": None,
            "error": "vector store unavailable",
        }
    ]


def test_ingest_knowledge_base_documents_missing_files_returns_400():
    """Knowledge-base ingestion should reject requests without uploaded files."""

    client = TestClient(app)
    response = client.post("/knowledge-base/ingest")

    assert response.status_code == 422


def test_health_check_includes_dependency_statuses(monkeypatch):
    """Health route should expose Redis and Qdrant dependency states."""

    monkeypatch.setattr(
        "src.api.routes.health._check_redis_health",
        lambda: DependencyHealthResponse(status="healthy", detail="Redis responded to ping."),
    )
    monkeypatch.setattr(
        "src.api.routes.health._check_qdrant_health",
        lambda: DependencyHealthResponse(status="unhealthy", detail="connection refused"),
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "version": "0.1.0",
        "environment": "development",
        "redis": {
            "status": "healthy",
            "detail": "Redis responded to ping.",
        },
        "qdrant": {
            "status": "unhealthy",
            "detail": "connection refused",
        },
    }


def test_frontend_single_document_ingest_route_is_removed():
    """Removed frontend single-document ingest route should not be registered."""
    client = TestClient(app)
    response = client.post(
        "/documents/ingest",
        files={"file": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")},
    )

    assert response.status_code == 404


def test_frontend_batch_document_ingest_route_is_removed():
    """Removed frontend batch ingest route should not be registered."""
    client = TestClient(app)
    response = client.post(
        "/documents/ingest/batch",
        files={"file": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_service_returns_actual_workflow_intent():
    class FakeAgent:
        async def chat_with_state(self, message: str, files=None):
            assert message == "apa saja Profil Lulusan dan deskripsinya"
            assert files is None
            return type(
                "State",
                (),
                {
                    "current_intent": "query_document",
                    "messages": [AIMessage(content="Profil lulusan ada empat.")],
                },
            )()

    class FakeSessionManager:
        def get_or_create(self, thread_id=None):
            assert thread_id == "session-1"
            return FakeAgent(), "session-1"

    service = ChatRouteService(session_manager=FakeSessionManager())

    response = await service.handle_chat_message(
        "apa saja Profil Lulusan dan deskripsinya",
        thread_id="session-1",
    )

    assert response.response == "Profil lulusan ada empat."
    assert response.thread_id == "session-1"
    assert response.intent == "query_document"
    assert response.documents_processed == 0
    assert response.id  # UUID is generated


@pytest.mark.asyncio
async def test_chat_service_returns_workflow_citations():
    citation = Citation(
        id=1,
        document_id="doc-123",
        filename="kurikulum.pdf",
        section="3.1 Profil Lulusan",
        page=14,
        snippet="Profil lulusan dan deskripsinya",
    )

    class FakeAgent:
        async def chat_with_state(self, message: str, files=None):
            assert message == "apa saja Profil Lulusan dan deskripsinya"
            assert files is None
            return type(
                "State",
                (),
                {
                    "current_intent": "query_document",
                    "draft_response": "Profil lulusan ada empat.\n\nSources:\n[1] kurikulum.pdf",
                    "citations": [citation],
                    "messages": [],
                },
            )()

    class FakeSessionManager:
        def get_or_create(self, thread_id=None):
            assert thread_id == "session-1"
            return FakeAgent(), "session-1"

    service = ChatRouteService(session_manager=FakeSessionManager())

    response = await service.handle_chat_message(
        "apa saja Profil Lulusan dan deskripsinya",
        thread_id="session-1",
    )

    assert response.citations == [citation]
    assert response.response.endswith("[1] kurikulum.pdf")


@pytest.mark.asyncio
async def test_chat_stream_service_emits_structured_events():
    class FakeAgent:
        async def chat_with_state(self, message: str, files=None):
            raise AssertionError("stream flow should not fall back to chat_with_state")

        async def stream_chat(self, message: str, files=None) -> AsyncIterator[AgentStreamUpdate]:
            yield AgentStreamUpdate(
                kind="status",
                node="retrieval",
                payload={"retrieved_chunks": [{"id": "chunk-1"}]},
                state=type(
                    "State",
                    (),
                    {
                        "draft_response": None,
                        "messages": [],
                    },
                )(),
            )
            yield AgentStreamUpdate(
                kind="status",
                node="response_generation",
                payload={"draft_response": "Profil lulusan ada empat."},
                state=type(
                    "State",
                    (),
                    {
                        "draft_response": "Profil lulusan ada empat.",
                        "messages": [AIMessage(content="Profil lulusan ada empat.")],
                    },
                )(),
            )
            yield AgentStreamUpdate(
                kind="final",
                state=type(
                    "State",
                    (),
                    {
                        "draft_response": "Profil lulusan ada empat.",
                        "messages": [AIMessage(content="Profil lulusan ada empat.")],
                        "current_intent": "query_document",
                        "citations": [],
                        "error": None,
                    },
                )(),
            )

    class FakeSessionManager:
        def get_or_create(self, thread_id=None):
            return FakeAgent(), thread_id or "thread-1"

    service = ChatRouteService(session_manager=FakeSessionManager())

    events = [
        event async for event in service.stream_chat_message("profil lulusan", thread_id="thread-1")
    ]

    assert [event.event_type for event in events] == [
        "run.started",
        "run.status",
        "run.status",
        "message.delta",
        "run.completed",
    ]
    assert events[0].thread_id == "thread-1"
    assert events[-1].data["response"] == "Profil lulusan ada empat."
    assert events[-1].data["intent"] == "query_document"


def test_chat_stream_endpoint_returns_sse(monkeypatch):
    async def fake_stream(
        *, message: str, thread_id: str | None = None, last_event_id: str | None = None
    ):
        del message, last_event_id
        yield ChatStreamEvent(
            event_id="run-1:1",
            event_type="run.started",
            thread_id=thread_id or "thread-1",
            run_id="run-1",
            timestamp="2026-05-06T00:00:00Z",
            sequence=1,
            data={"message": "accepted", "resume_supported": True},
        )
        yield ChatStreamEvent(
            event_id="run-1:2",
            event_type="run.completed",
            thread_id=thread_id or "thread-1",
            run_id="run-1",
            timestamp="2026-05-06T00:00:01Z",
            sequence=2,
            data={
                "response": "done",
                "intent": "general_chat",
                "documents_processed": 0,
                "citations": [],
            },
        )

    monkeypatch.setattr("src.api.routes.chat.handle_chat_message_stream", fake_stream)

    client = TestClient(app)
    response = client.post("/chat/stream", json={"message": "hello", "thread_id": "thread-1"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run.started" in response.text
    assert "event: run.completed" in response.text
    assert '"thread_id": "thread-1"' in response.text


def test_chat_upload_stream_endpoint_rejects_empty_upload():
    client = TestClient(app)
    response = client.post("/chat/upload/stream", data={"message": "hello"})

    assert response.status_code == 422
