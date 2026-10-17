"""Deterministic doubles for external systems and agent boundaries.

The suite replaces only dependencies that are external to the request path:
the LLM provider, the OCR/indexing processor, and the agent runtime that wraps
the LangGraph workflow. Routes, guards, services, and response mapping always
run as production code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from src.services.contracts import AgentStreamUpdate
from src.services.response_generation import ResponseGenerationService
from src.services.session_registry import InMemorySessionManager
from src.utils.state import (
    AgentState,
    DocumentType,
    DocumentUpload,
    OCRResult,
    ProcessingStatus,
)


class FakeClock:
    """Controllable monotonic clock for rate-limiter tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start

    def monotonic(self) -> float:
        """Return the current fake time in seconds."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self._now += seconds


class FakeLLM:
    """Scripted LLM double that records prompts and returns a fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[Sequence[Any]] = []

    def invoke(self, messages: Sequence[Any], /, **kwargs: Any) -> SimpleNamespace:
        """Record the prompt and return a LangChain-shaped message."""
        del kwargs
        self.prompts.append(messages)
        return SimpleNamespace(content=self.response)

    async def ainvoke(self, messages: Sequence[Any], /, **kwargs: Any) -> SimpleNamespace:
        """Async counterpart used by the classifier fallback path."""
        return self.invoke(messages, **kwargs)


class FakeLLMProvider:
    """Callable LLM provider that counts how often it was resolved."""

    def __init__(self, response: str = "general_chat") -> None:
        self.calls = 0
        self.llm = FakeLLM(response)

    def __call__(self) -> FakeLLM:
        self.calls += 1
        return self.llm


class FakeDecisionClient:
    """Decision client double recording questions and returning scripted answers."""

    def __init__(self, choice: str = "general_chat", confidence: float = 0.9) -> None:
        self.choice = choice
        self.confidence = confidence
        self.calls = 0
        self.states: list[Any] = []

    async def ask_choice(self, *args: Any, **kwargs: Any) -> Any:
        from src.clients.decision import ChoiceAnswer

        self.calls += 1
        self.states.append(kwargs.get("state"))
        return ChoiceAnswer(
            choice=self.choice,
            confidence=self.confidence,
            probabilities={self.choice: self.confidence},
        )


class RecordingChatAgent:
    """ChatAgent double that records turns and replays scripted states."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or "test-agent-session"
        self.chat_calls: list[tuple[str, list[tuple[str, bytes]] | None]] = []
        self.stream_calls: list[tuple[str, list[tuple[str, bytes]] | None]] = []
        self.final_state: AgentState = AgentState(session_id=self.session_id)
        self.stream_script: list[AgentStreamUpdate] | None = None
        self._doc_processor = FakeDocumentProcessor()

    @property
    def thread_id(self) -> str:
        """Public thread identifier alias."""
        return self.session_id

    @property
    def doc_processor(self) -> FakeDocumentProcessor:
        """Document processor double used for uploads."""
        return self._doc_processor

    async def chat(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> str:
        """Return the scripted final response text."""
        state = await self.chat_with_state(message, files)
        return str(state.draft_response or "")

    async def chat_with_state(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> AgentState:
        """Record the turn and return the scripted final state."""
        self.chat_calls.append((message, files))
        return self.final_state

    async def stream_chat(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> AsyncIterator[AgentStreamUpdate]:
        """Yield scripted stream updates, or a default status/final sequence."""
        self.stream_calls.append((message, files))
        if self.stream_script is not None:
            for update in self.stream_script:
                yield update
            return

        final_state = self.final_state
        yield AgentStreamUpdate(kind="status", node="retrieval", payload={"retrieved_chunks": []})
        yield AgentStreamUpdate(
            kind="status",
            node="response_generation",
            payload={},
            state=final_state,
        )
        yield AgentStreamUpdate(kind="final", state=final_state)


class LlmBackedChatAgent:
    """ChatAgent double that runs the real response pipeline with a fake LLM.

    The LLM is replaced at the provider boundary, so response generation,
    citation footers, and the output guard all execute production code.
    """

    def __init__(self, llm: FakeLLM, session_id: str | None = None) -> None:
        self.session_id = session_id or "llm-backed-session"
        self._service = ResponseGenerationService(llm_provider=lambda: llm)
        self._doc_processor = FakeDocumentProcessor()

    @property
    def thread_id(self) -> str:
        """Public thread identifier alias."""
        return self.session_id

    @property
    def doc_processor(self) -> FakeDocumentProcessor:
        """Document processor double used for uploads."""
        return self._doc_processor

    async def chat(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> str:
        """Return the generated response text."""
        state = await self.chat_with_state(message, files)
        return str(state.draft_response or "")

    async def chat_with_state(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> AgentState:
        """Run the real response-generation service for the turn."""
        del files
        state = AgentState(
            session_id=self.session_id,
            current_intent="query_document",
            messages=[HumanMessage(content=message)],
        )
        update = self._service.generate(state)
        return state.model_copy(update=update)


class FakeDocumentProcessor:
    """DocumentProcessor double that simulates OCR and indexing outcomes."""

    def __init__(
        self,
        *,
        document_type: DocumentType = DocumentType.POLICY,
        title: str = "Academic Policy",
    ) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.indexed_document_ids: list[str] = []
        self.fail_with: Exception | None = None
        self.document_type = document_type
        self.title = title

    def prepare_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Prepare a document for later processing without indexing."""
        return self._build_document(file_bytes, filename, status=ProcessingStatus.PENDING)

    async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Record the upload, optionally fail, then simulate indexing exactly once."""
        self.uploads.append((filename, file_bytes))
        if self.fail_with is not None:
            raise self.fail_with

        document = self._build_document(file_bytes, filename, status=ProcessingStatus.COMPLETED)
        self.indexed_document_ids.append(document.document_id)
        return document

    def _build_document(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        status: ProcessingStatus,
    ) -> DocumentUpload:
        return DocumentUpload(
            document_id=f"doc-{len(self.uploads):04d}",
            filename=filename,
            file_path=f"data/raw/{filename}",
            document_type=self.document_type,
            mime_type="application/pdf",
            file_size=len(file_bytes),
            processing_status=status,
            processing_error=(str(self.fail_with) if self.fail_with is not None else None),
            document_title=self.title,
            parsed_pages=1 if status is ProcessingStatus.COMPLETED else None,
        )


class ChatBackend:
    """Wire the real in-memory session manager to agent doubles.

    ``agent_factory`` can be replaced per test to install a different double,
    mirroring how an LLM-backed agent would be created in production.
    ``final_state_factory`` scripts the workflow state recorded agents replay.
    """

    def __init__(self) -> None:
        self.agents: list[Any] = []
        self.agent_factory: Callable[[str | None], Any] | None = None
        self.final_state_factory: Callable[[str], AgentState] | None = None
        self.manager = InMemorySessionManager(agent_factory=self._create_agent)

    def _create_agent(self, thread_id: str | None) -> Any:
        if self.agent_factory is not None:
            agent = self.agent_factory(thread_id)
        else:
            generated_id = thread_id or f"generated-thread-{len(self.agents) + 1}"
            agent = RecordingChatAgent(session_id=generated_id)
            if self.final_state_factory is not None:
                agent.final_state = self.final_state_factory(generated_id)
        self.agents.append(agent)
        return agent

    @property
    def last_agent(self) -> Any:
        """Return the most recently created agent double."""
        return self.agents[-1]


def make_ocr_result(
    *,
    page_texts: list[str] | None = None,
    statuses: list[str] | None = None,
) -> OCRResult:
    """Build a realistic multi-page OCR result for ingestion/reindex tests."""
    texts = page_texts if page_texts is not None else ["Halo dunia", "Tabel nilai"]
    page_statuses = statuses if statuses is not None else ["success"] * len(texts)
    layouts = [
        [
            {
                "index": 0,
                "label": "text",
                "bbox_2d": [0, 0, 100, 20],
                "content": text,
                "height": 20,
                "width": 100,
            }
        ]
        for text in texts
    ]
    page_results = [
        {
            "page_index": index,
            "status": page_statuses[index],
            "text_length": len(text),
            "layout_blocks": len(layouts[index]),
            "error": None if page_statuses[index] != "failed" else "provider error",
        }
        for index, text in enumerate(texts)
    ]
    return OCRResult(
        text="\n\n".join(text for text in texts if text),
        text_quality_score=1.0,
        num_pages=len(texts),
        layout_details=layouts,
        parsed_pages=sum(1 for status in page_statuses if status == "success"),
        failed_pages=[i for i, status in enumerate(page_statuses) if status == "failed"],
        page_results=page_results,
        ocr_warnings=[],
        page_texts=texts,
    )


class FakeTextExtractor:
    """DocumentTextExtractor double returning a fixed OCRResult and recording calls."""

    def __init__(self, result: OCRResult | None = None) -> None:
        self.result = result if result is not None else make_ocr_result()
        self.calls: list[tuple[str, DocumentType]] = []

    async def extract_text(
        self,
        file_path: str,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> OCRResult:
        self.calls.append((file_path, document_type))
        return self.result


class FakeUploadPreparer:
    """DocumentUploadPreparer double that stores bytes under a temp raw directory."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_bytes: bytes, original_filename: str) -> DocumentUpload:
        document_id = f"doc-{uuid4().hex[:8]}"
        storage_path = self.raw_dir / f"{document_id}.pdf"
        storage_path.write_bytes(file_bytes)
        return DocumentUpload(
            document_id=document_id,
            filename=original_filename,
            file_path=str(storage_path),
            document_type=DocumentType.POLICY,
            mime_type="application/pdf",
            file_size=len(file_bytes),
        )


class FakeIndexer:
    """DocumentIndexer double recording documents and replace flags."""

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.documents: list[DocumentUpload] = []
        self.replace_flags: list[bool] = []
        self.fail_for = fail_for or set()

    async def index_document(
        self,
        document: DocumentUpload,
        *,
        replace_existing: bool = False,
    ) -> SimpleNamespace:
        self.documents.append(document)
        self.replace_flags.append(replace_existing)
        if document.document_id in self.fail_for:
            raise RuntimeError(f"indexing exploded for {document.document_id}")
        return SimpleNamespace(
            chunk_count=1,
            indexed_chars=len(document.extracted_text or ""),
            issues=[],
        )
