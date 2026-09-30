"""Deterministic doubles for external systems and agent boundaries.

The suite replaces only dependencies that are external to the request path:
the LLM provider, the OCR/indexing processor, and the agent runtime that wraps
the LangGraph workflow. Routes, guards, services, and response mapping always
run as production code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage

from src.services.contracts import AgentStreamUpdate
from src.services.response_generation import ResponseGenerationService
from src.services.session_registry import InMemorySessionManager
from src.utils.state import AgentState, DocumentType, DocumentUpload, ProcessingStatus


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


class FakeLLMProvider:
    """Callable LLM provider that counts how often it was resolved."""

    def __init__(self, response: str = "general_chat") -> None:
        self.calls = 0
        self.llm = FakeLLM(response)

    def __call__(self) -> FakeLLM:
        self.calls += 1
        return self.llm


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
