"""Service helpers shared across route modules."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from fastapi import UploadFile
from langchain_core.messages import AIMessage

from src.api.helpers import create_document_processor, read_upload_files
from src.api.models import (
    ChatResponse,
    ChatStreamEvent,
    ChatStreamEventType,
    DocumentIngestionResponse,
)
from src.api.sessions import get_or_create_agent
from src.services.contracts import ChatAgent, DocumentProcessor, SessionManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RunStreamState:
    """Shared replay state for a streamed run."""

    thread_id: str
    run_id: str
    task: asyncio.Task[None] | None = None
    events: list[ChatStreamEvent] = field(default_factory=list)
    subscribers: set[asyncio.Queue[ChatStreamEvent | None]] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    sequence: int = 0

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


class ChatRouteService:
    """Coordinate chat API requests with session-backed agents."""

    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager
        self._active_runs: dict[str, _RunStreamState] = {}
        self._run_lock = asyncio.Lock()

    async def handle_chat_message(
        self,
        message: str,
        thread_id: str | None = None,
        session_id: str | None = None,
    ) -> ChatResponse:
        """Send a plain chat message to an agent session."""
        resolved_thread_id = thread_id or session_id
        agent, active_thread_id = self._session_manager.get_or_create(resolved_thread_id)
        final_state = await agent.chat_with_state(message)
        return ChatResponse(
            response=self._extract_response_text(final_state),
            thread_id=active_thread_id,
            intent=final_state.current_intent,
            citations=self._extract_citations(final_state),
        )

    async def handle_chat_upload(
        self,
        message: str,
        files: list[UploadFile] | list[tuple[str, bytes]],
        thread_id: str | None = None,
        session_id: str | None = None,
    ) -> ChatResponse:
        """Send uploaded files to an agent session."""
        resolved_thread_id = thread_id or session_id
        agent, active_thread_id = self._session_manager.get_or_create(resolved_thread_id)
        file_tuples = await self._resolve_upload_file_tuples(files)
        final_state = await agent.chat_with_state(message, files=file_tuples)
        return ChatResponse(
            response=self._extract_response_text(final_state),
            thread_id=active_thread_id,
            intent=final_state.current_intent,
            documents_processed=len(file_tuples),
            citations=self._extract_citations(final_state),
        )

    async def stream_chat_message(
        self,
        message: str,
        thread_id: str | None = None,
        last_event_id: str | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Start or resume a streamed chat run."""
        run_state, is_new_run = await self._get_or_start_run(
            message=message,
            files=None,
            thread_id=thread_id,
            last_event_id=last_event_id,
        )
        del is_new_run
        async for event in self._subscribe_to_run(run_state, last_event_id):
            yield event

    async def stream_chat_upload(
        self,
        message: str,
        files: list[UploadFile] | list[tuple[str, bytes]],
        thread_id: str | None = None,
        last_event_id: str | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Start or resume a streamed upload chat run."""
        file_tuples = await self._resolve_upload_file_tuples(files)
        run_state, _ = await self._get_or_start_run(
            message=message,
            files=file_tuples,
            thread_id=thread_id,
            last_event_id=last_event_id,
        )
        async for event in self._subscribe_to_run(run_state, last_event_id):
            yield event

    def encode_sse_event(self, event: ChatStreamEvent) -> str:
        """Encode a stream event as one SSE frame."""
        payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=True)
        return f"id: {event.event_id}\nevent: {event.event_type}\ndata: {payload}\n\n"

    def _extract_response_text(self, final_state: Any) -> str:
        """Resolve the assistant response text from a workflow state."""
        if getattr(final_state, "draft_response", None):
            return str(final_state.draft_response)

        messages = list(getattr(final_state, "messages", []) or [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content)

        if messages:
            return str(messages[-1].content)
        return "I processed your request but have no response to provide."

    def _extract_citations(self, final_state: Any) -> list[Any]:
        """Resolve structured citations from a workflow state or test double."""
        return list(getattr(final_state, "citations", []) or [])

    async def _resolve_upload_file_tuples(
        self,
        files: list[UploadFile] | list[tuple[str, bytes]],
    ) -> list[tuple[str, bytes]]:
        """Normalize upload payloads from HTTP and adapter-based channels."""
        if not files:
            return []

        first_file = files[0]
        if isinstance(first_file, UploadFile):
            return await read_upload_files(files)  # type: ignore[arg-type]

        resolved_files: list[tuple[str, bytes]] = []
        for filename, contents in files:  # type: ignore[misc]
            upload = UploadFile(filename=filename, file=BytesIO(contents))
            resolved_files.append((upload.filename or "upload", await upload.read()))
        return resolved_files

    async def _get_or_start_run(
        self,
        *,
        message: str,
        files: list[tuple[str, bytes]] | None,
        thread_id: str | None,
        last_event_id: str | None,
    ) -> tuple[_RunStreamState, bool]:
        """Return an existing resumable run or start a new one."""
        agent, active_thread_id = self._session_manager.get_or_create(thread_id)

        async with self._run_lock:
            existing = self._active_runs.get(active_thread_id)
            if existing is not None and (not existing.done.is_set() or last_event_id):
                return existing, False

            run_id = str(uuid.uuid4())
            run_state = _RunStreamState(thread_id=active_thread_id, run_id=run_id)
            self._active_runs[active_thread_id] = run_state
            run_state.task = asyncio.create_task(
                self._produce_run_events(
                    agent=agent,
                    thread_id=active_thread_id,
                    run_id=run_id,
                    message=message,
                    files=files,
                )
            )
            return run_state, True

    async def _subscribe_to_run(
        self,
        run_state: _RunStreamState,
        last_event_id: str | None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Replay buffered events and then stream live events."""
        last_sequence = self._parse_event_sequence(last_event_id)
        replay_events = [event for event in run_state.events if event.sequence > last_sequence]

        queue: asyncio.Queue[ChatStreamEvent | None] = asyncio.Queue()
        run_state.subscribers.add(queue)
        try:
            for event in replay_events:
                yield event

            while True:
                if run_state.done.is_set() and queue.empty():
                    break
                event = await queue.get()
                if event is None:
                    break
                if event.sequence <= last_sequence:
                    continue
                yield event
        finally:
            run_state.subscribers.discard(queue)

    async def _produce_run_events(
        self,
        *,
        agent: ChatAgent,
        thread_id: str,
        run_id: str,
        message: str,
        files: list[tuple[str, bytes]] | None,
    ) -> None:
        """Drive agent streaming and broadcast normalized API events."""
        run_state = self._active_runs[thread_id]
        emitted_text = ""
        final_state: Any | None = None

        await self._append_event(
            run_state,
            "run.started",
            {
                "message": "user request accepted",
                "resume_supported": True,
            },
        )

        try:
            async for update in agent.stream_chat(message, files=files):
                if update.kind == "final":
                    final_state = update.state
                    continue
                if update.kind != "status":
                    continue

                await self._append_event(
                    run_state,
                    "run.status",
                    {
                        "stage": update.node or "workflow",
                        "label": self._format_stage_label(update.node),
                        "meta": update.payload,
                    },
                )

                if update.state is None:
                    continue
                response_text = self._extract_stream_text(update.state)
                delta = self._next_delta(emitted_text, response_text)
                if delta:
                    emitted_text = response_text
                    await self._append_event(
                        run_state,
                        "message.delta",
                        {
                            "text": delta,
                            "index": len(emitted_text),
                        },
                    )

            if final_state is None:
                final_state = await agent.chat_with_state(message, files=files)
            response_text = self._extract_response_text(final_state)
            delta = self._next_delta(emitted_text, response_text)
            if delta:
                emitted_text = response_text
                await self._append_event(
                    run_state,
                    "message.delta",
                    {
                        "text": delta,
                        "index": len(emitted_text),
                    },
                )

            if getattr(final_state, "error", None):
                await self._append_event(
                    run_state,
                    "run.failed",
                    {
                        "code": "STREAM_EXECUTION_ERROR",
                        "message": str(final_state.error),
                        "retryable": True,
                    },
                )
            else:
                await self._append_event(
                    run_state,
                    "run.completed",
                    {
                        "response": response_text,
                        "intent": getattr(final_state, "current_intent", None),
                        "documents_processed": len(files or []),
                        "citations": [
                            citation.model_dump(mode="json")
                            if hasattr(citation, "model_dump")
                            else citation
                            for citation in self._extract_citations(final_state)
                        ],
                    },
                )
        except Exception as exc:
            logger.error("Streaming run failed: %s", exc, exc_info=True)
            await self._append_event(
                run_state,
                "run.failed",
                {
                    "code": "STREAM_EXECUTION_ERROR",
                    "message": str(exc),
                    "retryable": True,
                },
            )
        finally:
            run_state.done.set()
            for subscriber in list(run_state.subscribers):
                await subscriber.put(None)

    async def _append_event(
        self,
        run_state: _RunStreamState,
        event_type: ChatStreamEventType,
        data: dict[str, Any],
    ) -> ChatStreamEvent:
        """Append an event to replay storage and publish it to subscribers."""
        sequence = run_state.next_sequence()
        event = ChatStreamEvent(
            event_id=f"{run_state.run_id}:{sequence}",
            event_type=event_type,
            thread_id=run_state.thread_id,
            run_id=run_state.run_id,
            timestamp=datetime.now(UTC),
            sequence=sequence,
            data=data,
        )
        run_state.events.append(event)
        for subscriber in list(run_state.subscribers):
            await subscriber.put(event)
        return event

    def _parse_event_sequence(self, event_id: str | None) -> int:
        """Parse the numeric sequence from a replay cursor."""
        if not event_id:
            return 0
        if ":" not in event_id:
            try:
                return int(event_id)
            except ValueError:
                return 0
        _, sequence_text = event_id.rsplit(":", 1)
        try:
            return int(sequence_text)
        except ValueError:
            return 0

    def _next_delta(self, emitted_text: str, candidate_text: str) -> str:
        """Return only the new suffix for a streamed assistant response."""
        if not candidate_text:
            return ""
        if candidate_text.startswith(emitted_text):
            return candidate_text[len(emitted_text) :]
        return candidate_text

    def _format_stage_label(self, stage: str | None) -> str:
        """Turn a node name into a readable status label."""
        if not stage:
            return "Processing request"
        return stage.replace("_", " ").strip().title()

    def _extract_stream_text(self, state: Any) -> str:
        """Return only explicit assistant text for incremental message events."""
        if getattr(state, "draft_response", None):
            return str(state.draft_response)
        messages = list(getattr(state, "messages", []) or [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content)
        return ""


class DocumentRouteService:
    """Coordinate document-upload API requests."""

    def __init__(
        self,
        processor_factory: Callable[[], DocumentProcessor],
    ):
        self._processor_factory = processor_factory

    async def handle_knowledge_base_ingestion(
        self,
        files: list[UploadFile],
    ) -> list[DocumentIngestionResponse]:
        """Ingest uploaded documents into the retrieval stack."""
        processor = self._processor_factory()
        responses: list[DocumentIngestionResponse] = []

        for file in files:
            try:
                contents = await file.read()
                filename = file.filename or "upload"
                document = await processor.ingest_upload(contents, filename)
                responses.append(self._build_ingestion_response(document))
            except Exception as exc:
                logger.error(
                    "Knowledge-base ingestion failed for %s: %s",
                    file.filename,
                    exc,
                    exc_info=True,
                )
                responses.append(
                    DocumentIngestionResponse(
                        success=False,
                        document_id="error",
                        filename=file.filename or "upload",
                        document_type="other",
                        status="failed",
                        error=str(exc),
                    )
                )

        return responses

    def _build_ingestion_response(self, document: Any) -> DocumentIngestionResponse:
        """Build the API response for a processed document."""
        return DocumentIngestionResponse(
            success=document.processing_status.value == "completed",
            document_id=document.document_id,
            filename=document.filename,
            document_type=document.document_type.value,
            status=document.processing_status.value,
            document_title=document.document_title,
            parsed_pages=document.parsed_pages,
            failed_pages=list(document.failed_pages or []),
            ocr_warnings=list(document.ocr_warnings or []),
            quality_warning=document.quality_warning,
            error=document.processing_error,
        )


class _SessionManagerAdapter:
    """Adapt the legacy function-based session access to the SessionManager protocol."""

    def get_or_create(self, thread_id: str | None = None) -> tuple[ChatAgent, str]:
        """Delegate session access to the existing module function."""
        return get_or_create_agent(thread_id)


chat_service = ChatRouteService(session_manager=_SessionManagerAdapter())
document_service = DocumentRouteService(
    processor_factory=create_document_processor,
)


async def handle_chat_message(
    message: str,
    thread_id: str | None = None,
    session_id: str | None = None,
) -> ChatResponse:
    """Send a plain chat message to an agent session."""
    return await chat_service.handle_chat_message(
        message,
        thread_id=thread_id,
        session_id=session_id,
    )


async def handle_chat_upload(
    message: str,
    files: list[UploadFile] | list[tuple[str, bytes]],
    thread_id: str | None = None,
    session_id: str | None = None,
) -> ChatResponse:
    """Send uploaded files to an agent session."""
    return await chat_service.handle_chat_upload(
        message=message,
        files=files,
        thread_id=thread_id,
        session_id=session_id,
    )


async def handle_chat_message_stream(
    message: str,
    thread_id: str | None = None,
    last_event_id: str | None = None,
) -> AsyncIterator[ChatStreamEvent]:
    """Stream a plain chat message."""
    async for event in chat_service.stream_chat_message(
        message=message,
        thread_id=thread_id,
        last_event_id=last_event_id,
    ):
        yield event


async def handle_chat_upload_stream(
    message: str,
    files: list[UploadFile] | list[tuple[str, bytes]],
    thread_id: str | None = None,
    last_event_id: str | None = None,
) -> AsyncIterator[ChatStreamEvent]:
    """Stream a chat message with uploads."""
    async for event in chat_service.stream_chat_upload(
        message=message,
        files=files,
        thread_id=thread_id,
        last_event_id=last_event_id,
    ):
        yield event


async def handle_knowledge_base_ingestion(
    files: list[UploadFile],
) -> list[DocumentIngestionResponse]:
    """Ingest uploaded documents into the retrieval stack."""
    return await document_service.handle_knowledge_base_ingestion(files)
