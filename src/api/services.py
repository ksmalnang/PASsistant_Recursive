"""Service helpers shared across route modules."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import Any

from fastapi import UploadFile
from langchain_core.messages import AIMessage

from src.api.helpers import create_document_processor, read_upload_files
from src.api.models import (
    ChatResponse,
    ChatStreamEvent,
    DocumentIngestionResponse,
)
from src.api.sessions import session_manager
from src.services.contracts import ChatAgent, DocumentProcessor, SessionManager

logger = logging.getLogger(__name__)

UploadInput = list[UploadFile] | list[tuple[str, bytes]]
FileTuples = list[tuple[str, bytes]]


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _assistant_text(state: Any) -> str | None:
    """Latest explicit assistant text in a workflow state, or None."""
    if getattr(state, "draft_response", None):
        return str(state.draft_response)
    for message in reversed(getattr(state, "messages", None) or []):
        if isinstance(message, AIMessage):
            return str(message.content)
    return None


def _response_text(state: Any) -> str:
    """Final reply text, falling back to the last message of any kind."""
    text = _assistant_text(state)
    if text is not None:
        return text
    messages = getattr(state, "messages", None) or []
    if messages:
        return str(messages[-1].content)
    return "I processed your request but have no response to provide."


def _citations(state: Any) -> list[Any]:
    return list(getattr(state, "citations", None) or [])


def _next_delta(emitted_text: str, candidate_text: str) -> str:
    """Return only the new suffix of a streamed assistant response."""
    if not candidate_text:
        return ""
    if candidate_text.startswith(emitted_text):
        return candidate_text[len(emitted_text) :]
    return candidate_text


def _stage_label(stage: str | None) -> str:
    if not stage:
        return "Processing request"
    return stage.replace("_", " ").strip().title()


def _cursor_sequence(event_id: str | None, run_id: str) -> int:
    """Sequence number from a resume cursor ("<run_id>:<n>" or a bare "<n>").

    A cursor that belongs to a different run is ignored, so it can't make a
    fresh run skip its first events.
    """
    if not event_id:
        return 0
    prefix, _, sequence = event_id.rpartition(":")
    if prefix and prefix != run_id:
        return 0
    try:
        return int(sequence)
    except ValueError:
        return 0


async def _to_file_tuples(files: UploadInput) -> FileTuples:
    """Normalize uploads from HTTP (UploadFile) and adapters ((name, bytes)).

    Check for the tuple form instead of for `UploadFile`: the multipart parser
    hands back Starlette's UploadFile, which is not a `fastapi.UploadFile`.
    """
    if not files:
        return []
    if isinstance(files[0], tuple):
        return [(name or "upload", data) for name, data in files]  # type: ignore[misc]
    return await read_upload_files(files)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
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


class ChatRouteService:
    """Coordinate chat API requests with session-backed agents."""

    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager
        self._active_runs: dict[str, _RunStreamState] = {}
        self._run_lock = asyncio.Lock()

    async def handle_chat_message(
        self, message: str, thread_id: str | None = None
    ) -> ChatResponse:
        """Send a plain chat message to an agent session."""
        return await self._chat(message, None, thread_id)

    async def handle_chat_upload(
        self, message: str, files: UploadInput, thread_id: str | None = None
    ) -> ChatResponse:
        """Send a chat message with files to an agent session."""
        return await self._chat(message, await _to_file_tuples(files), thread_id)

    def stream_chat_message(
        self,
        message: str,
        thread_id: str | None = None,
        last_event_id: str | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Start or resume a streamed chat run."""
        return self._stream(message, None, thread_id, last_event_id)

    def stream_chat_upload(
        self,
        message: str,
        files: UploadInput,
        thread_id: str | None = None,
        last_event_id: str | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Start or resume a streamed chat run that includes files."""
        return self._stream(message, files, thread_id, last_event_id)

    def encode_sse_event(self, event: ChatStreamEvent) -> str:
        """Encode a stream event as one SSE frame."""
        return (
            f"id: {event.event_id}\nevent: {event.event_type}\n"
            f"data: {event.model_dump_json()}\n\n"
        )

    async def _chat(
        self, message: str, files: FileTuples | None, thread_id: str | None
    ) -> ChatResponse:
        agent, active_thread_id = self._session_manager.get_or_create(thread_id)
        state = await agent.chat_with_state(message, files=files)
        return ChatResponse(
            response=_response_text(state),
            thread_id=active_thread_id,
            intent=state.current_intent,
            documents_processed=len(files or []),
            citations=_citations(state),
        )

    async def _stream(
        self,
        message: str,
        files: UploadInput | None,
        thread_id: str | None,
        last_event_id: str | None,
    ) -> AsyncIterator[ChatStreamEvent]:
        file_tuples = await _to_file_tuples(files) if files else None
        run_state = await self._start_or_resume_run(
            message=message,
            files=file_tuples,
            thread_id=thread_id,
            last_event_id=last_event_id,
        )
        async for event in self._subscribe_to_run(run_state, last_event_id):
            yield event

    async def _start_or_resume_run(
        self,
        *,
        message: str,
        files: FileTuples | None,
        thread_id: str | None,
        last_event_id: str | None,
    ) -> _RunStreamState:
        """Return an existing resumable run or start a new one."""
        agent, active_thread_id = self._session_manager.get_or_create(thread_id)

        async with self._run_lock:
            # Forget runs whose session no longer exists.
            self._active_runs = {
                thread: run
                for thread, run in self._active_runs.items()
                if self._session_manager.contains(thread)
            }
            existing = self._active_runs.get(active_thread_id)
            if existing is not None and (not existing.done.is_set() or last_event_id):
                return existing

            run_state = _RunStreamState(
                thread_id=active_thread_id, run_id=str(uuid.uuid4())
            )
            self._active_runs[active_thread_id] = run_state
            run_state.task = asyncio.create_task(
                self._produce_run_events(
                    run_state=run_state, agent=agent, message=message, files=files
                )
            )
            return run_state

    async def _subscribe_to_run(
        self, run_state: _RunStreamState, last_event_id: str | None
    ) -> AsyncIterator[ChatStreamEvent]:
        """Replay buffered events, then stream live ones."""
        last_sequence = _cursor_sequence(last_event_id, run_state.run_id)
        replay = [e for e in run_state.events if e.sequence > last_sequence]

        queue: asyncio.Queue[ChatStreamEvent | None] = asyncio.Queue()
        run_state.subscribers.add(queue)
        try:
            for event in replay:
                yield event
            while not (run_state.done.is_set() and queue.empty()):
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            run_state.subscribers.discard(queue)

    async def _produce_run_events(
        self,
        *,
        run_state: _RunStreamState,
        agent: ChatAgent,
        message: str,
        files: FileTuples | None,
    ) -> None:
        """Drive agent streaming and broadcast normalized API events."""
        append = partial(self._append_event, run_state)
        emitted_text = ""
        final_state: Any | None = None

        async def emit_delta(response_text: str) -> None:
            nonlocal emitted_text
            delta = _next_delta(emitted_text, response_text)
            if delta:
                emitted_text = response_text
                await append("message.delta", {"text": delta, "index": len(emitted_text)})

        async def fail(reason: str) -> None:
            await append(
                "run.failed",
                {"code": "STREAM_EXECUTION_ERROR", "message": reason, "retryable": True},
            )

        await append(
            "run.started",
            {"message": "user request accepted", "resume_supported": True},
        )

        try:
            async for update in agent.stream_chat(message, files=files):
                if update.kind == "final":
                    final_state = update.state
                elif update.kind == "status":
                    await append(
                        "run.status",
                        {
                            "stage": update.node or "workflow",
                            "label": _stage_label(update.node),
                            "meta": update.payload,
                        },
                    )
                    if update.state is not None:
                        await emit_delta(_assistant_text(update.state) or "")

            if final_state is None:
                final_state = await agent.chat_with_state(message, files=files)
            response_text = _response_text(final_state)
            await emit_delta(response_text)

            if error := getattr(final_state, "error", None):
                await fail(str(error))
            else:
                await append(
                    "run.completed",
                    {
                        "response": response_text,
                        "intent": getattr(final_state, "current_intent", None),
                        "documents_processed": len(files or []),
                        "citations": [
                            c.model_dump(mode="json") if hasattr(c, "model_dump") else c
                            for c in _citations(final_state)
                        ],
                    },
                )
        except Exception:
            # Details go to the logs, not to the client.
            logger.error("Streaming run failed", exc_info=True)
            await fail("The chat run failed unexpectedly.")
        finally:
            run_state.done.set()
            for subscriber in list(run_state.subscribers):
                await subscriber.put(None)

    async def _append_event(
        self,
        run_state: _RunStreamState,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Store an event for replay and publish it to live subscribers."""
        run_state.sequence += 1
        event = ChatStreamEvent(
            event_id=f"{run_state.run_id}:{run_state.sequence}",
            event_type=event_type,  # type: ignore[arg-type]
            thread_id=run_state.thread_id,
            run_id=run_state.run_id,
            timestamp=datetime.now(UTC),
            sequence=run_state.sequence,
            data=data,
        )
        run_state.events.append(event)
        for subscriber in list(run_state.subscribers):
            await subscriber.put(event)


chat_service = ChatRouteService(session_manager=session_manager)


# --------------------------------------------------------------------------- #
# Document ingestion
# --------------------------------------------------------------------------- #
async def _ingest_one(
    processor: DocumentProcessor, file: UploadFile
) -> DocumentIngestionResponse:
    filename = file.filename or "upload"
    try:
        document = await processor.ingest_upload(await file.read(), filename)
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", filename, exc, exc_info=True)
        return DocumentIngestionResponse(
            success=False,
            document_id="error",
            filename=filename,
            document_type="other",
            status="failed",
            error=str(exc),
        )
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


async def handle_knowledge_base_ingestion(
    files: list[UploadFile],
) -> list[DocumentIngestionResponse]:
    """Ingest uploaded documents into the retrieval stack, one file at a time.

    A failing file is reported in its own response and doesn't stop the rest.
    """
    processor = create_document_processor()
    return [await _ingest_one(processor, file) for file in files]
