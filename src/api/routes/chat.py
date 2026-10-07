"""Chat endpoints.

One chat endpoint (plus its streaming twin). Files are optional: send
`multipart/form-data` with `message`, optional `thread_id`, optional `files`.
"""

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from src.api.models import ChatInput, ChatResponse, ErrorResponse
from src.api.services import chat_service
from src.guardrails.input_guard import InputGuard

_guard = InputGuard()

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "The request payload is invalid."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
}


def _enforce_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    if not request.app.state.rate_limiter.allow(client):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limit exceeded. Please try again later.",
        )


router = APIRouter(
    dependencies=[Depends(_enforce_rate_limit)],
    responses=ERROR_RESPONSES,
)


async def _chat_input(
    message: Annotated[str, Form()],
    thread_id: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> ChatInput:
    """Parse the form and run the input guard once, for every chat route."""
    result = _guard.validate(message)
    if not result.safe:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Message rejected: {result.reason}"
        )
    return ChatInput(
        message=result.sanitized or message.strip(),
        thread_id=thread_id,
        # Browsers/Swagger send an empty file part when nothing is selected.
        files=[f for f in files or [] if f.filename],
    )


ChatInputDep = Annotated[ChatInput, Depends(_chat_input)]


def _sse(events: AsyncIterator[Any]) -> StreamingResponse:
    async def encoded() -> AsyncIterator[str]:
        async for event in events:
            yield chat_service.encode_sse_event(event)

    return StreamingResponse(encoded(), media_type="text/event-stream")


@router.post("/chat", summary="Send a chat message (files optional)")
async def chat(data: ChatInputDep) -> ChatResponse:
    """Chat with the agent. Attached files are used as context for this turn only
    (they are not added to the shared retrieval index)."""
    if data.files:
        return await chat_service.handle_chat_upload(
            message=data.message, files=data.files, thread_id=data.thread_id
        )
    return await chat_service.handle_chat_message(
        message=data.message, thread_id=data.thread_id
    )


@router.post("/chat/stream", summary="Stream a chat message (files optional)")
async def chat_stream(data: ChatInputDep, request: Request) -> StreamingResponse:
    """Same as `/chat`, but replies with SSE events."""
    last_event_id = request.headers.get("Last-Event-ID") or request.query_params.get(
        "last_event_id"
    )
    if data.files:
        events = chat_service.stream_chat_upload(
            message=data.message,
            files=data.files,
            thread_id=data.thread_id,
            last_event_id=last_event_id,
        )
    else:
        events = chat_service.stream_chat_message(
            message=data.message,
            thread_id=data.thread_id,
            last_event_id=last_event_id,
        )
    return _sse(events)
