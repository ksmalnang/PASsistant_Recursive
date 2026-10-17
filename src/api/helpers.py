"""Common helpers for API routes."""

from fastapi import HTTPException, Request, UploadFile, status

from src.utils.nodes.document_processing import DocumentProcessingNode, get_document_processor


def enforce_rate_limit(request: Request) -> None:
    """Reject a request when its client exceeds the configured rate limit."""
    client = request.client.host if request.client else "unknown"
    if not request.app.state.rate_limiter.allow(client):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limit exceeded. Please try again later.",
        )


async def read_upload_files(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Read uploaded files into memory as filename/content tuples."""
    file_tuples: list[tuple[str, bytes]] = []
    for file in files:
        contents = await file.read()
        file_tuples.append((file.filename or "upload", contents))
    return file_tuples


def create_document_processor() -> DocumentProcessingNode:
    """Return the process-wide document processor."""
    return get_document_processor()
