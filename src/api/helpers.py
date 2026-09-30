"""Common helpers for API routes."""

from fastapi import UploadFile

from src.utils.nodes.document_processing import DocumentProcessingNode, get_document_processor


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
