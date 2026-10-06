"""Document upload endpoints."""

import logging
import mimetypes
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.models import (
    DocumentDeleteResponse,
    DocumentIngestionResponse,
    DocumentListItem,
    ErrorResponse,
)
from src.api.services import handle_knowledge_base_ingestion
from src.utils.tools import VectorStoreTools
from src.utils.tools.ocr import PDF_MIME_TYPE, SUPPORTED_MIME_TYPES, GLMOCRTool

logger = logging.getLogger(__name__)
router = APIRouter()

PDF_MIME_ALIASES = {"application/x-pdf", "application/acrobat"}

UPLOAD_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    413: {"model": ErrorResponse, "description": "A file exceeds the size limit."},
    415: {"model": ErrorResponse, "description": "A file has an unsupported type."},
}


def _normalize_mime_type(content_type: str | None, filename: str) -> str | None:
    """Map the declared or guessed MIME type onto the OCR-supported set."""
    guessed_type, _ = mimetypes.guess_type(filename)
    for candidate in (content_type, guessed_type):
        if candidate in PDF_MIME_ALIASES:
            return PDF_MIME_TYPE
        if candidate in SUPPORTED_MIME_TYPES:
            return candidate
    return None


def _upload_size(file: UploadFile) -> int:
    """Get the upload size without reading the body."""
    if file.size is not None:
        return file.size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    return size


async def validate_document_upload_files(
    files: Annotated[
        list[UploadFile],
        File(description="One or more PDF or image files to ingest."),
    ],
) -> list[UploadFile]:
    """Reject unsupported or oversized uploads before ingestion."""
    for file in files:
        filename = file.filename or "upload"
        mime_type = _normalize_mime_type(file.content_type, filename)
        if mime_type is None:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Unsupported file type for '{filename}'. "
                "Only PDF, PNG, JPEG, GIF, and WEBP uploads are supported.",
            )

        max_bytes = (
            GLMOCRTool.MAX_PDF_BYTES
            if mime_type == PDF_MIME_TYPE
            else GLMOCRTool.MAX_IMAGE_BYTES
        )
        if _upload_size(file) > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Uploaded file '{filename}' is too large. "
                f"Maximum size for {mime_type} is {max_bytes / (1024 * 1024):.1f} MB.",
            )
    return files


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest documents",
    responses=UPLOAD_ERROR_RESPONSES,
)
async def upload_documents(
    files: Annotated[list[UploadFile], Depends(validate_document_upload_files)],
) -> list[DocumentIngestionResponse]:
    """Run OCR on the files and store the vectors in the retrieval index."""
    return await handle_knowledge_base_ingestion(files)


# Plain `def` on purpose: the vector store client is synchronous, so FastAPI
# runs these in a threadpool instead of blocking the event loop.
@router.get("/documents", summary="List ingested documents")
def list_documents() -> list[DocumentListItem]:
    """Summarize every document currently in the retrieval index."""
    entries = VectorStoreTools().list_documents()
    return [DocumentListItem.model_validate(entry) for entry in entries]


@router.delete(
    "/documents/by-filename/{filename:path}",
    summary="Delete an ingested document by filename",
    responses={
        404: {"model": ErrorResponse, "description": "No document with that filename."},
    },
)
def delete_document_by_filename(filename: str) -> DocumentDeleteResponse:
    """Delete all vector chunks and context records for the filename."""
    tools = VectorStoreTools()
    document_ids = tools.find_document_ids_by_filename(filename)
    if not document_ids:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No ingested document found with filename '{filename}'.",
        )

    for document_id in document_ids:
        tools.delete_document_chunks(document_id)
        logger.info(
            "Deleted document by filename",
            extra={"deleted_filename": filename, "document_id": document_id},
        )
    return DocumentDeleteResponse(document_id=min(document_ids), filename=filename)
