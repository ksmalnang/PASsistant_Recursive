"""Document upload endpoints."""

import logging
import mimetypes
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.api.models import (
    DocumentDeleteResponse,
    DocumentIngestionResponse,
    ErrorResponse,
)
from src.api.services import handle_knowledge_base_ingestion
from src.utils.tools import VectorStoreTools
from src.utils.tools.ocr import PDF_MIME_TYPE, SUPPORTED_MIME_TYPES, GLMOCRTool

logger = logging.getLogger(__name__)
router = APIRouter()


class DocumentListItem(BaseModel):
    """Summary of an ingested document in the knowledge base."""

    document_id: str
    filename: str
    chunks: int
    doc_title: str | None = None


UploadFiles = Annotated[
    list[UploadFile],
    File(
        ...,
        description="One or more PDF or image files to ingest into the retrieval index.",
    ),
]
PDF_MIME_ALIASES = {"application/x-pdf", "application/acrobat"}
DOCUMENT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The upload request is missing a required file payload.",
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "One or more uploaded files exceed the supported size limit.",
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": "One or more uploaded files use an unsupported MIME type.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "The server failed to process the uploaded documents.",
    },
}


def _normalize_mime_type(content_type: str | None, filename: str | None) -> str | None:
    """Normalize declared or inferred MIME types into the OCR-supported set."""
    candidates = [content_type]
    guessed_type, _ = mimetypes.guess_type(filename or "")
    candidates.append(guessed_type)

    for candidate in candidates:
        if not candidate:
            continue
        if candidate in PDF_MIME_ALIASES:
            return PDF_MIME_TYPE
        if candidate in SUPPORTED_MIME_TYPES:
            return candidate
    return None


def _resolve_upload_size(file: UploadFile) -> int:
    """Resolve upload size without consuming the request body stream."""
    size = getattr(file, "size", None)
    if size is not None:
        return int(size)

    stream = file.file
    current_position = stream.tell()
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(current_position)
    return int(size)


def _validate_files(files: list[UploadFile]) -> list[UploadFile]:
    """Validate multipart uploads before downstream processing."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be uploaded.",
        )

    for file in files:
        filename = file.filename or "upload"
        mime_type = _normalize_mime_type(file.content_type, filename)
        if mime_type is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported file type for '{filename}'. "
                    "Only PDF, PNG, JPEG, GIF, and WEBP uploads are supported."
                ),
            )

        size_bytes = _resolve_upload_size(file)
        max_bytes = (
            GLMOCRTool.MAX_PDF_BYTES if mime_type == PDF_MIME_TYPE else GLMOCRTool.MAX_IMAGE_BYTES
        )
        if size_bytes > max_bytes:
            max_size_mb = max_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Uploaded file '{filename}' is too large. "
                    f"Maximum size for {mime_type} is {max_size_mb:.1f} MB."
                ),
            )

    return files


async def validate_document_upload_files(files: UploadFiles) -> list[UploadFile]:
    """Validate multipart file uploads for ingestion endpoints."""
    return _validate_files(files)


@router.post(
    "/upload",
    response_model=list[DocumentIngestionResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest documents",
    description=(
        "Upload one or more documents, run OCR, and store the "
        "resulting vectors in the retrieval index."
    ),
    responses=DOCUMENT_ERROR_RESPONSES,
)
async def upload_documents(
    files: UploadFiles,
) -> list[DocumentIngestionResponse]:
    """Upload documents and ingest them into the retrieval stack."""
    try:
        files = await validate_document_upload_files(files)
        return await handle_knowledge_base_ingestion(files)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document upload endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload and ingest documents.",
        ) from exc


@router.post(
    "/knowledge-base/ingest",
    response_model=list[DocumentIngestionResponse],
    status_code=status.HTTP_200_OK,
    deprecated=True,
    summary="Legacy upload-and-ingest alias",
    description=(
        "Deprecated alias for `/upload`. Uploads documents, runs OCR, "
        "and stores the resulting vectors in the retrieval index."
    ),
    responses=DOCUMENT_ERROR_RESPONSES,
)
async def ingest_knowledge_base_documents(
    files: UploadFiles,
) -> list[DocumentIngestionResponse]:
    """Legacy batch ingestion endpoint for retrieval documents."""
    try:
        files = await validate_document_upload_files(files)
        return await handle_knowledge_base_ingestion(files)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Knowledge-base ingestion endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest knowledge-base documents.",
        ) from exc


@router.get(
    "/documents",
    response_model=list[DocumentListItem],
    status_code=status.HTTP_200_OK,
    summary="List ingested documents",
    description="Return a summary of all documents currently in the retrieval index.",
)
async def list_documents() -> list[DocumentListItem]:
    """List all ingested documents in the knowledge base."""
    try:
        tools = VectorStoreTools()
        raw = tools.list_documents()
    except Exception as exc:
        logger.warning("Failed to list documents from Qdrant: %s", exc)
        return []
    return [
        DocumentListItem(
            document_id=entry["document_id"],
            filename=entry["filename"],
            chunks=entry["chunks"],
            doc_title=entry.get("doc_title"),
        )
        for entry in raw
    ]


@router.delete(
    "/documents/by-filename/{filename:path}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete an ingested document by filename",
    description=(
        "Remove a previously ingested document from the retrieval index. "
        "Deletes all vector chunks and context records associated with the filename."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No document with the given filename exists in the index.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "The server failed to delete the document.",
        },
    },
)
async def delete_document_by_filename(filename: str) -> DocumentDeleteResponse:
    """Delete an ingested document and its chunks by filename."""
    try:
        tools = VectorStoreTools()
        matching_doc_ids = tools.find_document_ids_by_filename(filename)

        if not matching_doc_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No ingested document found with filename '{filename}'.",
            )

        for document_id in matching_doc_ids:
            tools.delete_document_chunks(document_id)
            logger.info(
                "Deleted document by filename",
                extra={"deleted_filename": filename, "document_id": document_id},
            )

        return DocumentDeleteResponse(
            success=True,
            document_id=sorted(matching_doc_ids)[0],
            filename=filename,
            deleted=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document deletion failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document '{filename}'.",
        ) from exc
