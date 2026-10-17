"""Document upload, listing, deletion, and reindex endpoints."""

import logging
import mimetypes
import secrets
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from src.api.helpers import enforce_rate_limit
from src.api.models import (
    DocumentDeleteResponse,
    DocumentIngestionResponse,
    DocumentListItem,
    DocumentReindexResponse,
    ErrorResponse,
    ReindexDocumentResult,
    ReindexJobCreateRequest,
    ReindexJobCreateResponse,
    ReindexJobStatusResponse,
)
from src.api.services import handle_knowledge_base_ingestion
from src.config import get_settings
from src.services.document_deletion import get_document_deletion_service
from src.services.document_locks import DocumentLockedError
from src.services.reindex import FAILED, NoOCRCacheError, get_reindex_service
from src.services.reindex_jobs import JobAlreadyRunningError, get_job_runner
from src.utils.tools import VectorStoreTools
from src.utils.tools.ocr import PDF_MIME_TYPE, SUPPORTED_MIME_TYPES, GLMOCRTool

logger = logging.getLogger(__name__)
router = APIRouter()

PDF_MIME_ALIASES = {"application/x-pdf", "application/acrobat"}

UPLOAD_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    413: {"model": ErrorResponse, "description": "A file exceeds the size limit."},
    415: {"model": ErrorResponse, "description": "A file has an unsupported type."},
}

REINDEX_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: {"model": ErrorResponse, "description": "Missing or invalid admin key."},
    404: {"model": ErrorResponse, "description": "No OCR cache for the document."},
    409: {"model": ErrorResponse, "description": "The document is locked."},
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
            GLMOCRTool.MAX_PDF_BYTES if mime_type == PDF_MIME_TYPE else GLMOCRTool.MAX_IMAGE_BYTES
        )
        if _upload_size(file) > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Uploaded file '{filename}' is too large. "
                f"Maximum size for {mime_type} is {max_bytes / (1024 * 1024):.1f} MB.",
            )
    return files


def require_admin_key(
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> None:
    """Require X-Admin-Key to match ADMIN_API_KEY; 403 when no key is configured."""
    expected = get_settings().ADMIN_API_KEY
    if not expected:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Reindexing is disabled: ADMIN_API_KEY is not configured.",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Invalid or missing X-Admin-Key header.",
        )


# Reindex endpoints are admin-only and share the chat rate limiter.
admin_router = APIRouter(
    dependencies=[Depends(require_admin_key), Depends(enforce_rate_limit)],
    responses=REINDEX_ERROR_RESPONSES,
)


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
        409: {"model": ErrorResponse, "description": "The document is locked."},
    },
)
def delete_document_by_filename(
    filename: str,
    keep_files: Annotated[
        bool,
        Query(description="Keep the OCR cache and raw file; only remove vectors."),
    ] = False,
) -> DocumentDeleteResponse:
    """Delete chunks, and unless keep_files the OCR cache and raw file too."""
    try:
        document_ids = get_document_deletion_service().delete_documents(
            filename,
            keep_files=keep_files,
        )
    except DocumentLockedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not document_ids:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No ingested document found with filename '{filename}'.",
        )
    return DocumentDeleteResponse(document_id=min(document_ids), filename=filename)


@admin_router.post(
    "/documents/{document_id}/reindex",
    summary="Reindex one document from its OCR cache",
)
async def reindex_document(
    document_id: str,
    rebuild_text: Annotated[bool, Query()] = False,
) -> DocumentReindexResponse:
    """Rebuild chunks for one document without re-running OCR."""
    try:
        result = await get_reindex_service().reindex_document(
            document_id,
            rebuild_text=rebuild_text,
        )
    except NoOCRCacheError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except DocumentLockedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if result.status == FAILED:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            result.error or "Reindex failed.",
        )
    return DocumentReindexResponse(
        document_id=result.document_id,
        filename=result.filename,
        status=result.status,
        chunks=result.chunks,
    )


@admin_router.post(
    "/documents/reindex/by-filename/{filename:path}",
    summary="Reindex every cached document with a filename",
)
async def reindex_documents_by_filename(
    filename: str,
    rebuild_text: Annotated[bool, Query()] = False,
) -> list[ReindexDocumentResult]:
    """Reindex all OCR cache entries that carry the given filename."""
    try:
        results = await get_reindex_service().reindex_by_filename(
            filename,
            rebuild_text=rebuild_text,
        )
    except NoOCRCacheError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except DocumentLockedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return [ReindexDocumentResult(**result.to_dict()) for result in results]


@admin_router.post(
    "/documents/reindex-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a bulk reindex job",
)
async def create_reindex_job(
    payload: ReindexJobCreateRequest,
    response: Response,
) -> ReindexJobCreateResponse:
    """Queue an asynchronous bulk reindex; only one job may be active."""
    service = get_reindex_service()
    document_ids = payload.document_ids
    if document_ids is None:
        document_ids = service.all_cached_document_ids()
    try:
        job = get_job_runner().create_job(document_ids, rebuild_text=payload.rebuild_text)
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"reindex job {exc.active_job_id} is already active",
        ) from exc

    response.headers["Location"] = f"/documents/reindex-jobs/{job['job_id']}"
    return ReindexJobCreateResponse(
        job_id=str(job["job_id"]),
        status=str(job["status"]),
        total=int(job["total"]),
    )


@admin_router.get(
    "/documents/reindex-jobs/{job_id}",
    summary="Get bulk reindex job status",
)
async def get_reindex_job(job_id: str) -> ReindexJobStatusResponse:
    """Return the current state of a bulk reindex job."""
    job = get_job_runner().get_job(job_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown reindex job '{job_id}'.",
        )
    return ReindexJobStatusResponse.model_validate(job)


@admin_router.delete(
    "/documents/reindex-jobs/{job_id}",
    summary="Cancel a bulk reindex job",
)
async def cancel_reindex_job(job_id: str) -> ReindexJobStatusResponse:
    """Request cancellation; the job stops after the current document."""
    runner = get_job_runner()
    job = runner.get_job(job_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown reindex job '{job_id}'.",
        )
    if not runner.request_cancel(job_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Reindex job '{job_id}' is already {job['status']}.",
        )
    updated = runner.get_job(job_id)
    return ReindexJobStatusResponse.model_validate(updated)


router.include_router(admin_router)
