"""Pydantic models for the API layer."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field

from src.utils.state import Citation


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(description="User message text", min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None,
        description="Thread identifier for continuity",
        validation_alias=AliasChoices("thread_id", "session_id"),
    )


class ChatResponse(BaseModel):
    """Chat response payload."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message identifier",
    )
    response: str = Field(description="Agent response text")
    thread_id: str = Field(description="Thread identifier")
    intent: str | None = Field(default=None, description="Classified intent")
    documents_processed: int = Field(
        default=0,
        description="Number of documents processed",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Document sources used for the answer",
    )


ChatStreamEventType = Literal[
    "run.started",
    "run.status",
    "message.delta",
    "run.completed",
    "run.failed",
]


class ChatStreamEvent(BaseModel):
    """Transport-neutral chat streaming event."""

    event_id: str = Field(description="Opaque event identifier")
    event_type: ChatStreamEventType = Field(description="Event type")
    thread_id: str = Field(description="Thread identifier")
    run_id: str = Field(description="Run identifier for this chat turn")
    timestamp: datetime = Field(description="Event creation timestamp")
    sequence: int = Field(description="Monotonic event sequence within a run")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    detail: str = Field(description="Human-readable error detail")


class DependencyHealthResponse(BaseModel):
    """Health status for an external dependency."""

    status: str = Field(description="Dependency state such as healthy, unhealthy, or disabled")
    detail: str | None = Field(
        default=None,
        description="Additional dependency status detail",
    )


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion."""

    success: bool = Field(default=True, description="Whether deletion completed")
    document_id: str = Field(description="Deleted document identifier")
    filename: str = Field(description="Deleted document filename")
    chunks_deleted: bool = Field(
        default=True,
        description="Whether vector chunks were removed from the index",
    )


class DocumentIngestionResponse(BaseModel):
    """Knowledge-base ingestion response."""

    success: bool = Field(default=True, description="Whether ingestion completed")
    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    document_type: str = Field(description="Detected document type")
    status: str = Field(description="Processing status")
    chunks_stored: int = Field(default=0, description="Number of vector chunks stored")
    parent_chunks_stored: int = Field(
        default=0,
        description="Number of parent chunks stored",
    )
    document_title: str | None = Field(
        default=None,
        description="Extracted or inferred document title",
    )
    parsed_pages: int | None = Field(
        default=None,
        description="Number of OCR pages parsed successfully",
    )
    failed_pages: list[int] = Field(
        default_factory=list,
        description="Zero-based OCR page indexes that failed",
    )
    ocr_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal OCR warnings",
    )
    quality_warning: str | None = Field(
        default=None,
        description="Ingestion quality warning if chunk coverage looked suspicious",
    )
    error: str | None = Field(default=None, description="Processing error, if any")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current environment")
    redis: DependencyHealthResponse = Field(description="Redis connectivity status")
    qdrant: DependencyHealthResponse = Field(description="Qdrant connectivity status")
