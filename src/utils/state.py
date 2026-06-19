"""
State schema definitions for the LangGraph agent.
Defines the complete state structure for student records and document processing.
"""

from collections.abc import MutableSequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, NamedTuple, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class OCRResult(NamedTuple):
    """Structured result from an OCR text extraction call."""

    text: str
    text_quality_score: float
    num_pages: Optional[int] = None
    layout_details: Optional[list[Any]] = None
    parsed_pages: Optional[int] = None
    failed_pages: Optional[list[int]] = None
    page_results: Optional[list[dict[str, Any]]] = None
    ocr_warnings: Optional[list[str]] = None


class DocumentType(StrEnum):
    """Supported document types for processing."""

    TRANSCRIPT = "transcript"
    ID_CARD = "id_card"
    APPLICATION = "application"
    RECOMMENDATION = "recommendation"
    CURRICULUM = "curriculum"
    SYLLABUS = "syllabus"
    CERTIFICATE = "certificate"
    INVOICE = "invoice"
    POLICY = "policy"
    OTHER = "other"


class ProcessingStatus(StrEnum):
    """Document processing status lifecycle."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentUpload(BaseModel):
    """Represents an uploaded document with metadata."""

    document_id: str = Field(description="Unique document identifier (UUID)")
    filename: str = Field(description="Original filename")
    file_path: str = Field(description="Local path to stored file")
    document_type: DocumentType = Field(default=DocumentType.OTHER)
    mime_type: str = Field(description="File MIME type")
    file_size: int = Field(description="File size in bytes")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Processing results
    extracted_text: Optional[str] = Field(default=None)
    text_quality_score: Optional[float] = Field(default=None)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    processing_error: Optional[str] = Field(default=None)
    quality_warning: Optional[str] = Field(default=None)
    ingestion_report: Optional[dict[str, Any]] = Field(default=None)

    # Layout parsing results
    num_pages: Optional[int] = Field(default=None)
    layout_details: Optional[list[Any]] = Field(default=None)
    parsed_pages: Optional[int] = Field(default=None)
    failed_pages: list[int] = Field(default_factory=list)
    ocr_warnings: list[str] = Field(default_factory=list)
    ocr_page_status: list[dict[str, Any]] = Field(default_factory=list)

    document_title: Optional[str] = Field(default=None)


class StudentRecord(BaseModel):
    """Structured student record extracted from documents or conversation."""

    student_id: Optional[str] = Field(default=None)
    full_name: Optional[str] = Field(default=None)
    date_of_birth: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)

    # Academic information
    program: Optional[str] = Field(default=None)
    major: Optional[str] = Field(default=None)
    gpa: Optional[float] = Field(default=None)
    enrollment_date: Optional[str] = Field(default=None)
    expected_graduation: Optional[str] = Field(default=None)

    # Document references
    document_ids: list[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: Optional[str] = Field(
        default=None, description="Source of record (e.g., 'upload', 'conversation')"
    )


class Citation(BaseModel):
    """Source citation attached to a document-backed response."""

    id: int = Field(description="One-based citation marker shown to users")
    document_id: Optional[str] = Field(default=None)
    filename: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    section: Optional[str] = Field(default=None)
    page: Optional[int] = Field(
        default=None,
        description="One-based page number for user display when known",
    )
    source_locations: list[dict[str, Any]] = Field(default_factory=list)
    score: Optional[float] = Field(default=None)
    chunk_id: Optional[str] = Field(default=None)
    snippet: Optional[str] = Field(default=None)


class AgentState(BaseModel):
    """
    Core state schema for the LangGraph agent.

    This state is passed between nodes and maintains the complete
    context of the conversation and processing pipeline.
    """

    # --- Conversation ---
    messages: Annotated[MutableSequence[BaseMessage], add_messages] = Field(
        default_factory=list, description="Conversation history with LangChain messages"
    )

    # --- Intent & Routing ---
    current_intent: Optional[str] = Field(
        default=None,
        description="Classified intent (e.g., 'upload_document', 'query_student', 'general_chat')",
    )
    requires_upload: bool = Field(default=False)
    requires_retrieval: bool = Field(default=False)

    # --- Document Processing ---
    pending_documents: list[DocumentUpload] = Field(default_factory=list)
    processed_documents: list[DocumentUpload] = Field(default_factory=list)
    current_document: Optional[DocumentUpload] = Field(default=None)

    # --- Student Records ---
    student_records: dict[str, StudentRecord] = Field(
        default_factory=dict, description="Map of student_id to StudentRecord"
    )
    current_student_id: Optional[str] = Field(default=None)

    # --- Retrieved Context ---
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_query: Optional[str] = Field(default=None)
    retrieval_warning: Optional[str] = Field(default=None)
    quality_warning: Optional[str] = Field(default=None)

    # --- Response Generation ---
    draft_response: Optional[str] = Field(default=None)
    response_confidence: float = Field(default=0.0)
    citations: list[Citation] = Field(default_factory=list)

    # --- Error Handling ---
    error: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)

    # --- Metadata ---
    session_id: str = Field(default="default")
    turn_count: int = Field(default=0)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(arbitrary_types_allowed=True)
