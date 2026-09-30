"""Student-record domain services."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from src.clients.llm import get_llm
from src.services.contracts import LLMProvider, StudentRecordRepository, StudentTextExtractor
from src.utils.nodes.prompts import STUDENT_RECORD_PROMPT
from src.utils.state import AgentState, StudentRecord

logger = logging.getLogger(__name__)


class StudentIdentifierParser:
    """Extract student identifiers from user queries."""

    _EMAIL_PATTERN = re.compile(r"[\w.-]+@[\w.-]+\.\w+")
    _STUDENT_ID_PATTERN = re.compile(r"STU_[A-Z0-9]{8}", re.IGNORECASE)

    def extract(self, text: str) -> str | None:
        """Return an email or student id when present in the text."""
        email_match = self._EMAIL_PATTERN.search(text)
        if email_match:
            return email_match.group()

        id_match = self._STUDENT_ID_PATTERN.search(text)
        if id_match:
            return id_match.group().upper()

        return None


class StudentRecordFactory:
    """Construct StudentRecord entities from extracted data."""

    def build(
        self,
        data: dict[str, Any],
        document_id: str | None,
        source: str,
    ) -> StudentRecord:
        """Build a record from raw extracted data."""
        record = StudentRecord(**data)
        if document_id:
            record.document_ids.append(document_id)
        record.source = source
        return record


class StudentDataExtractionService:
    """Extract structured student data using LLM first, then rules."""

    def __init__(
        self,
        fallback_extractor: StudentTextExtractor,
        llm_provider: LLMProvider = get_llm,
    ):
        self._fallback_extractor = fallback_extractor
        self._llm_provider = llm_provider
        self._llm = None

    def extract(self, text: str) -> tuple[dict[str, Any], str]:
        """Return structured data and the extraction source."""
        try:
            return self._extract_with_llm(text), "document_extraction"
        except Exception as exc:
            logger.error("Failed to create record from text: %s", exc)
            fallback_data = self._fallback_extractor.extract_from_text(text)
            if fallback_data:
                return fallback_data, "rule_extraction"
            raise ValueError(f"Could not extract student data: {exc}") from exc

    def _extract_with_llm(self, text: str) -> dict[str, Any]:
        """Use the configured LLM to extract structured student data."""
        if self._llm is None:
            self._llm = self._llm_provider()
        if self._llm is None:
            raise ValueError("OPENAI_API_KEY is required for LLM-based extraction")
        prompt = STUDENT_RECORD_PROMPT.format(text=text[:3000])
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)


class StudentRecordService:
    """Handle student-record creation and lookup decisions."""

    def __init__(
        self,
        repository: StudentRecordRepository,
        extractor: StudentDataExtractionService,
        identifier_parser: StudentIdentifierParser | None = None,
        record_factory: StudentRecordFactory | None = None,
    ):
        self._repository = repository
        self._extractor = extractor
        self._identifier_parser = identifier_parser or StudentIdentifierParser()
        self._record_factory = record_factory or StudentRecordFactory()

    def handle(self, state: AgentState, intent: str | None, message: str) -> dict[str, Any]:
        """Dispatch to the appropriate record operation for the current intent."""
        if intent == "manage_record":
            return self._handle_management(state, message)
        if intent == "query_student":
            return self._handle_query(state, message)
        return {}

    def _handle_management(self, state: AgentState, message: str) -> dict[str, Any]:
        """Create or update student records from document or message text."""
        if state.processed_documents:
            last_doc = state.processed_documents[-1]
            if last_doc.extracted_text:
                return self._create_from_text(
                    last_doc.extracted_text,
                    last_doc.document_id,
                )
        return self._create_from_text(message)

    def _handle_query(self, state: AgentState, message: str) -> dict[str, Any]:
        """Resolve direct record lookups before falling back to retrieval."""
        identifier = self._identifier_parser.extract(message)
        if identifier:
            record = self._repository.get_record(identifier)
            if not record:
                record = self._repository.find_by_email(identifier)
            if record:
                return {
                    "student_records": {
                        **state.student_records,
                        record.student_id: record,
                    },
                    "current_student_id": record.student_id,
                }
        return {"requires_retrieval": True, "retrieval_query": message}

    def _create_from_text(self, text: str, document_id: str | None = None) -> dict[str, Any]:
        """Create a student record from extracted text."""
        try:
            data, source = self._extractor.extract(text)
            record = self._record_factory.build(data, document_id, source=source)
            created = self._repository.create_record(record)
            return {
                "student_records": {created.student_id: created},
                "current_student_id": created.student_id,
            }
        except Exception as exc:
            logger.error("Student record creation failed: %s", exc)
            return {"error": str(exc)}
