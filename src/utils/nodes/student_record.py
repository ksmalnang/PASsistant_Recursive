"""Student record node."""

import logging

from src.clients.llm import get_llm
from src.clients.student_records import get_student_repository
from src.services.contracts import LLMProvider, StudentRecordRepository, StudentTextExtractor
from src.services.student_records import (
    StudentDataExtractionService,
    StudentIdentifierParser,
    StudentRecordFactory,
    StudentRecordService,
)
from src.utils.state import AgentState
from src.utils.tools import RuleBasedStudentExtractor

logger = logging.getLogger(__name__)


def _message_content_to_text(content: object) -> str:
    """Normalize LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    return str(content)


class StudentRecordNode:
    """
    Handles student-record creation and direct lookup workflows.

    Handles:
    - Record creation from processed document text or message text
    - Record lookup by student id or email
    - Delegation back to retrieval when no direct record match is found
    """

    def __init__(
        self,
        repository: StudentRecordRepository | None = None,
        fallback_extractor: StudentTextExtractor | None = None,
        service: StudentRecordService | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        resolved_repository = repository or get_student_repository()
        resolved_extractor = fallback_extractor or RuleBasedStudentExtractor()
        self._service = service or StudentRecordService(
            repository=resolved_repository,
            extractor=StudentDataExtractionService(
                fallback_extractor=resolved_extractor,
                llm_provider=llm_provider or get_llm,
            ),
            identifier_parser=StudentIdentifierParser(),
            record_factory=StudentRecordFactory(),
        )

    def run(self, state: AgentState) -> dict:
        """
        Execute student record operations based on intent.

        Args:
            state: Current agent state

        Returns:
            State updates with record operation results
        """
        intent = state.current_intent
        last_message = (
            _message_content_to_text(state.messages[-1].content) if state.messages else ""
        )

        try:
            return self._service.handle(state, intent, last_message)
        except Exception as exc:
            logger.error("Student record operation failed: %s", exc)
            return {"error": str(exc)}
