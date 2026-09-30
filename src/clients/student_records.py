"""Student record storage, mock today and academic API later."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # runtime imports would cycle through src.utils.tools -> src.clients
    from src.services.contracts import StudentRecordRepository
    from src.utils.state import StudentRecord

logger = logging.getLogger(__name__)
_student_repository: StudentRecordRepository | None = None


class InMemoryStudentRecordRepository:
    """In-memory StudentRecordRepository used until the academic API is integrated.

    Mirrors the intended API behavior: identifier-keyed records with an email index.
    """

    def __init__(self) -> None:
        self._records: dict[str, StudentRecord] = {}
        self._email_index: dict[str, str] = {}
        self._lock = threading.Lock()

    def create_record(self, record: StudentRecord) -> StudentRecord:
        """Create and return a student record."""
        with self._lock:
            if not record.student_id:
                record.student_id = f"STU_{uuid.uuid4().hex[:8].upper()}"

            timestamp = datetime.now(UTC)
            record.created_at = timestamp
            record.updated_at = timestamp

            self._records[record.student_id] = record
            if record.email:
                self._email_index[record.email] = record.student_id

            logger.info("Created student record: %s", record.student_id)
            return record

    def get_record(self, student_id: str) -> StudentRecord | None:
        """Lookup a record by student id."""
        with self._lock:
            return self._records.get(student_id)

    def find_by_email(self, email: str) -> StudentRecord | None:
        """Lookup a record by email."""
        with self._lock:
            student_id = self._email_index.get(email)
            return self._records.get(student_id) if student_id else None


def get_student_repository() -> StudentRecordRepository:
    """Return the process-wide student repository.

    Today this is the in-memory mock. When the academic records API is
    integrated, this function reads STUDENT_API_URL and returns the HTTP
    implementation instead; consumers never change.
    """
    global _student_repository
    if _student_repository is None:
        _student_repository = InMemoryStudentRecordRepository()
    return _student_repository
