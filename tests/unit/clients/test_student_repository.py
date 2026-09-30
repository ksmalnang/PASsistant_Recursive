"""Unit tests for the in-memory student record repository."""

from langchain_core.messages import HumanMessage

from src.clients.student_records import (
    InMemoryStudentRecordRepository,
    get_student_repository,
)
from src.utils.nodes.student_record import StudentRecordNode
from src.utils.state import AgentState
from tests.factories import make_student_record


def test_unit_01_create_and_get_record_by_id() -> None:
    """Creation stamps timestamps and stores the record under its identifier."""
    repository = InMemoryStudentRecordRepository()
    record = make_student_record(student_id="STU_A1B2C3D4")

    created = repository.create_record(record)

    assert created.created_at == created.updated_at
    assert repository.get_record("STU_A1B2C3D4") is record


def test_unit_02_find_by_email_returns_matching_record() -> None:
    """The email index resolves to the record that was created with it."""
    repository = InMemoryStudentRecordRepository()
    record = repository.create_record(make_student_record(email="rina.kusuma@example.com"))

    assert repository.find_by_email("rina.kusuma@example.com") is record


def test_unit_03_unknown_identifier_returns_none() -> None:
    """Unknown ids and emails resolve to nothing."""
    repository = InMemoryStudentRecordRepository()

    assert repository.get_record("STU_MISSING0") is None
    assert repository.find_by_email("nobody@example.com") is None


def test_unit_04_records_created_through_node_repository_are_shared_across_sessions() -> None:
    """A record created by one node is visible to a node serving another session."""
    writer = StudentRecordNode(llm_provider=lambda: None)
    reader = StudentRecordNode(llm_provider=lambda: None)

    created = writer.run(
        AgentState(
            session_id="session-a",
            current_intent="manage_record",
            messages=[HumanMessage(content="Full name: Rina Kusuma")],
        )
    )
    student_id = created["current_student_id"]

    found = reader.run(
        AgentState(
            session_id="session-b",
            current_intent="query_student",
            messages=[
                HumanMessage(content=f"Show the student record for student id {student_id}.")
            ],
        )
    )

    assert found["student_records"][student_id] is created["student_records"][student_id]


def test_unit_05_provider_returns_process_wide_singleton() -> None:
    """The provider resolves the same repository across calls."""
    assert get_student_repository() is get_student_repository()
