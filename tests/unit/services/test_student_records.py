"""Unit tests for student identifier parsing (UT-005)."""

from src.services.student_records import StudentIdentifierParser


def test_unit_005_student_identifier_parser_extracts_email_and_normalizes_student_id() -> None:
    """Emails are extracted verbatim; student ids are normalized to upper case."""
    parser = StudentIdentifierParser()

    assert (
        parser.extract("Contact me at budi.santoso@example.com for the transcript.")
        == "budi.santoso@example.com"
    )
    assert parser.extract("My student code is STU_abc12345") == "STU_ABC12345"
    assert parser.extract("There is no identifier in this message.") is None
