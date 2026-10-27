"""Unit tests for the response system prompt."""

from src.utils.nodes.prompts import RESPONSE_SYSTEM_PROMPT


def test_response_prompt_renders_and_documents_course_codes() -> None:
    """The prompt stays a valid context template and carries the code reference."""
    rendered = RESPONSE_SYSTEM_PROMPT.format(context="CTX")

    assert "CTX" in rendered
    assert "IF21W0103" in rendered
    assert "elective" in rendered.lower()
    assert "Cross-turn consistency" in rendered
    assert "Previous-turn evidence" in rendered
    assert "citation ID" in rendered
