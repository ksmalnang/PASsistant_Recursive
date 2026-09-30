"""Rule-based student record text extraction."""

from contextlib import suppress
from typing import Any


class RuleBasedStudentExtractor:
    """Extract structured student fields from raw OCR text without an LLM."""

    def extract_from_text(self, text: str) -> dict[str, Any]:
        """
        Extract structured student information from raw text.

        This is a simple rule-based extractor. In production,
        use an LLM for more sophisticated extraction.

        Args:
            text: Raw text from OCR

        Returns:
            Dictionary of extracted fields
        """
        extracted: dict[str, Any] = {}
        for raw_line in text.lower().splitlines():
            line = raw_line.strip()
            if "name:" in line or "full name:" in line:
                extracted["full_name"] = self._extract_text_value(line).title()
            elif "student id:" in line or "id:" in line:
                extracted["student_id"] = self._extract_text_value(line).upper()
            elif "email:" in line or "e-mail:" in line:
                extracted["email"] = self._extract_text_value(line)
            elif "gpa:" in line or "grade point average:" in line:
                with suppress(ValueError):
                    extracted["gpa"] = float(self._extract_text_value(line))
            elif "program:" in line or "degree:" in line:
                extracted["program"] = self._extract_text_value(line).title()
            elif "major:" in line:
                extracted["major"] = self._extract_text_value(line).title()

        return extracted

    def _extract_text_value(self, line: str) -> str:
        """Extract the value segment from a key-value line."""
        return line.split(":", 1)[-1].strip()
