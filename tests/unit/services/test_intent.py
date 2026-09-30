"""Unit tests for intent classification and routing (UT-006)."""

from src.services.intent import IntentClassifier
from tests.doubles import FakeLLMProvider


def test_unit_006_intent_classifier_maps_known_messages() -> None:
    """Keyword-routed messages map to workflow intents without an LLM call."""
    provider = FakeLLMProvider(response="general_chat")
    classifier = IntentClassifier(llm_provider=provider)

    upload = classifier.classify("upload dokumen ini")
    assert upload["current_intent"] == "upload_document"
    assert upload["requires_upload"] is True

    academic = classifier.classify("graduation requirements")
    assert academic["current_intent"] == "query_document"
    assert academic["requires_retrieval"] is True
    assert academic["retrieval_query"] == "graduation requirements"

    student = classifier.classify("GPA saya berapa?")
    assert student["current_intent"] == "query_student"
    assert student["requires_retrieval"] is True

    # Keyword matches must not pay for an LLM round trip.
    assert provider.calls == 0

    general = classifier.classify("halo")
    assert general["current_intent"] == "general_chat"
    assert provider.calls == 1
