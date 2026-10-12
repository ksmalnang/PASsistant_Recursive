"""Unit tests for intent classification and routing (UT-006)."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.services.intent import INTENT_LABELS, IntentClassifier, valid_intents
from src.utils.nodes.router import RouterNode
from src.utils.state import AgentState
from tests.doubles import FakeDecisionClient, FakeLLMProvider


async def test_unit_006_intent_classifier_maps_known_messages() -> None:
    """Keyword-routed messages map to workflow intents without a Jev or LLM call."""
    provider = FakeLLMProvider(response="general_chat")
    decision = FakeDecisionClient(choice="general_chat", confidence=0.9)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    academic = await classifier.classify("graduation requirements")
    assert academic["current_intent"] == "query_document"
    assert academic["requires_retrieval"] is True
    assert academic["retrieval_query"] == "graduation requirements"

    student = await classifier.classify("GPA saya berapa?")
    assert student["current_intent"] == "query_student"
    assert student["requires_retrieval"] is True

    # Keyword matches must not pay for a Jev or LLM round trip.
    assert decision.calls == 0
    assert provider.calls == 0

    general = await classifier.classify("halo")
    assert general["current_intent"] == "general_chat"
    assert general["intent_confidence"] == 0.9
    assert decision.calls == 1
    assert provider.calls == 0


async def test_unit_006_intent_classifier_falls_back_to_llm_on_low_confidence() -> None:
    """Low-confidence Jev classification falls back to LLM provider."""
    provider = FakeLLMProvider(response="general_chat")
    decision = FakeDecisionClient(choice="general_chat", confidence=0.3)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    general = await classifier.classify("halo")
    assert general["current_intent"] == "general_chat"
    assert decision.calls == 1
    assert provider.calls == 1
    assert general.get("intent_fallback") == "llm_fallback"


async def test_unit_006_intent_classifier_llm_only_when_jev_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Jev is disabled, intent classification uses LLM directly."""
    monkeypatch.setattr(
        "src.services.intent.get_settings",
        lambda: SimpleNamespace(JEV_INTENT_ENABLED=False),
    )
    provider = FakeLLMProvider(response="general_chat")
    classifier = IntentClassifier(llm_provider=provider)

    general = await classifier.classify("halo")
    assert general["current_intent"] == "general_chat"
    assert provider.calls == 1


class _RecordingClassifier:
    """Intent classifier double that records the context it received."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def classify(
        self,
        user_text: str,
        session_id: str | None = None,
        *,
        history: str | None = None,
    ) -> dict:
        self.calls.append({"text": user_text, "history": history})
        return {"current_intent": "query_document"}


async def test_unit_006_follow_up_reaches_jev_with_conversation_context() -> None:
    """A referential follow-up is classified with the prior turn as evidence."""
    provider = FakeLLMProvider(response="query_document")
    decision = FakeDecisionClient(choice="query_document", confidence=0.9)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    await classifier.classify(
        "kalau yang tadi bagaimana?",
        history="User: berapa syarat mata kuliah pilihan?",
    )

    assert decision.calls == 1
    assert "berapa syarat mata kuliah pilihan?" in decision.states[0]
    assert "kalau yang tadi bagaimana?" in decision.states[0]


async def test_unit_006_llm_fallback_receives_conversation_context() -> None:
    """Low-confidence Jev hands the same conversation to the LLM fallback."""
    provider = FakeLLMProvider(response="query_document")
    decision = FakeDecisionClient(choice="general_chat", confidence=0.3)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    result = await classifier.classify(
        "kalau yang tadi bagaimana?",
        history="User: berapa syarat mata kuliah pilihan?",
    )

    assert result["current_intent"] == "query_document"
    assert result["intent_fallback"] == "llm_fallback"
    assert "berapa syarat mata kuliah pilihan?" in provider.llm.prompts[0][0].content


async def test_unit_006_router_forwards_prior_turns_as_history() -> None:
    """RouterNode hands the classifier the turns before the current message."""
    classifier = _RecordingClassifier()
    node = RouterNode(classifier=classifier)
    state = AgentState(
        session_id="router-history",
        messages=[
            HumanMessage(content="berapa syarat mata kuliah pilihan?"),
            AIMessage(content="Minimal 24 SKS."),
            HumanMessage(content="kalau yang tadi bagaimana?"),
        ],
    )

    update = await node.run(state)

    assert update["current_intent"] == "query_document"
    assert classifier.calls[0]["text"] == "kalau yang tadi bagaimana?"
    assert "berapa syarat mata kuliah pilihan?" in classifier.calls[0]["history"]
    assert "kalau yang tadi bagaimana?" not in classifier.calls[0]["history"]

    assert "kalau yang tadi bagaimana?" not in classifier.calls[0]["history"]


class _LoopBoundDecisionClient:
    """Decision client double that records the event loop of every call."""

    def __init__(self) -> None:
        self.loops: list[Any] = []

    async def ask_choice(self, *args: Any, **kwargs: Any) -> Any:
        from src.clients.decision import ChoiceAnswer

        self.loops.append(asyncio.get_running_loop())
        return ChoiceAnswer(
            choice="general_chat",
            confidence=0.9,
            probabilities={"general_chat": 0.9},
        )


async def test_unit_006_repeated_classification_stays_on_one_loop() -> None:
    """The Jev client is loop-bound: every call must run on the caller's loop."""
    decision = _LoopBoundDecisionClient()
    classifier = IntentClassifier(llm_provider=FakeLLMProvider(), decision_client=decision)

    for _ in range(3):
        await classifier.classify("halo")

    assert len(decision.loops) == 3
    assert all(loop is decision.loops[0] for loop in decision.loops)


def test_unit_006_manage_record_is_removed_from_the_intent_set() -> None:
    """The retired record-creation intent can no longer be classified."""
    assert "manage_record" not in valid_intents
    assert "manage_record" not in INTENT_LABELS


async def test_unit_006_retired_label_falls_back_to_the_llm() -> None:
    """A Jev answer outside the intent set is re-decided by the LLM fallback."""
    provider = FakeLLMProvider(response="general_chat")
    decision = FakeDecisionClient(choice="manage_record", confidence=0.95)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    result = await classifier.classify("tolong daftarkan data baru untuk saya")

    assert result["current_intent"] == "general_chat"
    assert result["intent_fallback"] == "llm_fallback"
    assert provider.calls == 1


async def test_unit_006_broad_record_phrases_reach_jev() -> None:
    """Ambiguous record phrasing no longer fast-paths to query_student."""
    provider = FakeLLMProvider(response="query_document")
    decision = FakeDecisionClient(choice="query_document", confidence=0.9)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    result = await classifier.classify("di mana saya bisa melihat data mahasiswa?")

    assert decision.calls == 1
    assert result["current_intent"] == "query_document"


async def test_unit_006_sks_policy_phrasing_routes_to_query_document() -> None:
    """Graduation SKS questions are policy, not personal records, and skip Jev."""
    provider = FakeLLMProvider(response="general_chat")
    decision = FakeDecisionClient(choice="general_chat", confidence=0.9)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    result = await classifier.classify(
        "total SKS yang harus ditempuh buat lulus S1 Teknik Informatika berapa?"
    )

    assert result["current_intent"] == "query_document"
    assert result["requires_retrieval"] is True
    assert decision.calls == 0
