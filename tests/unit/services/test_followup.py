"""Unit tests for the Jev follow-up classifier."""

from typing import Any

from src.clients.decision import NoulAnswer
from src.services.followup import JevFollowUpClassifier


class _NoulClient:
    def __init__(self, noul: float) -> None:
        self.noul = noul
        self.calls: list[dict[str, Any]] = []

    async def ask_noul(self, **kwargs: Any) -> NoulAnswer:
        self.calls.append(kwargs)
        return NoulAnswer(noul=self.noul)


class _FailingClient:
    async def ask_noul(self, **kwargs: Any) -> NoulAnswer:
        raise RuntimeError("decision endpoint unavailable")


async def test_follow_up_uses_noul_probability() -> None:
    """A high noul probability means the message needs the previous conversation."""
    client = _NoulClient(0.82)
    classifier = JevFollowUpClassifier(decision_client=client)

    verdict = await classifier.needs_previous_context(
        "kalau semester lima?",
        "User: berapa syarat mata kuliah pilihan?\nAssistant: Minimal 24 SKS.",
    )

    assert verdict is True
    assert client.calls[0]["state"] == {
        "conversation": (
            "User: berapa syarat mata kuliah pilihan?\nAssistant: Minimal 24 SKS."
        ),
        "latest_user_message": "kalau semester lima?",
    }


async def test_self_contained_question_returns_false() -> None:
    """A low noul probability means the message stands on its own."""
    classifier = JevFollowUpClassifier(decision_client=_NoulClient(0.12))

    verdict = await classifier.needs_previous_context("Apa syarat cuti akademik?", "")

    assert verdict is False


async def test_uncertain_probability_returns_none() -> None:
    """A probability near 0.5 is left to the rule-based fallback."""
    classifier = JevFollowUpClassifier(decision_client=_NoulClient(0.5))

    verdict = await classifier.needs_previous_context("kalau yang tadi gimana?", "")

    assert verdict is None


async def test_failure_returns_none() -> None:
    """A decision-endpoint failure is left to the rule-based fallback."""
    classifier = JevFollowUpClassifier(decision_client=_FailingClient())

    verdict = await classifier.needs_previous_context("kalau yang tadi gimana?", "")

    assert verdict is None


async def test_empty_history_uses_a_placeholder() -> None:
    """Jev receives an explicit placeholder when there is no prior conversation."""
    client = _NoulClient(0.9)
    classifier = JevFollowUpClassifier(decision_client=client)

    await classifier.needs_previous_context("Halo", "")

    assert client.calls[0]["state"]["conversation"] == "(no previous conversation)"
