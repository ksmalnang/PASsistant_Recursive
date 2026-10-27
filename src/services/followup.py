"""Follow-up classification using Jev."""

from __future__ import annotations

import logging

import httpx

from src.clients.decision import JevDecisionClient, get_decision_client

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "In an academic services chat, does the latest user message need the previous "
    "conversation to be understood?"
)
_CRITERIA_TRUE = (
    "The latest message refers to, continues, or omits information from the previous "
    "conversation. Read on its own, it is incomplete or ambiguous."
)
_CRITERIA_FALSE = (
    "The latest message is a self-contained question or request that can be understood "
    "without the previous conversation."
)

# Noul has no separate confidence field, so a probability near 0.5 is treated as
# uncertain and left to the rule-based fallback.
_TRUE_THRESHOLD = 0.6
_FALSE_THRESHOLD = 0.4


class JevFollowUpClassifier:
    """Ask Jev whether a message needs the previous chat context."""

    def __init__(self, decision_client: JevDecisionClient | None = None):
        self._decision_client = decision_client

    async def needs_previous_context(
        self,
        question: str,
        history: str,
        *,
        session_id: str | None = None,
    ) -> bool | None:
        """Return True/False, or None when Jev is unavailable or uncertain."""
        state = {
            "conversation": history or "(no previous conversation)",
            "latest_user_message": question,
        }
        try:
            client = self._decision_client or get_decision_client()
            answer = await client.ask_noul(
                state=state,
                instructions=_INSTRUCTIONS,
                criteria_true=_CRITERIA_TRUE,
                criteria_false=_CRITERIA_FALSE,
                session_id=session_id,
            )
        except (AssertionError, KeyError, RuntimeError, ValueError, httpx.HTTPError) as exc:
            logger.warning("Follow-up classification failed: %s", exc)
            return None

        if answer.noul >= _TRUE_THRESHOLD:
            return True
        if answer.noul <= _FALSE_THRESHOLD:
            return False
        return None
