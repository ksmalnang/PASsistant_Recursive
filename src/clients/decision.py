"""Typesafe client for OpenRouter Decision model (Jev / TypeSafe).

Sends structured decision requests to the OpenRouter Decisions API
and returns typed answers (Choice, Noul, Score) with probabilities.

Usage:
    client = get_decision_client()

    # Single question helpers
    result = client.ask_noul(
        state="The sky is green.",
        instructions="Is this statement factually correct?",
        criteria_true="The statement matches observable reality.",
        criteria_false="The statement contradicts observable reality.",
    )
    if result.noul > 0.95:
        ...

    # Multi-question decisions
    answers = client.ask({
        "is_bug": NoulQuestion(
            instructions="Is the customer reporting a bug?",
            criteria_true="...",
            criteria_false="...",
        ),
        "team": ChoiceQuestion(
            instructions="Which team?",
            criteria={"frontend": "...", "backend": "..."},
        ),
    }, state={...})
    print(answers["is_bug"].noul)
"""

import json
import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel, field_validator

from src.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class NoulQuestion(BaseModel):
    """A yes/no (nouL) question for Jev.

    Attributes:
        instructions: The question being asked.
        criteria_true: Description of what "true" (yes) means.
        criteria_false: Description of what "false" (no) means.
    """

    type: Literal["noul"] = "noul"
    instructions: str
    criteria_true: str
    criteria_false: str


class ChoiceQuestion(BaseModel):
    """A choice question for Jev.

    Attributes:
        instructions: The question being asked.
        criteria: Mapping from option label to description of that option.
    """

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class ScoreQuestion(BaseModel):
    """An ordered-scale (score) question for Jev.

    Attributes:
        instructions: The question being asked.
        criteria: Ordered list of level descriptions, from lowest to highest.
    """

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class NoulAnswer(BaseModel):
    """Answer to a noul question."""

    type: Literal["noul"] = "noul"
    noul: float
    """Probability (0-1) that the answer is yes / true."""


class ChoiceAnswer(BaseModel):
    """Answer to a choice question."""

    type: Literal["choice"] = "choice"
    choice: str
    """The selected option label."""
    confidence: float
    """How concentrated the probability distribution is (0-1)."""
    probabilities: dict[str, float]
    """Probability per option (maps option label -> probability)."""


class ScoreAnswer(BaseModel):
    """Answer to a score question."""

    type: Literal["score"] = "score"
    score: float
    """Probability-weighted position on the scale (0-indexed)."""
    confidence: float
    """How concentrated the probability distribution is (0-1)."""
    probabilities: dict[str, float]
    """Probability per level (maps string index -> probability)."""
    legend: dict[str, str]
    """Maps index string -> original criterion description."""


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion
Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(BaseModel):
    """Token and cost info returned by the Decisions API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class DecisionResponse(BaseModel):
    """Top-level response from the Decisions API."""

    id: str | None = None
    model: str | None = None
    provider: str | None = None
    answers: dict[str, Any]
    usage: Usage = Usage()

    @field_validator("answers", mode="before")
    @classmethod
    def _parse_answers(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Deserialize raw answer dicts into typed Answer objects."""
        parsed: dict[str, Any] = {}
        for key, raw in v.items():
            if not isinstance(raw, dict):
                parsed[key] = raw
                continue
            atype = raw.get("type")
            if atype == "noul":
                parsed[key] = NoulAnswer(**raw)
            elif atype == "choice":
                parsed[key] = ChoiceAnswer(**raw)
            elif atype == "score":
                parsed[key] = ScoreAnswer(**raw)
            else:
                parsed[key] = raw
        return parsed


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class JevDecisionClient:
    """Typesafe HTTP client for the OpenRouter Decisions API (Jev).

    Provides both low-level ``ask()`` for multi-question requests
    and convenience helpers ``ask_noul()``, ``ask_choice()``, ``ask_score()``
    for single-question calls.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/alpha/decisions",
        model: str = "typesafe/jev-1.13",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(
        self,
        questions: dict[str, Question],
        state: str | dict[str, Any] | list[Any],
        *,
        model: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Answer]:
        """Ask one or more questions in a single Decisions API call.

        Args:
            questions: Dict of question name -> question definition.
            state: The context to evaluate (string, dict, or list).
            model: Override the default model ID.
            session_id: Optional grouping identifier.

        Returns:
            Dict mapping question names to typed Answer objects.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        payload = self._build_payload(questions, state, model=model)
        response = await self._post(payload, session_id=session_id)
        return dict(response.answers)  # type: ignore[return-value]

    async def ask_noul(
        self,
        state: str | dict[str, Any] | list[Any],
        instructions: str,
        criteria_true: str,
        criteria_false: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
    ) -> NoulAnswer:
        """Ask a single yes/no (noul) question."""
        answers = await self.ask(
            {"q": NoulQuestion(
                instructions=instructions,
                criteria_true=criteria_true,
                criteria_false=criteria_false,
            )},
            state=state,
            model=model,
            session_id=session_id,
        )
        result = answers["q"]
        assert isinstance(result, NoulAnswer), f"Expected NoulAnswer, got {type(result).__name__}"
        return result

    async def ask_choice(
        self,
        state: str | dict[str, Any] | list[Any],
        instructions: str,
        criteria: dict[str, str],
        *,
        model: str | None = None,
        session_id: str | None = None,
    ) -> ChoiceAnswer:
        """Ask a single choice question."""
        answers = await self.ask(
            {"q": ChoiceQuestion(
                instructions=instructions,
                criteria=criteria,
            )},
            state=state,
            model=model,
            session_id=session_id,
        )
        result = answers["q"]
        assert isinstance(result, ChoiceAnswer), \
            f"Expected ChoiceAnswer, got {type(result).__name__}"
        return result

    async def ask_score(
        self,
        state: str | dict[str, Any] | list[Any],
        instructions: str,
        criteria: list[str],
        *,
        model: str | None = None,
        session_id: str | None = None,
    ) -> ScoreAnswer:
        """Ask a single ordered-scale (score) question."""
        answers = await self.ask(
            {"q": ScoreQuestion(
                instructions=instructions,
                criteria=criteria,
            )},
            state=state,
            model=model,
            session_id=session_id,
        )
        result = answers["q"]
        assert isinstance(result, ScoreAnswer), \
            f"Expected ScoreAnswer, got {type(result).__name__}"
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        questions: dict[str, Question],
        state: str | dict[str, Any] | list[Any],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON request body from typed question objects."""
        raw_questions: dict[str, dict[str, Any]] = {}
        for name, q in questions.items():
            if isinstance(q, NoulQuestion):
                raw_questions[name] = {
                    "type": "noul",
                    "instructions": q.instructions,
                    "criteria": {"true": q.criteria_true, "false": q.criteria_false},
                }
            elif isinstance(q, ChoiceQuestion):
                raw_questions[name] = {
                    "type": "choice",
                    "instructions": q.instructions,
                    "criteria": q.criteria,
                }
            elif isinstance(q, ScoreQuestion):
                raw_questions[name] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": q.criteria,
                }
            else:
                raise TypeError(f"Unknown question type: {type(q).__name__}")

        return {
            "model": model or self._model,
            "state": state,
            "questions": raw_questions,
        }

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> DecisionResponse:
        """POST the payload to the Decisions API and return a typed response."""
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if session_id:
            headers["X-Session-Id"] = session_id

        logger.debug(
            "Decision request: model=%s state_len=%d questions=%d",
            payload.get("model"),
            _estimate_size(payload.get("state", "")),
            len(payload.get("questions", {})),
        )

        response = await self._client.post(
            self._base_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        try:
            raw = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Decisions API returned non-JSON response (status {response.status_code}): "
                f"{response.text[:500]}"
            ) from exc

        logger.debug(
            "Decision response: cost=%s input_tokens=%d output_tokens=%d",
            raw.get("usage", {}).get("cost"),
            raw.get("usage", {}).get("input_tokens", 0),
            raw.get("usage", {}).get("output_tokens", 0),
        )

        return DecisionResponse(**raw)

    async def aclose(self) -> None:
        """Close the underlying HTTP client session."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

_decision_client: JevDecisionClient | None = None


def _resolve_api_key(settings: Any) -> str:
    """Resolve the Decisions API key, falling back to OPENAI_API_KEY."""
    key = (
        getattr(settings, "DECISION_API_KEY", None)
        or getattr(settings, "OPENROUTER_API_KEY", None)
        or getattr(settings, "OPENAI_API_KEY", None)
    )
    if not key:
        raise ValueError(
            "No API key found for Decisions API. Set DECISION_API_KEY, "
            "OPENROUTER_API_KEY, or OPENAI_API_KEY in your .env."
        )
    return key


def get_decision_client() -> JevDecisionClient:
    """Return the process-wide JevDecisionClient singleton.

    The client is created once from settings and reused for the
    lifetime of the process. Raises ValueError if no API key
    is configured.
    """
    global _decision_client
    if _decision_client is None:
        settings = get_settings()
        _decision_client = JevDecisionClient(
            api_key=_resolve_api_key(settings),
            base_url=getattr(settings, "DECISION_BASE_URL", "https://openrouter.ai/api/alpha/decisions"),
            model=getattr(settings, "DECISION_MODEL", "typesafe/jev-1.13"),
            timeout_seconds=getattr(settings, "DECISION_TIMEOUT_SECONDS", 30.0),
        )
    return _decision_client


def close_decision_client() -> None:
    """Close and reset the process-wide decision client.

    If a running event loop is available (e.g., FastAPI async shutdown),
    the underlying httpx AsyncClient is awaited explicitly. Otherwise the
    reference is dropped and GC will release connections.

    Safe to call more than once.
    """
    global _decision_client
    client = _decision_client
    if client is None:
        return
    _decision_client = None

    logger.debug("Closing JevDecisionClient...")
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            logger.debug("Event loop is closed; dropping reference for GC cleanup")
        else:
            # Explicitly await the async close via the running loop
            future = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
            future.result(timeout=10.0)
            logger.debug("JevDecisionClient closed via awaited aclose()")
            return
    except RuntimeError:
        logger.debug("No running event loop; dropping reference for GC cleanup")


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _estimate_size(state: Any) -> int:
    """Rough estimate of state content size (characters or item count)."""
    if isinstance(state, str):
        return len(state)
    if isinstance(state, dict):
        return sum(len(str(v)) for v in state.values())
    if isinstance(state, list):
        return len(state)
    return 0
