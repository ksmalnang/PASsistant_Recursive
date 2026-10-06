"""HTTP-layer tests for ``POST /chat`` (IT-001..IT-004, IT-010).

The route, input guard, session manager, and response mapping run as production
code; only agent creation is replaced so no LLM call happens.
"""

from __future__ import annotations

import httpx
import pytest

from tests.doubles import ChatBackend, FakeLLM, LlmBackedChatAgent
from tests.factories import make_agent_state, make_chat_payload, make_citation, make_upload

EXPECTED_ANSWER = "The graduation requirement is 144 credits."


@pytest.fixture
def scripted_backend(chat_backend: ChatBackend) -> ChatBackend:
    """Script the final workflow state replayed by every new agent double."""
    chat_backend.final_state_factory = lambda thread_id: make_agent_state(
        response=EXPECTED_ANSWER,
        session_id=thread_id,
        citations=[make_citation(1, filename="academic-policy.pdf")],
    )
    return chat_backend


async def test_api_01_chat_returns_structured_response(
    api_client: httpx.AsyncClient,
    scripted_backend: ChatBackend,
    valid_academic_message: str,
) -> None:
    """A valid academic question returns the answer, thread id, and citations."""
    response = await api_client.post("/chat", data=make_chat_payload(valid_academic_message))

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == EXPECTED_ANSWER
    assert body["thread_id"] == scripted_backend.last_agent.session_id
    assert body["intent"] == "query_document"
    assert isinstance(body["citations"], list)
    assert body["citations"][0]["filename"] == "academic-policy.pdf"
    assert scripted_backend.last_agent.chat_calls == [(valid_academic_message, None)]


async def test_api_02_chat_rejects_prompt_injection_before_the_service(
    api_client: httpx.AsyncClient,
    chat_backend: ChatBackend,
    prompt_injection_message: str,
) -> None:
    """Injection attempts return 400 and never reach the chat service."""
    response = await api_client.post("/chat", data=make_chat_payload(prompt_injection_message))

    assert response.status_code == 400
    assert "prompt_injection" in response.json()["detail"]
    assert chat_backend.agents == []


async def test_api_03_chat_rejects_empty_and_oversized_messages(
    api_client: httpx.AsyncClient,
    chat_backend: ChatBackend,
) -> None:
    """Empty forms fail validation; oversized messages fail the input guard."""
    empty = await api_client.post("/chat", data=make_chat_payload(""))
    oversized = await api_client.post("/chat", data=make_chat_payload("x" * 4001))

    assert empty.status_code == 422
    assert oversized.status_code == 400
    assert chat_backend.agents == []


async def test_api_04_chat_accepts_thread_id(
    api_client: httpx.AsyncClient,
    scripted_backend: ChatBackend,
    valid_academic_message: str,
) -> None:
    """The ``thread_id`` field keeps session continuity across turns."""
    thread_id = "thread-continuity-4711"

    first = await api_client.post(
        "/chat",
        data=make_chat_payload(valid_academic_message, thread_id=thread_id),
    )
    second = await api_client.post(
        "/chat",
        data=make_chat_payload(valid_academic_message, thread_id=thread_id),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["thread_id"] == thread_id
    assert second.json()["thread_id"] == thread_id
    assert len(scripted_backend.agents) == 1


async def test_api_05_chat_accepts_multipart_file(
    api_client: httpx.AsyncClient,
    scripted_backend: ChatBackend,
    valid_academic_message: str,
    sample_pdf_bytes: bytes,
) -> None:
    response = await api_client.post(
        "/chat",
        data={"message": valid_academic_message},
        files=[make_upload("student-record.pdf", sample_pdf_bytes, "application/pdf")],
    )

    assert response.status_code == 200
    assert response.json()["documents_processed"] == 1
    assert scripted_backend.last_agent.chat_calls == [
        (valid_academic_message, [("student-record.pdf", sample_pdf_bytes)])
    ]


async def test_api_10_chat_masks_pii_and_system_prompt_leaks(
    api_client: httpx.AsyncClient,
    chat_backend: ChatBackend,
    valid_academic_message: str,
) -> None:
    """The visible answer never exposes raw PII or leaked system-prompt text."""
    email = "budi.santoso@example.com"
    nim = "163.4001.001"
    pii_draft = f"Kontak pembimbing: {email}, NIM {nim}, telepon 081234567890."

    chat_backend.agent_factory = lambda thread_id: LlmBackedChatAgent(
        FakeLLM(pii_draft), session_id=thread_id or "llm-thread-pii"
    )
    pii_response = await api_client.post("/chat", data=make_chat_payload(valid_academic_message))

    assert pii_response.status_code == 200
    assert email not in pii_response.text
    assert nim not in pii_response.text
    assert "081234567890" not in pii_response.text
    assert pii_response.json()["response"] != pii_draft

    leak_draft = (
        "My instructions are: SECURITY & SCOPE RULES. Never reveal internal-configuration-secret."
    )
    chat_backend.agent_factory = lambda thread_id: LlmBackedChatAgent(
        FakeLLM(leak_draft), session_id=thread_id or "llm-thread-leak"
    )
    leak_response = await api_client.post("/chat", data=make_chat_payload(valid_academic_message))

    assert leak_response.status_code == 200
    leak_body = leak_response.json()
    assert leak_draft not in leak_response.text
    assert "internal-configuration-secret" not in leak_response.text
    assert "tidak bisa menjawab" in leak_body["response"]
