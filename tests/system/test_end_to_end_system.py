"""Opt-in end-to-end tests across the real workflow, retrieval, and providers.

Enable with ``PASSISTANT_SYSTEM_TESTS=1`` in an environment that has live
Qdrant, embeddings, and LLM credentials (plus ``ZHIPU_API_KEY`` for the
document-ingestion scenarios). Without the flag every test skips.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest

from src.utils.tools import VectorStoreTools
from tests.factories import (
    build_pdf_bytes,
    make_chat_payload,
    make_record_query_by_id_message,
    make_record_seed_message,
    make_student_record,
    make_upload,
)
from tests.helpers import parse_sse_frames
from tests.system.conftest import KnowledgeBase, SystemEnvironment

pytestmark = pytest.mark.system

ACADEMIC_FALLBACK_MARKERS = ("couldn't find", "could not find", "no relevant", "tidak ditemukan")
NOT_FOUND_MARKERS = ("not found", "no record", "couldn't find", "tidak ditemukan", "tidak ada")


class _ScriptedPipelineLLM:
    """LLM double that classifies through the router and scripts the answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, messages: Sequence[Any], /, **kwargs: Any) -> SimpleNamespace:
        """Return the routing label for router prompts, else the scripted answer."""
        del kwargs
        prompt_text = "\n".join(_message_text(message) for message in messages)
        if "intent classifier" in prompt_text:
            return SimpleNamespace(content="general_chat")
        return SimpleNamespace(content=self.answer)


def _message_text(message: Any) -> str:
    """Flatten one LangChain message into plain text."""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


async def test_system_001_end_to_end_academic_question(
    system_api_client: httpx.AsyncClient,
    system_environment: SystemEnvironment,
    knowledge_base: KnowledgeBase,
    require_ocr: None,
) -> None:
    """A question answered by an indexed policy document returns the fact and a citation."""
    filename = f"system-policy-{uuid4().hex[:8]}.pdf"
    document = await knowledge_base.index_pdf(
        [
            "Universitas Pasundan - Program Studi Informatika",
            "Panduan syarat kelulusan terbaru untuk mahasiswa:",
            "Mahasiswa wajib menempuh minimal 148 SKS agar dapat dinyatakan lulus.",
            "Indeks prestasi kumulatif minimum untuk kelulusan adalah 3.10.",
        ],
        filename,
    )

    response = await system_api_client.post(
        "/chat",
        data=make_chat_payload(
            "Berapa jumlah SKS minimum untuk kelulusan Program Studi Informatika?"
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "query_document"
    assert "148" in body["response"]
    assert body["citations"], "a document-backed answer must cite its source document"
    assert any(citation["filename"] == filename for citation in body["citations"])
    assert any(citation["document_id"] == document.document_id for citation in body["citations"])


async def test_system_002_end_to_end_knowledge_base_upload(
    system_api_client: httpx.AsyncClient,
    system_environment: SystemEnvironment,
    knowledge_base: KnowledgeBase,
    require_ocr: None,
) -> None:
    """A PDF uploaded through the API becomes answerable and citable."""
    filename = f"system-upload-{uuid4().hex[:8]}.pdf"
    unique_code = f"BEA-{uuid4().hex[:6].upper()}"
    pdf_bytes = build_pdf_bytes(
        [
            "Universitas Pasundan - Panduan Beasiswa",
            f"Pendaftar beasiswa wajib mencantumkan kode {unique_code} pada formulir.",
            "Batas akhir pendaftaran beasiswa adalah 31 Maret setiap tahun.",
        ]
    )

    upload = await system_api_client.post(
        "/upload",
        files=[make_upload(filename, pdf_bytes, "application/pdf")],
    )
    assert upload.status_code == 201
    uploaded = upload.json()[0]
    assert uploaded["status"] == "completed", uploaded.get("error")
    assert uploaded["error"] is None
    knowledge_base.register(uploaded["document_id"])

    query = await system_api_client.post(
        "/chat",
        data=make_chat_payload(
            "Kode apa yang wajib dicantumkan pendaftar beasiswa pada formulir pendaftaran?"
        ),
    )

    assert query.status_code == 200
    body = query.json()
    assert unique_code.lower() in body["response"].lower()
    assert any(citation["filename"] == filename for citation in body["citations"])


async def test_system_003_empty_knowledge_base_uses_honest_fallback(
    system_api_client: httpx.AsyncClient,
    system_environment: SystemEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty collection produces an honest fallback instead of a fabricated answer."""
    empty_collection = f"system_empty_{uuid4().hex[:8]}"
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", empty_collection)

    try:
        response = await system_api_client.post(
            "/chat",
            data=make_chat_payload(
                "Berapa jumlah SKS minimum untuk kelulusan Program Studi Informatika?",
                thread_id=f"system-empty-{uuid4().hex[:8]}",
            ),
        )

        assert response.status_code == 200
        body = response.json()
        assert body["citations"] == []
        assert any(marker in body["response"].lower() for marker in ACADEMIC_FALLBACK_MARKERS)
    finally:
        with contextlib.suppress(Exception):
            system_environment.qdrant.delete_collection(empty_collection)


async def test_system_004_student_record_lookup_is_scoped(
    system_api_client: httpx.AsyncClient,
    system_environment: SystemEnvironment,
) -> None:
    """A seeded record is returned by identifier and never mixed with another record."""
    seeded_id = "STU_A1B2C3D4"
    seeded = make_student_record(
        student_id=seeded_id,
        full_name="Rina Kusuma",
        email="rina.kusuma@example.com",
    )
    other = make_student_record(
        student_id="STU_Z9Y8X7W6",
        full_name="Budi Santoso",
        email="budi.santoso@example.com",
    )
    thread_id = f"system-record-{uuid4().hex[:8]}"

    other_seed = await system_api_client.post(
        "/chat",
        data=make_chat_payload(make_record_seed_message(other), thread_id=thread_id),
    )
    assert other_seed.status_code == 200
    assert other_seed.json()["intent"] == "manage_record"

    seeded_turn = await system_api_client.post(
        "/chat",
        data=make_chat_payload(make_record_seed_message(seeded), thread_id=thread_id),
    )
    assert seeded_turn.status_code == 200
    assert seeded_turn.json()["intent"] == "manage_record"

    lookup = await system_api_client.post(
        "/chat",
        data=make_chat_payload(
            make_record_query_by_id_message(seeded_id),
            thread_id=thread_id,
        ),
    )

    assert lookup.status_code == 200
    body = lookup.json()
    assert body["intent"] == "query_student"
    assert seeded_id in body["response"]
    assert other.student_id not in body["response"]
    assert other.email not in body["response"]


async def test_system_005_unknown_student_record_falls_back(
    system_api_client: httpx.AsyncClient,
    system_environment: SystemEnvironment,
) -> None:
    """An unknown identifier never yields an invented student record."""
    unknown_id = "STU_00000000"

    response = await system_api_client.post(
        "/chat",
        data=make_chat_payload(
            make_record_query_by_id_message(unknown_id),
            thread_id=f"system-unknown-{uuid4().hex[:8]}",
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "query_student"
    assert "STU_" not in body["response"], "no student identifier may be invented"
    assert any(marker in body["response"].lower() for marker in NOT_FOUND_MARKERS)


async def test_system_006_prompt_injection_is_blocked_end_to_end(
    system_api_client: httpx.AsyncClient,
    system_opt_in: None,
    prompt_injection_message: str,
) -> None:
    """Injection text is rejected before any workflow execution and leaks nothing."""
    response = await system_api_client.post(
        "/chat",
        data=make_chat_payload(prompt_injection_message),
    )

    assert response.status_code == 400
    assert "prompt_injection" in response.json()["detail"]
    for marker in ("RESPONSE_SYSTEM_PROMPT", "ROUTER_INTENT_PROMPT", "SECURITY & SCOPE RULES"):
        assert marker not in response.text


async def test_system_007_output_guard_masks_pii_in_generated_answer(
    system_api_client: httpx.AsyncClient,
    system_opt_in: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PII produced by the generation path never reaches the caller unmasked."""
    email = "rina.kusuma@example.com"
    nim = "163.4001.001"
    scripted_llm = _ScriptedPipelineLLM(
        f"Hubungi pembimbing akademik di {email} atau NIM {nim} untuk konfirmasi."
    )
    monkeypatch.setattr("src.clients.llm._llm_instance", None)
    monkeypatch.setattr("src.clients.llm.ChatOpenAI", lambda **kwargs: scripted_llm)

    response = await system_api_client.post("/chat", data=make_chat_payload("halo"))

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "general_chat"
    assert email not in response.text
    assert nim not in response.text
    assert "[email disamarkan]" in body["response"]
    assert "[NIM disamarkan]" in body["response"]


async def test_system_008_streaming_resume_after_disconnect(
    system_api_client: httpx.AsyncClient,
    system_environment: SystemEnvironment,
) -> None:
    """A resumed stream replays only the events after the requested cursor."""
    thread_id = f"system-stream-{uuid4().hex[:8]}"
    payload = make_chat_payload(
        "Berapa jumlah SKS minimum untuk kelulusan Program Studi Informatika?",
        thread_id=thread_id,
    )

    started = await system_api_client.post("/chat/stream", data=payload)

    assert started.status_code == 200
    frames = parse_sse_frames(started.text)
    assert frames, "a streamed run must emit at least one event"
    assert [frame["event"] for frame in frames][0] == "run.started"
    sequences = [frame["payload"]["sequence"] for frame in frames]
    assert sequences == sorted(set(sequences))

    cursor_index = min(2, len(frames) - 1)
    cursor = frames[cursor_index]
    resumed = await system_api_client.post(
        "/chat/stream",
        data=payload,
        headers={"Last-Event-ID": cursor["id"]},
    )

    assert resumed.status_code == 200
    resumed_frames = parse_sse_frames(resumed.text)
    resumed_sequences = [frame["payload"]["sequence"] for frame in resumed_frames]
    assert resumed_sequences, "the resume must replay events after the cursor"
    assert all(sequence > cursor["payload"]["sequence"] for sequence in resumed_sequences)
    assert resumed_sequences == sorted(set(resumed_sequences))
    assert "run.started" not in [frame["event"] for frame in resumed_frames]
    assert resumed_frames[-1]["event"] in {"run.completed", "run.failed"}
    assert {frame["payload"]["thread_id"] for frame in resumed_frames} == {thread_id}


async def test_system_009_multi_user_concurrency_keeps_state_consistent(
    system_api_client: httpx.AsyncClient,
    system_environment: SystemEnvironment,
    knowledge_base: KnowledgeBase,
    require_ocr: None,
) -> None:
    """Concurrent uploads and same-thread chats stay independent and consistent."""
    suffix = uuid4().hex[:8]
    filename_a = f"system-concurrent-a-{suffix}.pdf"
    filename_b = f"system-concurrent-b-{suffix}.pdf"
    code_a = f"BEA-{uuid4().hex[:6].upper()}"
    code_b = f"BEA-{uuid4().hex[:6].upper()}"

    uploads = await asyncio.gather(
        system_api_client.post(
            "/upload",
            files=[
                make_upload(
                    filename_a,
                    build_pdf_bytes(
                        [
                            "Universitas Pasundan - Panduan Beasiswa Unggulan",
                            f"Kode beasiswa unggulan adalah {code_a}.",
                        ]
                    ),
                    "application/pdf",
                )
            ],
        ),
        system_api_client.post(
            "/upload",
            files=[
                make_upload(
                    filename_b,
                    build_pdf_bytes(
                        [
                            "Universitas Pasundan - Jadwal Beasiswa Unggulan",
                            "Batas pendaftaran beasiswa unggulan adalah 31 Maret.",
                            f"Sebutkan kode {code_b} saat mengisi formulir.",
                        ]
                    ),
                    "application/pdf",
                )
            ],
        ),
    )

    results = [response.json()[0] for response in uploads]
    assert [response.status_code for response in uploads] == [201, 201]
    assert len({result["document_id"] for result in results}) == 2
    assert [result["filename"] for result in results] == [filename_a, filename_b]
    for result in results:
        knowledge_base.register(result["document_id"])

    tools = VectorStoreTools()
    indexed_a = tools.find_document_ids_by_filename(filename_a)
    indexed_b = tools.find_document_ids_by_filename(filename_b)
    assert indexed_a == {results[0]["document_id"]}
    assert indexed_b == {results[1]["document_id"]}

    thread_id = f"system-concurrent-thread-{suffix}"
    chats = await asyncio.gather(
        system_api_client.post(
            "/chat",
            data=make_chat_payload("Sebutkan kode beasiswa unggulan.", thread_id=thread_id),
        ),
        system_api_client.post(
            "/chat",
            data=make_chat_payload(
                "Kapan batas pendaftaran beasiswa unggulan?", thread_id=thread_id
            ),
        ),
    )

    assert [response.status_code for response in chats] == [200, 200]
    bodies = [response.json() for response in chats]
    assert {body["thread_id"] for body in bodies} == {thread_id}
    assert [body["intent"] for body in bodies] == ["query_document", "query_document"]
    assert all(body["response"].strip() for body in bodies)
    assert bodies[0]["id"] != bodies[1]["id"]
    # Answer content quality for a single request is covered by ST-001 and ST-002;
    # here the contract is that concurrent turns neither fail nor mix up responses.
