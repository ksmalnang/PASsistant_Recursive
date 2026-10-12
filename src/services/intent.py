"""Intent classification and routing helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from src.clients import get_decision_client
from src.clients.decision import (
    JevDecisionClient,
)
from src.clients.llm import get_llm
from src.config import get_settings
from src.services.contracts import InvokableLLM, LLMProvider
from src.utils.nodes.prompts import ROUTER_INTENT_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared keyword fast-paths (zero-cost, used by both classifiers)
# ---------------------------------------------------------------------------

academic_service_keywords = (
    "academic service",
    "academic services",
    "registration",
    "course registration",
    "enrollment verification",
    "leave of absence",
    "academic calendar",
    "deadline",
    "tuition",
    "scholarship",
    "advisor",
    "advising",
    "graduation requirement",
    "graduation requirements",
    "transcript request",
    "withdraw",
    "drop a course",
    "add a course",
    "kurikulum",
    "profil lulusan",
    "capaian pembelajaran",
    "cpl",
    "visi misi",
    "syarat kelulusan",
    "kalender akademik",
    "beasiswa",
    "jadwal kuliah",
    "mata kuliah",
    "rps",
    "dokumen",
    "cuti akademik",
    "cuti kuliah",
    "cuti",
    "absen",
    "tidak aktif",
    "tidak hadir",
    "masa studi",
    "batas studi",
    "perpanjangan studi",
    "drop out",
    "dikeluarkan",
    "status mahasiswa",
    "sanksi akademik",
    "sanksi",
    "evaluasi akademik",
    "perkuliahan",
    "semester",
    "heregistrasi",
    "spp",
    "pembayaran",
    "nilai",
    "ipk",
    "ip semester",
    "krs",
    "khs",
    "transkrip",
    "wisuda",
    "skripsi",
    "sidang",
    "bimbingan",
    "dosen pembimbing",
    "pembimbing akademik",
    "pedoman akademik",
    "peraturan akademik",
    "tata tertib",
    "kemahasiswaan",
    "layanan akademik",
    "administrasi akademik",
    "bagian akademik",
    "biro akademik",
    "helpdesk akademik",
    "layanan mahasiswa",
    "daftar ulang",
    "registrasi ulang",
    "aktivasi mahasiswa",
    "aktif kembali",
    "status aktif",
    "nonaktif",
    "pengaktifan kembali",
    "pengisian krs",
    "ubah krs",
    "revisi krs",
    "batal tambah",
    "jadwal ujian",
    "jadwal uts",
    "jadwal uas",
    "kelas",
    "ruang kuliah",
    "praktikum",
    "ujian",
    "uts",
    "uas",
    "remedial",
    "perbaikan nilai",
    "konversi nilai",
    "nilai akhir",
    "hasil studi",
    "legalisir",
    "legalisir ijazah",
    "salinan ijazah",
    "surat keterangan aktif",
    "surat keterangan mahasiswa",
    "surat rekomendasi",
    "dokumen akademik",
    "tugas akhir",
    "ta",
    "proposal skripsi",
    "seminar proposal",
    "seminar hasil",
    "ujian skripsi",
    "sidang skripsi",
    "revisi skripsi",
    "judul skripsi",
    "kemajuan studi",
    "monitoring studi",
    "evaluasi hasil studi",
    "peringatan akademik",
    "drop out mahasiswa",
    "biaya kuliah",
    "ukt",
    "uang kuliah tunggal",
    "tagihan kuliah",
    "pembayaran spp",
    "cicilan kuliah",
    "yudisium",
    "kelulusan",
    "ijazah",
    "pengambilan ijazah",
    "aturan kampus",
    "kebijakan akademik",
    "panduan akademik",
    "buku pedoman",
    "total sks",
    "sks lulus",
    "sks ditempuh",
    "beban sks",
)
student_record_keywords = (
    "my gpa",
    "gpa",
    "my grade",
    "my grades",
    "grade point average",
    "my transcript",
    "credits earned",
    "academic standing",
    "student id",
    "my record",
    "my records",
    "record for",
    "grades for",
    "transcript for",
    "ipk saya",
    "ip saya",
    "berapa ipk",
    "berapa ip saya",
    "nilai saya",
    "nilai kuliah",
    "hasil nilai",
    "nilai semester",
    "ip semester",
    "ips",
    "lihat nilai",
    "cek nilai",
    "nilai mata kuliah",
    "nilai per mata kuliah",
    "daftar nilai",
    "rekap nilai",
    "transkrip nilai",
    "lihat transkrip",
    "cek transkrip",
    "download transkrip",
    "transkrip akademik",
    "jumlah sks saya",
    "sks saya",
    "status akademik saya",
    "status mahasiswa saya",
    "status aktif saya",
    "nim saya",
    "nomor induk mahasiswa",
    "riwayat akademik",
    "riwayat studi",
    "rekam akademik",
    "nilai saya semester ini",
    "nilai saya semester lalu",
    "ipk terbaru",
    "transkrip terbaru",
    "nilai terbaru",
)

valid_intents = {
    "upload_document",
    "query_student",
    "query_document",
    "general_chat",
}

INTENT_LABELS: dict[str, str] = {
    "upload_document": "The user wants to upload, process, or extract text/data from a file (e.g., PDF, image, document).",
    "query_student": "Coming Soon",
    "query_document": "The user is asking about academic policies, graduate, courses, procedures, deadlines, services, or the contents of documents (uploaded or system-provided), including a follow-up that refers back to such a topic.",
    "general_chat": "A greeting or casual conversation, or a message with no academic referent even after resolving pronouns and ellipsis against the conversation history.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keyword_matches(text: str, keywords: tuple[str, ...]) -> bool:
    """Match short keywords with word boundaries and longer ones by inclusion."""
    for keyword in keywords:
        if len(keyword) <= 4:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                return True
            continue
        if keyword in text:
            return True
    return False


def _build_retrieval_intent(intent: str, user_text: str) -> dict[str, Any]:
    """Attach retrieval-routing fields for a classified intent."""
    return {
        "current_intent": intent,
        "requires_retrieval": True,
        "retrieval_query": user_text,
    }


def _build_classification(intent: str, user_text: str) -> dict[str, Any]:
    """Build the full classification result dict from a raw intent string."""
    updates: dict[str, Any] = {"current_intent": intent}
    if intent in {"query_document", "query_student"}:
        updates.update(_build_retrieval_intent(intent, user_text))
    elif intent == "upload_document":
        updates["requires_upload"] = True
    return updates


def _decision_state(user_text: str, history: str | None) -> str:
    """Combine the recent conversation with the message being classified."""
    if not history:
        return user_text
    return f"Conversation so far:\n{history}\n\nLatest user message to classify:\n{user_text}"


# ---------------------------------------------------------------------------
# Keyword fast-path classifier (no external calls)
# ---------------------------------------------------------------------------


def classify_by_keyword(user_text: str) -> str | None:
    """Try keyword fast-path classification. Returns intent string or None."""
    text_lower = user_text.lower()
    if _keyword_matches(text_lower, academic_service_keywords):
        return "query_document"
    if _keyword_matches(text_lower, student_record_keywords):
        return "query_student"
    return None


# ---------------------------------------------------------------------------
# Jev intent classifier (async, primary)
# ---------------------------------------------------------------------------


class JevIntentClassifier:
    """Classify intent using Jev (Choice primitive) with LLM fallback.

    Uses a single Jev Choice question for the 5 known intents.
    If Jev's confidence is below the configured threshold, falls back
    to the existing LLM-based classification.
    """

    def __init__(
        self,
        decision_client: JevDecisionClient | None = None,
        llm_provider: LLMProvider = get_llm,
        *,
        confidence_threshold: float | None = None,
    ) -> None:
        self._decision_client = decision_client
        self._llm_provider = llm_provider
        self._llm: InvokableLLM | None = None
        self._confidence_threshold = confidence_threshold

    def _resolve_threshold(self) -> float:
        """Return the effective confidence threshold, resolving from settings lazily."""
        if self._confidence_threshold is not None:
            return self._confidence_threshold
        settings = get_settings()
        return getattr(settings, "JEV_INTENT_CONFIDENCE_THRESHOLD", 0.65)

    async def classify_async(
        self,
        user_text: str,
        *,
        history: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, float | None, str | None]:
        """
        Classify intent asynchronously.

        Returns:
            Tuple of (intent, confidence, fallback_reason).
            confidence is None and fallback_reason is set when LLM fallback was used.
        """
        # 1. Keyword fast-path (zero-cost, synchronous)
        keyword_intent = classify_by_keyword(user_text)
        if keyword_intent is not None:
            logger.info(
                "Intent classified by keyword: %s",
                keyword_intent,
                extra={"session": session_id},
            )
            return keyword_intent, 1.0, None

        # 2. Resolve confidence threshold lazily
        threshold = self._resolve_threshold()

        # 3. Jev decision model
        try:
            client = self._decision_client or get_decision_client()
            answer = await client.ask_choice(
                state=_decision_state(user_text, history),
                instructions=(
                    "Classify the latest user message into exactly one intent "
                    "category, reading it in the context of the conversation. "
                    "Prefer query_document unless the message explicitly asks "
                    "about a specific student's own record data."
                ),
                criteria=INTENT_LABELS,
                session_id=session_id,
            )
            intent = answer.choice
            confidence = answer.confidence

            if intent not in valid_intents:
                logger.warning(
                    "Jev returned unknown intent %r; falling back to LLM",
                    intent,
                    extra={"session": session_id},
                )
                return await self._fallback_to_llm(
                    user_text, history=history, session_id=session_id
                )

            if confidence >= threshold:
                logger.info(
                    "Jev intent: %s (confidence=%.3f)",
                    intent,
                    confidence,
                    extra={"session": session_id},
                )
                return intent, confidence, None

            # Confidence too low -- fall back to LLM
            logger.info(
                "Jev confidence %.3f below %.3f for intent %r; using LLM fallback",
                confidence,
                threshold,
                intent,
                extra={"session": session_id},
            )
            return await self._fallback_to_llm(
                user_text,
                history=history,
                session_id=session_id,
                hint=intent,
            )

        except Exception as exc:
            logger.warning(
                "Jev intent classification failed: %s; falling back to LLM",
                exc,
                extra={
                    "session": session_id,
                    "jev_error": str(exc),
                    "jev_error_type": type(exc).__name__,
                },
            )
            return await self._fallback_to_llm(user_text, history=history, session_id=session_id)

    async def _fallback_to_llm(
        self,
        user_text: str,
        *,
        history: str | None = None,
        session_id: str | None = None,
        hint: str | None = None,
    ) -> tuple[str, float | None, str]:
        """Fall back to the LLM-based classifier."""
        try:
            llm = self._get_llm()
            if llm is None:
                return "general_chat", None, "no_llm_available"

            prompt = ROUTER_INTENT_PROMPT.format(message=_decision_state(user_text, history))
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            intent = str(response.content).strip().lower()
            if intent not in valid_intents:
                intent = "general_chat"

            logger.info(
                "LLM fallback intent: %s (hint=%s)",
                intent,
                hint,
                extra={"session": session_id},
            )
            return intent, None, "llm_fallback"

        except Exception as exc:
            logger.error(
                "LLM fallback classification also failed: %s",
                exc,
                extra={"session": session_id},
            )
            return "general_chat", None, f"fallback_error: {exc}"

    def _get_llm(self) -> InvokableLLM | None:
        """Resolve the fallback LLM lazily."""
        if self._llm is None:
            self._llm = self._llm_provider()
        return self._llm


# ---------------------------------------------------------------------------
# Synchronous wrapper (for legacy callers e.g. RouterNode)
# ---------------------------------------------------------------------------


class IntentClassifier:
    """
    Intent classifier that uses Jev as primary (async) with LLM fallback.

    ``classify()`` is async so the process-wide Jev HTTP client stays on the
    caller's event loop. Running it through asyncio.run() per turn reused that
    client across closed loops and failed with "Event loop is closed".
    """

    def __init__(
        self,
        llm_provider: LLMProvider = get_llm,
        *,
        decision_client: JevDecisionClient | None = None,
        confidence_threshold: float | None = None,
    ):
        self._jev_enabled: bool | None = None  # resolved lazily

        self._jev_kwargs = {
            "decision_client": decision_client,
            "llm_provider": llm_provider,
            "confidence_threshold": confidence_threshold,
        }

        self._llm_provider = llm_provider
        self._llm = None
        self._jev: JevIntentClassifier | None = None

    def _get_jev(self) -> JevIntentClassifier | None:
        """Lazily instantiate the Jev classifier based on settings."""
        if self._jev is not None:
            return self._jev
        if self._jev_enabled is False:
            return None
        # Resolve JEV_INTENT_ENABLED lazily (avoid Settings() during __init__)
        if self._jev_enabled is None:
            settings = get_settings()
            self._jev_enabled = getattr(settings, "JEV_INTENT_ENABLED", True)
            if not self._jev_enabled:
                return None
        self._jev = JevIntentClassifier(**self._jev_kwargs)
        return self._jev

    async def classify(
        self,
        user_text: str,
        session_id: str | None = None,
        *,
        history: str | None = None,
    ) -> dict[str, Any]:
        """Return routing metadata for a user message.

        Uses Jev as primary; falls back to LLM when confidence is low.
        """
        jev = self._get_jev()
        if jev is None:
            return await self._classify_llm_only(user_text, session_id=session_id, history=history)

        intent, confidence, fallback = await jev.classify_async(
            user_text, history=history, session_id=session_id
        )
        result = _build_classification(intent, user_text)
        if confidence is not None:
            result["intent_confidence"] = confidence
        if fallback is not None:
            result["intent_fallback"] = fallback
        return result

    async def _classify_llm_only(
        self,
        user_text: str,
        session_id: str | None = None,
        *,
        history: str | None = None,
    ) -> dict[str, Any]:
        """Original LLM-only classification path (Jev disabled)."""
        text_lower = user_text.lower()

        if _keyword_matches(text_lower, academic_service_keywords):
            return _build_retrieval_intent("query_document", user_text)

        if _keyword_matches(text_lower, student_record_keywords):
            return _build_retrieval_intent("query_student", user_text)

        try:
            llm = self._get_llm()
            if llm is None:
                return {"current_intent": "general_chat"}

            prompt = ROUTER_INTENT_PROMPT.format(message=_decision_state(user_text, history))
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            intent = str(response.content).strip().lower()
            if intent not in valid_intents:
                intent = "general_chat"

            result = _build_classification(intent, user_text)
            logger.info("LLM-only intent: %s", intent, extra={"session": session_id})
            return result

        except Exception as exc:
            logger.error("Intent classification failed: %s", exc)
            return {"current_intent": "general_chat", "error": str(exc)}

    def _get_llm(self):
        """Resolve the fallback LLM lazily."""
        if self._llm is None:
            self._llm = self._llm_provider()
        return self._llm
