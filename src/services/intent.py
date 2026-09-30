"""Intent classification and routing helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from src.clients.llm import get_llm
from src.services.contracts import LLMProvider
from src.utils.nodes.prompts import ROUTER_INTENT_PROMPT

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify user messages into workflow intents."""

    upload_keywords = (
        "upload",
        "process this file",
        "extract from",
        "parse this",
        "ingest",
        "knowledge base",
    )
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
        "total sks",
        "sks lulus",
        "sks ditempuh",
        "beban sks",
        "status akademik saya",
        "status kuliah",
        "status mahasiswa saya",
        "status aktif saya",
        "status studi",
        "nim saya",
        "nomor induk mahasiswa",
        "data mahasiswa",
        "data saya",
        "profil mahasiswa",
        "riwayat akademik",
        "riwayat studi",
        "rekam akademik",
        "data akademik",
        "catatan akademik",
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
        "manage_record",
        "general_chat",
    }

    def __init__(self, llm_provider: LLMProvider = get_llm):
        self._llm_provider = llm_provider
        self._llm = None

    def classify(self, user_text: str, session_id: str | None = None) -> dict[str, Any]:
        """Return routing metadata for a user message."""
        user_text_lower = user_text.lower()

        if self._keyword_matches(user_text_lower, self.upload_keywords):
            return {
                "current_intent": "upload_document",
                "requires_upload": True,
            }

        if self._keyword_matches(user_text_lower, self.academic_service_keywords):
            return self._build_retrieval_intent("query_document", user_text)

        if self._keyword_matches(user_text_lower, self.student_record_keywords):
            return self._build_retrieval_intent("query_student", user_text)

        try:
            llm = self._get_llm()
            if llm is None:
                return {"current_intent": "general_chat"}

            prompt = ROUTER_INTENT_PROMPT.format(message=user_text)
            response = llm.invoke([HumanMessage(content=prompt)])
            intent = str(response.content).strip().lower()
            if intent not in self.valid_intents:
                intent = "general_chat"

            updates: dict[str, Any] = {"current_intent": intent}
            if intent in {"query_document", "query_student"}:
                updates.update(self._build_retrieval_intent(intent, user_text))
            elif intent == "upload_document":
                updates["requires_upload"] = True

            logger.info("Classified intent: %s", intent, extra={"session": session_id})
            return updates
        except Exception as exc:
            logger.error("Intent classification failed: %s", exc)
            return {"current_intent": "general_chat", "error": str(exc)}

    def _get_llm(self):
        """Resolve the current LLM lazily."""
        if self._llm is None:
            self._llm = self._llm_provider()
        return self._llm

    def _build_retrieval_intent(self, intent: str, user_text: str) -> dict[str, Any]:
        """Attach retrieval-routing fields for a classified intent."""
        return {
            "current_intent": intent,
            "requires_retrieval": True,
            "retrieval_query": user_text,
        }

    def _keyword_matches(self, text: str, keywords: tuple[str, ...]) -> bool:
        """Match short keywords with word boundaries and longer ones by inclusion."""
        for keyword in keywords:
            if len(keyword) <= 4:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return True
                continue
            if keyword in text:
                return True
        return False
