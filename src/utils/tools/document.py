"""Local raw-upload storage helpers."""

import logging
import mimetypes
import re
import uuid
from collections.abc import Container
from pathlib import Path

from src.config import get_settings
from src.utils.state import DocumentType, DocumentUpload

logger = logging.getLogger(__name__)


class DocumentTools:
    """Persist raw uploaded files and infer lightweight metadata.

    This helper only writes uploads to local storage and classifies filenames.
    It does not run OCR, chunk documents, or write anything to the vector store.
    """

    _DOCUMENT_TYPE_KEYWORDS: tuple[tuple[DocumentType, tuple[str, ...]], ...] = (
        (
            DocumentType.TRANSCRIPT,
            (
                "transcript",
                "grades",
                "grade",
                "academic record",
                "nilai",
                "khs",
                "kartu hasil studi",
                "transkrip nilai",
                "rekap nilai",
                "daftar nilai",
                "hasil studi",
                "ipk",
                "ips",
                "hasil nilai kuliah",
                "nilai akademik",
                "laporan nilai",
                "progress studi",
            ),
        ),
        (
            DocumentType.ID_CARD,
            (
                "id card",
                "identity",
                "ktp",
                "kartu tanda penduduk",
                "passport",
                "paspor",
                "license",
                "sim",
                "student card",
                "kartu mahasiswa",
                "kartu identitas",
                "identitas diri",
                "kartu identitas mahasiswa",
                "kartu tanda mahasiswa",
                "ktm",
                "id mahasiswa",
                "identitas mahasiswa",
            ),
        ),
        (
            DocumentType.APPLICATION,
            (
                "application",
                "apply",
                "admission",
                "registration form",
                "formulir pendaftaran",
                "form",
                "daftar",
                "pendaftaran",
                "form pendaftaran",
                "apply kuliah",
                "pengajuan",
                "ajukan",
                "submit form",
                "isi formulir",
                "form online",
                "pendaftaran online",
            ),
        ),
        (
            DocumentType.RECOMMENDATION,
            (
                "recommendation",
                "reference letter",
                "referral letter",
                "surat rekomendasi",
                "surat referensi",
                "surat pengantar",
                "letter of recommendation",
                "rekomendasi dosen",
            ),
        ),
        (
            DocumentType.CURRICULUM,
            (
                "curriculum",
                "kurikulum",
                "cpl",
                "capaian pembelajaran",
                "profil lulusan",
                "struktur kurikulum",
                "mata kuliah wajib",
                "mata kuliah pilihan",
                "struktur mata kuliah",
                "daftar mata kuliah",
                "kurikulum terbaru",
                "kurikulum prodi",
            ),
        ),
        (
            DocumentType.SYLLABUS,
            (
                "syllabus",
                "silabus",
                "rps",
                "course outline",
                "semester plan",
                "rencana pembelajaran semester",
                "deskripsi mata kuliah",
                "rincian materi",
                "materi kuliah",
                "topik perkuliahan",
                "outline mata kuliah",
            ),
        ),
        (
            DocumentType.CERTIFICATE,
            (
                "certificate",
                "sertifikat",
                "ijazah",
                "diploma",
                "surat keterangan",
                "surat keterangan aktif",
                "surat keterangan lulus",
                "surat keterangan mahasiswa",
                "legalisir",
                "legalisir ijazah",
                "bukti lulus",
                "surat lulus",
                "dokumen kelulusan",
                "pengganti ijazah",
            ),
        ),
        (
            DocumentType.INVOICE,
            (
                "invoice",
                "billing",
                "receipt",
                "tuition bill",
                "payment slip",
                "kwitansi",
                "tagihan",
                "tagihan kuliah",
                "ukt",
                "uang kuliah tunggal",
                "bukti pembayaran",
                "tagihan ukt",
                "biaya kuliah",
                "rincian pembayaran",
                "status pembayaran",
                "history pembayaran",
            ),
        ),
        (
            DocumentType.POLICY,
            (
                "policy",
                "regulation",
                "guideline",
                "handbook",
                "peraturan",
                "kebijakan",
                "panduan",
                "pedoman",
                "buku pedoman",
                "aturan kampus",
                "tata tertib",
                "aturan akademik",
                "regulasi kampus",
                "ketentuan akademik",
                "aturan mahasiswa",
            ),
        ),
    )

    def __init__(self):
        settings = get_settings()
        self.raw_dir = settings.raw_data_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
    ) -> DocumentUpload:
        """
        Save an uploaded file to raw storage only.

        Args:
            file_bytes: Raw file bytes
            original_filename: Original filename

        Returns:
            DocumentUpload with local storage metadata
        """
        document_id = str(uuid.uuid4())
        file_extension = Path(original_filename).suffix
        storage_filename = f"{document_id}{file_extension}"
        storage_path = self.raw_dir / storage_filename
        storage_path.write_bytes(file_bytes)

        mime_type, _ = mimetypes.guess_type(original_filename)
        document = DocumentUpload(
            document_id=document_id,
            filename=original_filename,
            file_path=str(storage_path),
            document_type=self._detect_document_type(original_filename),
            mime_type=mime_type or "application/octet-stream",
            file_size=len(file_bytes),
        )

        logger.info(
            "Document saved",
            extra={
                "document_id": document_id,
                "uploaded_filename": original_filename,
                "file_size_bytes": len(file_bytes),
            },
        )
        return document

    def _detect_document_type(self, filename: str) -> DocumentType:
        """Infer document type from filename keywords."""
        normalized_name = " ".join(re.findall(r"[a-z0-9]+", filename.lower()))
        tokens = set(normalized_name.split())

        for document_type, keywords in self._DOCUMENT_TYPE_KEYWORDS:
            if self._matches_keywords(normalized_name, tokens, keywords):
                return document_type

        return DocumentType.OTHER

    def _matches_keywords(
        self,
        normalized_name: str,
        tokens: Container[str],
        keywords: tuple[str, ...],
    ) -> bool:
        """Return whether a normalized filename matches any keyword or phrase."""
        for keyword in keywords:
            normalized_keyword = " ".join(re.findall(r"[a-z0-9]+", keyword.lower()))
            if not normalized_keyword:
                continue
            if " " in normalized_keyword:
                if normalized_keyword in normalized_name:
                    return True
                continue
            if normalized_keyword in tokens:
                return True
        return False

    def resolve_stored_file(self, document_id: str, raw_name: str | None = None) -> Path | None:
        """Locate the raw file stored for a document id.

        Prefers the cache metadata name, then the exact id, then the id plus
        its original extension. Returns None when nothing matches.
        """
        if not re.fullmatch(r"[A-Za-z0-9._-]+", document_id):
            return None
        if raw_name and Path(raw_name).name == raw_name:
            candidate = self.raw_dir / raw_name
            if candidate.is_file():
                return candidate
        exact = self.raw_dir / document_id
        if exact.is_file():
            return exact
        for candidate in sorted(self.raw_dir.glob(f"{document_id}.*")):
            if candidate.is_file():
                return candidate
        return None

    def delete_file(self, file_path: str) -> None:
        """Delete a stored file."""
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info("Deleted file: %s", file_path)
