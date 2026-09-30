"""Fixtures and live-service gating for the opt-in system suite.

System tests exercise the public API against real Qdrant, OCR, and LLM
providers. They stay skipped unless ``PASSISTANT_SYSTEM_TESTS=1`` is exported,
so default local and CI runs never spend model credits or touch real indexes.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

import httpx
import pytest
import pytest_asyncio
from qdrant_client import QdrantClient

from src.config import get_settings
from src.utils.state import DocumentUpload, ProcessingStatus
from tests.factories import build_pdf_bytes

SYSTEM_TESTS_ENV_VAR = "PASSISTANT_SYSTEM_TESTS"
OPT_IN_VALUES = {"1", "true", "yes", "on"}
QDRANT_PROBE_TIMEOUT_SECONDS = 3
SYSTEM_REQUEST_TIMEOUT_SECONDS = 120.0


@dataclass(slots=True)
class SystemEnvironment:
    """Live services available to a system test."""

    qdrant: QdrantClient
    collection_name: str
    ocr_enabled: bool


@dataclass(slots=True)
class KnowledgeBase:
    """Seed and clean up documents in the shared live knowledge base."""

    known_document_ids: list[str] = field(default_factory=list)

    async def index_pdf(self, lines: list[str], filename: str) -> DocumentUpload:
        """Ingest a generated PDF through the real OCR and indexing pipeline."""
        from src.utils.nodes.document_processing import DocumentProcessingNode

        document = await DocumentProcessingNode().ingest_upload(build_pdf_bytes(lines), filename)
        if document.processing_status is not ProcessingStatus.COMPLETED:
            raise AssertionError(
                f"document ingestion failed for {filename}: {document.processing_error}"
            )
        report = document.ingestion_report or {}
        if not report.get("chunk_count"):
            raise AssertionError(f"document {filename} produced no indexed chunks: {report}")
        self.register(document.document_id)
        return document

    def register(self, document_id: str) -> None:
        """Remember a document whose chunks must be removed after the test."""
        self.known_document_ids.append(document_id)

    def cleanup(self) -> None:
        """Delete seeded chunks from the shared index."""
        from src.utils.tools import VectorStoreTools

        if not self.known_document_ids:
            return
        with contextlib.suppress(Exception):
            tools = VectorStoreTools()
            for document_id in self.known_document_ids:
                tools.delete_document_chunks(document_id)


def _require_opt_in() -> None:
    """Skip the calling test unless system testing is explicitly enabled."""
    if os.getenv(SYSTEM_TESTS_ENV_VAR, "").strip().lower() not in OPT_IN_VALUES:
        pytest.skip(
            f"system tests are opt-in; export {SYSTEM_TESTS_ENV_VAR}=1 in an "
            "environment with live Qdrant and LLM credentials"
        )


@pytest.fixture
def system_opt_in() -> None:
    """Skip unless system testing is enabled, without requiring live services."""
    _require_opt_in()


@pytest.fixture(scope="session")
def system_environment() -> SystemEnvironment:
    """Skip unless system testing is opted in and live services are reachable."""
    _require_opt_in()

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        pytest.skip("system tests need OPENAI_API_KEY for embeddings and the LLM")

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=QDRANT_PROBE_TIMEOUT_SECONDS,
    )
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant is not reachable at {settings.QDRANT_URL}: {exc}")

    return SystemEnvironment(
        qdrant=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        ocr_enabled=bool(settings.ZHIPU_API_KEY),
    )


@pytest.fixture
def require_ocr(system_environment: SystemEnvironment) -> None:
    """Skip document-ingestion tests when the OCR provider is not configured."""
    if not system_environment.ocr_enabled:
        pytest.skip("document ingestion needs ZHIPU_API_KEY for GLM-OCR")


@pytest.fixture
def knowledge_base(system_environment: SystemEnvironment) -> Iterator[KnowledgeBase]:
    """Provide a seeding helper that removes its documents afterwards."""
    helper = KnowledgeBase()
    yield helper
    helper.cleanup()


@pytest_asyncio.fixture
async def system_api_client(live_server_url: str) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client bound to the live server with a generous system-test timeout."""
    async with httpx.AsyncClient(
        base_url=live_server_url,
        timeout=SYSTEM_REQUEST_TIMEOUT_SECONDS,
    ) as client:
        yield client
