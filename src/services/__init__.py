"""Application services and interface contracts."""

from src.services.contracts import (
    ChatAgent,
    DocumentIndexer,
    DocumentProcessor,
    DocumentRetriever,
    DocumentTextExtractor,
    DocumentUploadPreparer,
    LLMProvider,
    RetrievalStrategy,
    SessionManager,
    StateNode,
    StudentRecordRepository,
    StudentTextExtractor,
)
from src.services.document_processing import (
    DocumentIngestionService,
    DocumentProcessingResult,
    DocumentProcessingService,
)
from src.services.indexing import DocumentIndexingService
from src.services.intent import IntentClassifier
from src.services.response_generation import (
    ResponseContextBuilder,
    ResponseGenerationService,
)
from src.services.session_registry import InMemorySessionManager
from src.services.student_records import (
    StudentDataExtractionService,
    StudentIdentifierParser,
    StudentRecordFactory,
    StudentRecordService,
)
from src.utils.state import OCRResult

__all__ = [
    "ChatAgent",
    "DocumentIndexer",
    "DocumentIndexingService",
    "DocumentIngestionService",
    "DocumentProcessingResult",
    "DocumentProcessingService",
    "DocumentProcessor",
    "DocumentRetriever",
    "DocumentTextExtractor",
    "DocumentUploadPreparer",
    "InMemorySessionManager",
    "IntentClassifier",
    "LLMProvider",
    "OCRResult",
    "RetrievalStrategy",
    "ResponseContextBuilder",
    "ResponseGenerationService",
    "SessionManager",
    "StateNode",
    "StudentDataExtractionService",
    "StudentIdentifierParser",
    "StudentRecordFactory",
    "StudentRecordRepository",
    "StudentRecordService",
    "StudentTextExtractor",
]
