"""Session management for API agents."""

from src.agent import StudentRecordsAgent
from src.config import get_settings
from src.services.contracts import ChatAgent
from src.services.session_registry import InMemorySessionManager

session_manager = InMemorySessionManager(
    agent_factory=StudentRecordsAgent,
    max_sessions=get_settings().SESSION_MAX_ACTIVE,
)


def get_or_create_agent(thread_id: str | None = None) -> tuple[ChatAgent, str]:
    """Get existing agent or create a new one for the session."""
    agent, active_thread_id = session_manager.get_or_create(thread_id)
    return agent, active_thread_id
