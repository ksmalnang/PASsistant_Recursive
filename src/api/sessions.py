"""Session management for API agents."""

from src.agent import PASsistantAgent
from src.config import get_settings
from src.services.session_registry import InMemorySessionManager

session_manager = InMemorySessionManager(
    agent_factory=PASsistantAgent,
    max_sessions=get_settings().SESSION_MAX_ACTIVE,
)
