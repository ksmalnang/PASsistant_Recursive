"""Session-manager implementations for route handlers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

from src.services.contracts import ChatAgent


class InMemorySessionManager:
    """In-memory agent registry keyed by session id, bounded to max_sessions."""

    def __init__(
        self,
        agent_factory: Callable[[str | None], ChatAgent],
        max_sessions: int = 100,
    ):
        self._agent_factory = agent_factory
        self._sessions: OrderedDict[str, ChatAgent] = OrderedDict()
        self._max_sessions = max(1, max_sessions)

    def get_or_create(self, thread_id: str | None = None) -> tuple[ChatAgent, str]:
        """Return an existing agent or create a new one, evicting the least recently used."""
        if thread_id and thread_id in self._sessions:
            self._sessions.move_to_end(thread_id)
            return self._sessions[thread_id], thread_id

        agent = self._agent_factory(thread_id)
        self._sessions[agent.session_id] = agent
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return agent, agent.session_id

    def contains(self, thread_id: str) -> bool:
        """Whether a session is currently registered."""
        return thread_id in self._sessions
