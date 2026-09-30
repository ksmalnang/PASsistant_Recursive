"""Unit tests for the bounded in-memory session registry."""

from types import SimpleNamespace

from src.services.session_registry import InMemorySessionManager


def _manager(max_sessions: int) -> InMemorySessionManager:
    """Build a registry whose factory returns a minimal ChatAgent double."""
    return InMemorySessionManager(
        agent_factory=lambda thread_id: SimpleNamespace(session_id=thread_id or "generated"),
        max_sessions=max_sessions,
    )


def test_unit_01_returns_same_agent_for_existing_thread() -> None:
    """A known thread id resolves to its registered agent, not a new one."""
    manager = _manager(max_sessions=2)

    first_agent, first_thread = manager.get_or_create("thread-a")
    second_agent, second_thread = manager.get_or_create("thread-a")

    assert first_thread == second_thread == "thread-a"
    assert second_agent is first_agent


def test_unit_02_evicts_least_recently_used_session_at_capacity() -> None:
    """At capacity, the oldest session is dropped when a new one is created."""
    manager = _manager(max_sessions=2)
    first_agent, _ = manager.get_or_create("thread-a")
    manager.get_or_create("thread-b")

    manager.get_or_create("thread-c")

    assert not manager.contains("thread-a")
    assert manager.contains("thread-b")
    assert manager.contains("thread-c")
    recreated_agent, _ = manager.get_or_create("thread-a")
    assert recreated_agent is not first_agent


def test_unit_03_access_refreshes_recency() -> None:
    """Reading an existing session moves it to the most-recent position."""
    manager = _manager(max_sessions=2)
    manager.get_or_create("thread-a")
    manager.get_or_create("thread-b")

    manager.get_or_create("thread-a")
    manager.get_or_create("thread-c")

    assert manager.contains("thread-a")
    assert not manager.contains("thread-b")


def test_unit_04_contains_reflects_eviction() -> None:
    """contains() tracks registered sessions only."""
    manager = _manager(max_sessions=1)

    assert not manager.contains("thread-a")
    manager.get_or_create("thread-a")
    assert manager.contains("thread-a")

    manager.get_or_create("thread-b")

    assert not manager.contains("thread-a")
    assert manager.contains("thread-b")


def test_unit_05_new_session_created_when_cap_is_one() -> None:
    """A one-session cap evicts everything else, so the thread restarts fresh."""
    manager = _manager(max_sessions=1)
    first_agent, _ = manager.get_or_create("thread-a")

    manager.get_or_create("thread-b")
    restarted_agent, thread_id = manager.get_or_create("thread-a")

    assert thread_id == "thread-a"
    assert restarted_agent is not first_agent
    assert not manager.contains("thread-b")
