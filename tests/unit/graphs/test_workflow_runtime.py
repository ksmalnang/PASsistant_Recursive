"""Unit tests for the process-wide compiled application."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agent import PASsistantAgent
from src.graphs.workflow import get_compiled_app


def test_unit_01_default_agents_share_compiled_app() -> None:
    """Agents without injection reuse one compiled application and one node set."""
    first_agent = PASsistantAgent()
    second_agent = PASsistantAgent()

    assert first_agent.app is second_agent.app
    assert first_agent.app is get_compiled_app()
    assert first_agent.doc_processor is second_agent.doc_processor


def test_unit_02_custom_checkpointer_still_isolated() -> None:
    """An injected checkpointer yields a dedicated compiled application."""
    agent = PASsistantAgent(checkpointer=InMemorySaver())

    assert agent.app is not get_compiled_app()


def test_unit_03_shared_app_keeps_threads_isolated() -> None:
    """The shared checkpointer keeps conversation state per thread id."""
    app = get_compiled_app()
    thread_a = {"configurable": {"thread_id": "shared-app-thread-a"}}
    thread_b = {"configurable": {"thread_id": "shared-app-thread-b"}}

    app.update_state(thread_a, {"messages": [HumanMessage(content="hello from thread a")]})

    assert "hello from thread a" in str(app.get_state(thread_a).values)
    assert "hello from thread a" not in str(app.get_state(thread_b).values)
