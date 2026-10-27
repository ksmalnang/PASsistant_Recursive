"""Unit tests for cross-turn evidence retention."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.utils.state import AgentState


def _noop(state: AgentState) -> dict:
    return {}


def _build_app():
    graph = StateGraph(AgentState)
    graph.add_node("noop", _noop)
    graph.set_entry_point("noop")
    graph.add_edge("noop", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_evidence_survives_a_fresh_turn_input() -> None:
    """The per-turn input resets retrieval fields, but not the evidence memory."""
    app = _build_app()
    config = {"configurable": {"thread_id": "evidence"}}

    app.invoke(
        AgentState(messages=[HumanMessage(content="turn one")]).model_dump(),
        config=config,
    )
    app.update_state(config, {"conversation_evidence": [{"chunk_id": "c1", "text": "rule"}]})

    result = app.invoke(
        AgentState(messages=[HumanMessage(content="turn two")]).model_dump(),
        config=config,
    )

    assert result["conversation_evidence"] == [{"chunk_id": "c1", "text": "rule"}]
    assert [message.content for message in result["messages"]] == ["turn one", "turn two"]


def test_resolved_query_and_scope_survive_a_fresh_turn_input() -> None:
    """The retrieval memory keeps the canonical query and scope across turns."""
    app = _build_app()
    config = {"configurable": {"thread_id": "retrieval-memory"}}

    app.invoke(
        AgentState(messages=[HumanMessage(content="turn one")]).model_dump(),
        config=config,
    )
    app.update_state(
        config,
        {"resolved_query": "syarat cuti akademik", "retrieval_scope": "transcript"},
    )

    result = app.invoke(
        AgentState(messages=[HumanMessage(content="turn two")]).model_dump(),
        config=config,
    )

    assert result["resolved_query"] == "syarat cuti akademik"
    assert result["retrieval_scope"] == "transcript"


def test_new_evidence_replaces_the_previous_set() -> None:
    """A completed turn stores only its own evidence, not an ever-growing list."""
    app = _build_app()
    config = {"configurable": {"thread_id": "evidence-replace"}}

    app.invoke(
        AgentState(messages=[HumanMessage(content="turn one")]).model_dump(),
        config=config,
    )
    app.update_state(config, {"conversation_evidence": [{"chunk_id": "c1"}]})
    app.update_state(config, {"conversation_evidence": [{"chunk_id": "c2"}]})

    assert app.get_state(config).values["conversation_evidence"] == [{"chunk_id": "c2"}]
