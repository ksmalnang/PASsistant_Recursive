"""Unit tests for the no-context fallback response node."""

from langchain_core.messages import AIMessage, HumanMessage

from src.graphs.workflow import build_fallback_response
from src.utils.state import AgentState


def test_unit_04_fallback_message_survives_streamed_state_round_trip() -> None:
    """The fallback node emits a real message so streamed state re-validates."""
    state = AgentState(session_id="fallback-thread")
    state.messages.append(HumanMessage(content="A question with no indexed context"))

    # Mirrors Agent.stream_chat: dump the state, merge the node payload, re-validate.
    snapshot = state.model_dump()
    update = build_fallback_response(state)
    snapshot["messages"].extend(update["messages"])
    snapshot["draft_response"] = update["draft_response"]

    final_state = AgentState(**snapshot)

    assert isinstance(final_state.messages[-1], AIMessage)
    assert final_state.messages[-1].content == update["draft_response"]
