"""
Academic Services and Student Records Chatbot - Main Agent Entry Point

This module provides the primary interface for running the LangGraph agent.
It handles configuration, compilation, and execution of the chatbot workflow.

Usage:
    # Programmatic usage
    from src.agent import StudentRecordsAgent

    agent = StudentRecordsAgent()
    result = await agent.chat("Upload this transcript", files=[("transcript.pdf", bytes)])

    # LangGraph deployment (langgraph.json entry point)
    from src.agent import compiled_app
    # The `compiled_app` is the compiled LangGraph application for deployment
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.config import configure_logging, get_settings
from src.graphs.workflow import compile_app, get_compiled_app
from src.services.contracts import AgentStreamUpdate
from src.utils.nodes.document_processing import DocumentProcessingNode, get_document_processor
from src.utils.state import AgentState

configure_logging()
logger = logging.getLogger(__name__)


def _message_content_to_text(content: Any) -> str:
    """Normalize LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    return str(content)


class StudentRecordsAgent:
    """
    High-level interface for the academic services and student records chatbot.

    Encapsulates the LangGraph workflow and provides clean methods
    for common operations like chatting and document uploads.

    Example:
        agent = StudentRecordsAgent()

        # Simple chat
        response = await agent.chat("What is John's GPA?")

        # Upload and process document
        response = await agent.chat(
            "Process this transcript",
            files=[("john_transcript.pdf", file_bytes)]
        )
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        app_factory: Callable[..., Any] | None = None,
        document_processor: DocumentProcessingNode | None = None,
        checkpointer: InMemorySaver | None = None,
    ):
        """
        Initialize the agent.

        Args:
            session_id: Optional session identifier for persistence
            app_factory: Optional graph factory; the process-wide app is reused by default
            document_processor: Optional document processor override
            checkpointer: Optional checkpointer; forces a dedicated compiled app
        """
        self.settings = get_settings()
        self.session_id = session_id or str(uuid.uuid4())

        if app_factory is None and checkpointer is None:
            self.app = get_compiled_app()
        else:
            graph_factory: Callable[..., Any] = app_factory or compile_app
            self.checkpointer = checkpointer or InMemorySaver()
            self.app = graph_factory(checkpointer=self.checkpointer)

        self.doc_processor = document_processor or get_document_processor()

        logger.info("Agent initialized with session: %s", self.session_id)

    @property
    def thread_id(self) -> str:
        """Public thread identifier alias for the existing session id."""
        return self.session_id

    async def chat(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]] = None,
    ) -> str:
        """
        Send a message to the agent with optional file uploads.

        Args:
            message: User message text
            files: Optional list of (filename, file_bytes) tuples to upload

        Returns:
            Agent response text
        """
        final_state = await self.chat_with_state(message, files=files)
        return self._extract_response_text(final_state)

    async def chat_with_state(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]] = None,
    ) -> AgentState:
        """
        Run one chat turn and return the final workflow state.

        Args:
            message: User message text
            files: Optional list of (filename, file_bytes) tuples to upload

        Returns:
            Final agent state after graph execution
        """
        state = self._build_initial_state(message, files)
        config = self._build_run_config()

        try:
            result = await self.app.ainvoke(
                state.model_dump(),
                config=config,
            )
            return AgentState(**result)

        except Exception as exc:
            logger.error("Agent execution failed: %s", exc, exc_info=True)
            return AgentState(
                session_id=self.session_id,
                messages=state.messages,
                error=str(exc),
                draft_response=f"I encountered an error: {str(exc)}. Please try again.",
            )

    async def stream_chat(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]] = None,
    ) -> AsyncIterator[AgentStreamUpdate]:
        """
        Stream agent responses for real-time updates.

        Yields intermediate state updates during graph execution.

        Args:
            message: User message text
            files: Optional list of (filename, file_bytes) tuples

        Yields:
            Structured status and final-state updates
        """
        state = self._build_initial_state(message, files)
        config = self._build_run_config()
        state_snapshot = state.model_dump()

        try:
            async for update in self.app.astream(
                state.model_dump(),
                config=config,
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue

                for node_name, payload in update.items():
                    if not isinstance(payload, dict):
                        continue
                    self._merge_state_snapshot(state_snapshot, payload)
                    yield AgentStreamUpdate(
                        kind="status",
                        node=node_name,
                        payload=payload,
                        state=AgentState(**state_snapshot),
                    )

            final_state = await self._resolve_stream_final_state(config, state_snapshot)
            yield AgentStreamUpdate(kind="final", state=final_state)
        except Exception as exc:
            logger.error("Agent streaming failed: %s", exc, exc_info=True)
            self._merge_state_snapshot(
                state_snapshot,
                {
                    "error": str(exc),
                    "draft_response": f"I encountered an error: {str(exc)}. Please try again.",
                },
            )
            yield AgentStreamUpdate(kind="final", state=AgentState(**state_snapshot))

    def _build_initial_state(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]],
    ) -> AgentState:
        """Build the initial graph state for a chat turn."""
        state = AgentState(session_id=self.session_id)
        self._attach_files(state, files)
        state.messages.append(HumanMessage(content=message))
        return state

    def _attach_files(
        self,
        state: AgentState,
        files: Optional[list[tuple[str, bytes]]],
    ) -> None:
        """Prepare uploaded files and attach them to state."""
        if not files:
            return

        for filename, file_bytes in files:
            document = self.doc_processor.prepare_upload(file_bytes, filename)
            state.pending_documents.append(document)
        logger.info("Prepared %s documents for upload", len(files))

    def _build_run_config(self) -> dict[str, Any]:
        """Build the runtime configuration for LangGraph execution."""
        return {
            "configurable": {"thread_id": self.session_id},
            "recursion_limit": 25,
        }

    async def _resolve_stream_final_state(
        self,
        config: dict[str, Any],
        state_snapshot: dict[str, Any],
    ) -> AgentState:
        """Resolve the final graph state from the streamed run when possible."""
        get_state: object = getattr(self.app, "aget_state", None)
        if callable(get_state):
            try:
                state_obj = await get_state(config)  # type: ignore[reportGeneralTypeIssues]
                values = getattr(state_obj, "values", None)
                if isinstance(values, dict):
                    self._merge_state_snapshot(state_snapshot, values)
            except Exception:
                logger.debug("Compiled graph does not expose aget_state() for streamed runs.")
        return AgentState(**state_snapshot)

    def _merge_state_snapshot(self, snapshot: dict[str, Any], update: dict[str, Any]) -> None:
        """Merge a partial LangGraph state update into the current snapshot."""
        for key, value in update.items():
            if key == "messages" and isinstance(value, list):
                snapshot.setdefault("messages", [])
                snapshot["messages"].extend(value)
                continue
            if key in {"citations", "retrieved_chunks", "processed_documents", "pending_documents"}:
                snapshot[key] = list(value) if isinstance(value, list) else value
                continue
            snapshot[key] = value

    def _extract_response_text(self, final_state: AgentState) -> str:
        """Resolve the assistant response from the final graph state."""
        if final_state.draft_response:
            return str(final_state.draft_response)
        if final_state.messages:
            for message in reversed(final_state.messages):
                if isinstance(message, AIMessage):
                    return _message_content_to_text(message.content)
            last_message = final_state.messages[-1]
            return _message_content_to_text(last_message.content)
        return "I processed your request but have no response to provide."


# =============================================================================
# LangGraph Deployment Entry Point
# =============================================================================

# Process-wide compiled application; imported by langgraph.json as the entry point
compiled_app = get_compiled_app()


# =============================================================================
# CLI / Direct Execution
# =============================================================================


async def main() -> None:
    """Interactive CLI for testing the agent."""
    print("=" * 60)
    print("Academic Services and Student Records Chatbot")
    print("Type 'quit' to exit, 'upload <filepath>' to upload a document")
    print("=" * 60)

    agent = StudentRecordsAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.lower().startswith("upload "):
                filepath = user_input[7:].strip()
                path = Path(filepath)

                if not path.exists():
                    print(f"File not found: {filepath}")
                    continue

                with open(path, "rb") as f:
                    file_bytes = f.read()

                print(f"Uploading {path.name}...")
                response = await agent.chat(
                    "Process this document",
                    files=[(path.name, file_bytes)],
                )
            else:
                response = await agent.chat(user_input)

            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
