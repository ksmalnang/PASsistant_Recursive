"""Plain-text rendering for conversation messages."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage


def message_text(content: object) -> str:
    """Return the text of a LangChain message content payload."""
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


def conversation_transcript(messages: Sequence[BaseMessage], *, limit: int) -> str:
    """Render the last messages as a User/Assistant transcript."""
    lines: list[str] = []
    for message in list(messages)[-limit:]:
        text = message_text(message.content)
        if text:
            role = "User" if isinstance(message, HumanMessage) else "Assistant"
            lines.append(f"{role}: {text}")
    return "\n".join(lines)
