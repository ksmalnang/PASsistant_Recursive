"""Intent routing node."""

from langchain_core.messages import HumanMessage

from src.services.contracts import LLMProvider
from src.services.intent import IntentClassifier
from src.utils.messages import conversation_transcript, message_text
from src.utils.state import AgentState

_HISTORY_LIMIT = 6


class RouterNode:
    """
    Intent classification and routing node.

    Analyzes user message to determine:
    - upload_document: User wants to upload a document
    - query_student: User is asking about a student record or transcript-style data
    - query_document: User is asking about academic services or document content
    - general_chat: General conversation
    """

    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        if classifier is not None:
            self._classifier = classifier
        elif llm_provider is not None:
            self._classifier = IntentClassifier(llm_provider=llm_provider)
        else:
            self._classifier = IntentClassifier()

    async def run(self, state: AgentState) -> dict:
        """
        Classify intent and update state routing flags.

        Args:
            state: Current agent state

        Returns:
            State updates dict
        """
        if not state.messages:
            return {"current_intent": "general_chat"}

        last_message = state.messages[-1]
        if not isinstance(last_message, HumanMessage):
            return {}
        return await self._classifier.classify(
            message_text(last_message.content),
            history=conversation_transcript(state.messages[:-1], limit=_HISTORY_LIMIT),
            session_id=state.session_id,
        )
