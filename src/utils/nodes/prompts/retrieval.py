"""Retrieval node prompts."""

QUERY_REWRITE_PROMPT = """Rewrite the user's latest question into a concise, keyword-rich
search query for academic policy documents.

Use the conversation only to resolve pronouns and ellipsis (for example "itu",
"yang tadi", "the second one"). If the latest question is self-contained or
starts a new topic, ignore the conversation entirely and rewrite the question
on its own.

Conversation so far:
{history}

Latest user question: {question}

Search query (Indonesian, concise, keyword-rich):"""
