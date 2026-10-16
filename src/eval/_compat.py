"""Compatibility shim required before importing RAGAS.

RAGAS 0.4.3 imports ``langchain_community.chat_models.vertexai`` at module load, but
langchain-community 0.4.x removed that module (Vertex AI moved to a standalone
integration package). The name is only used for ``isinstance`` checks against the Vertex
clients, which the OpenRouter path never touches, so a stand-in class is enough.

Remove this module once RAGAS and langchain-community agree on the import again.
"""

from __future__ import annotations

import sys
import types
import warnings

_MISSING_MODULE = "langchain_community.chat_models.vertexai"


def install_ragas_compat() -> None:
    """Register a stand-in for the langchain-community module RAGAS expects."""
    if _MISSING_MODULE in sys.modules:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            __import__(_MISSING_MODULE)
            return
        except ModuleNotFoundError:
            pass

    module = types.ModuleType(_MISSING_MODULE)

    class ChatVertexAI:
        """Stand-in for the removed langchain-community Vertex chat model."""

    module.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[_MISSING_MODULE] = module

    import langchain_community.chat_models as chat_models

    chat_models.vertexai = module  # type: ignore[attr-defined]
