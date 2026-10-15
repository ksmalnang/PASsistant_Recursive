"""Evaluation toolkit: RAGAS dataset generation, scoring, and comparison.

Importing this package installs the compatibility shim RAGAS needs on
langchain-community 0.4.x before any submodule imports RAGAS itself.
"""

from __future__ import annotations

from src.eval._compat import install_ragas_compat

install_ragas_compat()

__all__ = ["install_ragas_compat"]
