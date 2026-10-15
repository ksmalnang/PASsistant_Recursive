"""Regression guard for the RAGAS import shim."""

from __future__ import annotations

import importlib


def test_ragas_imports_once_the_eval_package_is_loaded() -> None:
    importlib.import_module("src.eval")

    ragas = importlib.import_module("ragas")

    assert ragas.__version__
    assert importlib.import_module("ragas.llms").BaseRagasLLM is not None
