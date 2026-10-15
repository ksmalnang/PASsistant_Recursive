"""Workspace-local `tmp_path` for the evaluation tests.

pytest's default `tmp_path` lives under the system temp directory, which is not
writable in every environment this suite runs in. These tests only need throwaway
JSONL/CSV files, so keep them inside the repository and delete them with plain paths.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_SCRATCH_ROOT = Path(".tmp_eval_tests")


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    path = _SCRATCH_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    _remove(path)


def _remove(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove(child)
        else:
            child.unlink()
    path.rmdir()
