"""In-process locks that serialize reindexing and deletion per document.

These locks only coordinate callers inside one process. A multi-worker
deployment would need a distributed lock (for example a Redis SETNX lease),
because nothing here is visible to another worker.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class DocumentLockedError(RuntimeError):
    """Raised when another reindex or delete currently owns the document."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"document {document_id} is currently being modified")


class DocumentLockRegistry:
    """Non-blocking per-document locks shared by reindex and delete paths."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    @contextmanager
    def hold(self, document_id: str) -> Iterator[bool]:
        """Yield True when the lock was acquired; always release on exit."""
        lock = self._lock_for(document_id)
        acquired = lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()

    def _lock_for(self, document_id: str) -> threading.Lock:
        with self._guard:
            # Lock objects are kept for the process lifetime: the number of
            # documents is bounded and removing them would race with a waiter.
            return self._locks.setdefault(document_id, threading.Lock())


document_locks = DocumentLockRegistry()
