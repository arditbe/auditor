"""Run persistence.

Firestore is the real backend (and satisfies the Google Cloud requirement); the
in-memory store is the fallback so a missing credential degrades the demo to
"no history after restart" rather than "nothing works".
"""
from __future__ import annotations

import logging

from ..config import settings
from .base import RunStore
from .memory import MemoryRunStore

log = logging.getLogger(__name__)

_store: RunStore | None = None


def get_store() -> RunStore:
    """Process-wide store, chosen once from config."""
    global _store
    if _store is not None:
        return _store

    if settings.store_backend == "firestore":
        try:
            from .firestore import FirestoreRunStore

            _store = FirestoreRunStore()
            log.info("run store: firestore (collection=%s)",
                     settings.firestore_collection)
        except Exception as exc:  # noqa: BLE001 - fallback is the whole point
            log.warning(
                "firestore unavailable (%s); falling back to in-memory store",
                exc,
            )
            _store = MemoryRunStore()
    else:
        _store = MemoryRunStore()
        log.info("run store: in-memory")

    return _store


def reset_store() -> None:
    global _store
    _store = None


__all__ = ["RunStore", "MemoryRunStore", "get_store", "reset_store"]
