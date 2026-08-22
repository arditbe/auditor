"""Firestore-backed run store.

Runs are written as one document per run. The Firestore client is synchronous
and blocking, so every call is pushed to a worker thread -- doing otherwise
would stall the event loop mid-run and visibly stutter the live dashboard.
"""
from __future__ import annotations

import asyncio
import logging

from google.cloud import firestore

from ..config import settings
from ..models.schemas import ValidationRun
from .base import RunStore

log = logging.getLogger(__name__)


class FirestoreRunStore(RunStore):
    def __init__(self, collection: str | None = None) -> None:
        kwargs = {}
        if settings.google_cloud_project:
            kwargs["project"] = settings.google_cloud_project
        self._db = firestore.Client(**kwargs)
        self._collection = collection or settings.firestore_collection

    def _doc(self, run_id: str):
        return self._db.collection(self._collection).document(run_id)

    async def save(self, run: ValidationRun) -> None:
        payload = run.model_dump(mode="json")
        await asyncio.to_thread(self._doc(run.run_id).set, payload)

    async def get(self, run_id: str) -> ValidationRun | None:
        snapshot = await asyncio.to_thread(self._doc(run_id).get)
        if not snapshot.exists:
            return None
        return ValidationRun.model_validate(snapshot.to_dict())

    async def list_recent(self, limit: int = 20) -> list[ValidationRun]:
        def _query():
            return list(
                self._db.collection(self._collection)
                .order_by("created_at_ms", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream()
            )

        docs = await asyncio.to_thread(_query)
        runs: list[ValidationRun] = []
        for doc in docs:
            try:
                runs.append(ValidationRun.model_validate(doc.to_dict()))
            except Exception as exc:  # noqa: BLE001
                # A schema change should not take down the history endpoint.
                log.warning("skipping unreadable run doc %s: %s", doc.id, exc)
        return runs

    async def delete(self, run_id: str) -> None:
        await asyncio.to_thread(self._doc(run_id).delete)
