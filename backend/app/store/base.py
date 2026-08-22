"""Store interface."""
from __future__ import annotations

import abc

from ..models.schemas import ValidationRun


class RunStore(abc.ABC):
    @abc.abstractmethod
    async def save(self, run: ValidationRun) -> None:
        """Upsert the full run document."""

    @abc.abstractmethod
    async def get(self, run_id: str) -> ValidationRun | None:
        ...

    @abc.abstractmethod
    async def list_recent(self, limit: int = 20) -> list[ValidationRun]:
        """Most recently created runs first."""

    @abc.abstractmethod
    async def delete(self, run_id: str) -> None:
        ...
