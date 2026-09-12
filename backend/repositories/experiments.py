"""Experiment record persistence.

Small, queryable control-plane state only — status, ownership, dataset identity and the
headline result. Large ML artifacts always live in the object store.
"""

import threading
from datetime import UTC, datetime
from typing import Any, Protocol

from backend.errors import NotFoundError, UpstreamError
from ml_engine.contracts.experiment import ExperimentRecord


class ExperimentRepository(Protocol):
    def create(self, record: ExperimentRecord) -> ExperimentRecord: ...

    def get(self, experiment_id: str) -> ExperimentRecord: ...

    def list(self, limit: int = 100, created_by: str | None = None) -> list[ExperimentRecord]: ...

    def update(self, experiment_id: str, **changes: Any) -> ExperimentRecord: ...

    def delete(self, experiment_id: str) -> None: ...


class InMemoryExperimentRepository:
    """Used in local mode and in tests. Thread-safe because the local runner is threaded."""

    def __init__(self) -> None:
        self._records: dict[str, ExperimentRecord] = {}
        self._lock = threading.RLock()

    def create(self, record: ExperimentRecord) -> ExperimentRecord:
        with self._lock:
            self._records[record.experiment_id] = record
            return record.model_copy(deep=True)

    def get(self, experiment_id: str) -> ExperimentRecord:
        with self._lock:
            record = self._records.get(experiment_id)
            if record is None or record.deleted:
                raise NotFoundError(f"Experiment '{experiment_id}' does not exist.")
            return record.model_copy(deep=True)

    def list(self, limit: int = 100, created_by: str | None = None) -> list[ExperimentRecord]:
        with self._lock:
            records = [r for r in self._records.values() if not r.deleted]
        if created_by:
            records = [r for r in records if r.created_by == created_by]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return [r.model_copy(deep=True) for r in records[:limit]]

    def update(self, experiment_id: str, **changes: Any) -> ExperimentRecord:
        with self._lock:
            record = self._records.get(experiment_id)
            if record is None or record.deleted:
                raise NotFoundError(f"Experiment '{experiment_id}' does not exist.")
            updated = record.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
            self._records[experiment_id] = updated
            return updated.model_copy(deep=True)

    def delete(self, experiment_id: str) -> None:
        self.update(experiment_id, deleted=True)


class DynamoExperimentRepository:
    """DynamoDB-backed records. One item per experiment, keyed by ``experiment_id``."""

    def __init__(self, table: Any) -> None:
        self._table = table

    def create(self, record: ExperimentRecord) -> ExperimentRecord:
        self._put(record)
        return record

    def get(self, experiment_id: str) -> ExperimentRecord:
        try:
            response = self._table.get_item(Key={"experiment_id": experiment_id})
        except Exception as exc:
            raise UpstreamError(
                f"Could not read experiment '{experiment_id}' from DynamoDB."
            ) from exc
        item = response.get("Item")
        if not item or item.get("deleted"):
            raise NotFoundError(f"Experiment '{experiment_id}' does not exist.")
        return _from_item(item)

    def list(self, limit: int = 100, created_by: str | None = None) -> list[ExperimentRecord]:
        try:
            response = self._table.scan(Limit=max(limit * 2, limit))
        except Exception as exc:
            raise UpstreamError("Could not list experiments from DynamoDB.") from exc
        records = [
            _from_item(item) for item in response.get("Items", []) if not item.get("deleted")
        ]
        if created_by:
            records = [r for r in records if r.created_by == created_by]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def update(self, experiment_id: str, **changes: Any) -> ExperimentRecord:
        record = self.get(experiment_id)
        updated = record.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        self._put(updated)
        return updated

    def delete(self, experiment_id: str) -> None:
        self.update(experiment_id, deleted=True)

    def _put(self, record: ExperimentRecord) -> None:
        try:
            self._table.put_item(Item=_to_item(record))
        except Exception as exc:
            raise UpstreamError(
                f"Could not persist experiment '{record.experiment_id}' to DynamoDB."
            ) from exc


def _to_item(record: ExperimentRecord) -> dict[str, Any]:
    """DynamoDB rejects empty strings and floats; serialize through JSON mode and clean up."""
    item = record.model_dump(mode="json")
    return {key: value for key, value in item.items() if value not in ("", None)}


def _from_item(item: dict[str, Any]) -> ExperimentRecord:
    from decimal import Decimal

    cleaned = {
        key: (float(value) if isinstance(value, Decimal) else value) for key, value in item.items()
    }
    return ExperimentRecord.model_validate(cleaned)
