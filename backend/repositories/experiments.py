"""Experiment records, stored in the artifact bucket.

One store, one place to look: an experiment's state sits under the same prefix as its EDA,
its models and its comparison. Listing experiments is an S3 prefix listing; reading one is
two small GETs.

Each of the three documents has a single writer, so no component performs a read-modify-write
and two writers can never lose each other's fields. See ``ml_engine.contracts.experiment``.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Protocol

from backend.errors import NotFoundError
from ml_engine.contracts.experiment import (
    ControlPlaneState,
    ExperimentDefinition,
    ExperimentRecord,
    WorkflowState,
)
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists, write_model

LOGGER = logging.getLogger("ml_factory.repositories")

#: Records are read in parallel; a listing of hundreds should not take hundreds of round trips.
LIST_CONCURRENCY = 16

#: Fields owned by the control plane. Anything else belongs to the workflow.
_CONTROL_FIELDS = frozenset(ControlPlaneState.model_fields) - {
    "experiment_id",
    "updated_at",
    "status_updated_at",
}


class ExperimentRepository(Protocol):
    def create(self, definition: ExperimentDefinition) -> ExperimentRecord: ...

    def get(self, experiment_id: str, root: str | None = None) -> ExperimentRecord: ...

    def list(
        self, limit: int = 100, created_by: str | None = None, root: str | None = None
    ) -> list[ExperimentRecord]: ...

    def update(self, experiment_id: str, **changes: Any) -> ExperimentRecord: ...

    def delete(self, experiment_id: str) -> None: ...


class ObjectStoreExperimentRepository:
    """Reads and writes experiment records through the object store.

    ``root`` is the artifact root new experiments are created under. Reads accept a different
    root, which is what lets the UI open an experiment from any approved artifact bucket.
    """

    def __init__(self, store: ObjectStore, root: str) -> None:
        self._store = store
        self._root = root.rstrip("/")

    @property
    def root(self) -> str:
        return self._root

    # --- writes -----------------------------------------------------------
    def create(self, definition: ExperimentDefinition) -> ExperimentRecord:
        layout = ExperimentLayout(base=definition.artifact_prefix)
        if self._store.exists(layout.definition):
            raise ValueError(f"Experiment '{definition.experiment_id}' already exists")
        write_model(self._store, layout.definition, definition)
        control = ControlPlaneState(
            experiment_id=definition.experiment_id,
            updated_at=definition.created_at,
            status_updated_at=definition.created_at,
        )
        write_model(self._store, layout.control_state, control)
        return ExperimentRecord.compose(definition, control, None)

    def update(self, experiment_id: str, **changes: Any) -> ExperimentRecord:
        """Update the control-plane document only; workflow fields are not ours to write."""
        unknown = set(changes) - _CONTROL_FIELDS
        if unknown:
            raise ValueError(
                f"The control plane does not own {sorted(unknown)}; those fields are written "
                "by the workflow."
            )
        layout = self._layout(experiment_id)
        definition = self._definition(layout, experiment_id)
        current = read_model_if_exists(
            self._store, layout.control_state, ControlPlaneState
        ) or ControlPlaneState(experiment_id=experiment_id, updated_at=definition.created_at)
        now = datetime.now(UTC)
        # Claiming the status block is explicit: writing an execution ARN or a model list
        # must not take the status back from a workflow that has moved on.
        if "status" in changes:
            changes = {**changes, "status_updated_at": now}
        updated = current.model_copy(update={**changes, "updated_at": now})
        write_model(self._store, layout.control_state, updated)
        workflow = read_model_if_exists(self._store, layout.workflow_state, WorkflowState)
        return ExperimentRecord.compose(definition, updated, workflow)

    def delete(self, experiment_id: str) -> None:
        """Soft delete: the record is hidden, every artifact stays for audit."""
        self.update(experiment_id, deleted=True)

    # --- reads ------------------------------------------------------------
    def get(self, experiment_id: str, root: str | None = None) -> ExperimentRecord:
        layout = self._layout(experiment_id, root)
        record = self._read(layout)
        if record is None or record.deleted:
            raise NotFoundError(f"Experiment '{experiment_id}' does not exist.")
        return record

    def list(
        self, limit: int = 100, created_by: str | None = None, root: str | None = None
    ) -> list[ExperimentRecord]:
        prefixes = self._store.list_prefixes(
            ExperimentLayout.experiments_prefix(root or self._root)
        )
        if not prefixes:
            return []
        with ThreadPoolExecutor(max_workers=LIST_CONCURRENCY) as pool:
            records = list(
                pool.map(
                    lambda prefix: self._read(ExperimentLayout(base=prefix.rstrip("/"))), prefixes
                )
            )
        found = [record for record in records if record is not None and not record.deleted]
        if created_by:
            found = [record for record in found if record.created_by == created_by]
        found.sort(key=lambda record: record.created_at, reverse=True)
        return found[:limit]

    # --- helpers ----------------------------------------------------------
    def _read(self, layout: ExperimentLayout) -> ExperimentRecord | None:
        definition = read_model_if_exists(self._store, layout.definition, ExperimentDefinition)
        if definition is None:
            return None
        return ExperimentRecord.compose(
            definition,
            read_model_if_exists(self._store, layout.control_state, ControlPlaneState),
            read_model_if_exists(self._store, layout.workflow_state, WorkflowState),
        )

    def _definition(self, layout: ExperimentLayout, experiment_id: str) -> ExperimentDefinition:
        definition = read_model_if_exists(self._store, layout.definition, ExperimentDefinition)
        if definition is None:
            raise NotFoundError(f"Experiment '{experiment_id}' does not exist.")
        return definition

    def _layout(self, experiment_id: str, root: str | None = None) -> ExperimentLayout:
        return ExperimentLayout.for_experiment(root or self._root, experiment_id)
