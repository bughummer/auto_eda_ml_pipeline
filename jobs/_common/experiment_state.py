"""Terminal experiment state, written by the component that computes it.

Step Functions owns sequencing and every intermediate transition. The final state, however,
depends on the comparison the evaluation job produces — whether every model succeeded, and
whether the run carries warnings — so the evaluation job writes it rather than re-deriving
that judgement in the state machine. ``ml_engine.reporting.final_status`` remains the single
place the decision is made.

The state lands in the experiment's own artifact prefix, as ``state/workflow.json``: the same
object Step Functions writes at each stage, and the same object the control plane reads.
"""

import logging
from datetime import UTC, datetime
from typing import Protocol

from ml_engine.contracts.comparison import ComparisonReport
from ml_engine.contracts.experiment import WorkflowState
from ml_engine.io import ExperimentLayout, ObjectStore, write_model
from ml_engine.reporting import final_status

LOGGER = logging.getLogger("ml_factory.jobs.state")


class ExperimentStateWriter(Protocol):
    def record_completion(self, experiment_id: str, comparison: ComparisonReport) -> None: ...


class ArtifactExperimentStateWriter:
    """Writes the terminal status, best model and score next to the artifacts."""

    def __init__(self, store: ObjectStore, layout: ExperimentLayout) -> None:
        self._store = store
        self._layout = layout

    def record_completion(self, experiment_id: str, comparison: ComparisonReport) -> None:
        status = final_status(comparison)
        write_model(
            self._store,
            self._layout.workflow_state,
            WorkflowState(
                experiment_id=experiment_id,
                updated_at=datetime.now(UTC),
                status=status,
                current_stage="completed",
                best_model=comparison.best_model,
                best_score=comparison.best_score,
                primary_metric=comparison.primary_metric,
            ),
        )
        LOGGER.info("Recorded terminal state %s for %s", status.value, experiment_id)


def build_state_writer(
    store: ObjectStore, layout: ExperimentLayout, *, enabled: bool = True
) -> ExperimentStateWriter | None:
    """A dry run (``--no-record-state``) skips the state write and only builds the reports."""
    return ArtifactExperimentStateWriter(store, layout) if enabled else None
