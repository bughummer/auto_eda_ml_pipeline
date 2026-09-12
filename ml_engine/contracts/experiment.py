"""Experiment records, stored in the artifact bucket beside everything else.

There is no separate state database. An experiment is three small JSON objects under its own
artifact prefix, each with exactly one writer, so no component ever has to read-modify-write
a shared row:

``experiment.json``        the control plane, once, at creation — immutable facts
``state/control.json``     the control plane, on user-driven transitions
``state/workflow.json``    Step Functions and the evaluation job, on execution-driven ones

:class:`ExperimentRecord` is the composed view the API returns. Per-model progress is not
stored at all: it is derived from the model artifacts, which are the source of truth.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import ExperimentStatus, RequestedProblemType, StrictModel
from ml_engine.contracts.config import DatasetReference


class ExperimentDefinition(StrictModel):
    """``experiment.json`` — written once, never updated."""

    experiment_id: str
    name: str
    created_by: str | None = None
    created_at: datetime
    dataset: DatasetReference
    target_column: str
    requested_problem_type: RequestedProblemType = RequestedProblemType.AUTO
    artifact_prefix: str


class ControlPlaneState(StrictModel):
    """``state/control.json`` — the transitions a user drives through the API."""

    experiment_id: str
    updated_at: datetime
    status: ExperimentStatus = ExperimentStatus.CREATED
    current_stage: str = "created"
    status_updated_at: datetime | None = Field(
        default=None,
        description=(
            "When the control plane last set the status. Recording an execution ARN or a "
            "model list must not take the status back from the workflow, so ownership of the "
            "status block is decided by this, not by the document's own write time."
        ),
    )
    eda_execution_arn: str | None = None
    training_execution_arn: str | None = None
    requested_models: list[str] = Field(default_factory=list)
    primary_metric: str | None = None
    deleted: bool = False


class WorkflowState(StrictModel):
    """``state/workflow.json`` — what Step Functions and the evaluation job report."""

    experiment_id: str
    updated_at: datetime
    status: ExperimentStatus
    current_stage: str = ""
    best_model: str | None = None
    best_score: float | None = None
    primary_metric: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None


class ExperimentRecord(StrictModel):
    """The composed view returned by the API. Never stored as one object."""

    experiment_id: str
    name: str
    created_by: str | None = None
    created_at: datetime
    updated_at: datetime
    dataset: DatasetReference
    target_column: str
    requested_problem_type: RequestedProblemType = RequestedProblemType.AUTO
    status: ExperimentStatus = ExperimentStatus.CREATED
    current_stage: str = "created"
    artifact_prefix: str
    eda_execution_arn: str | None = None
    training_execution_arn: str | None = None
    requested_models: list[str] = Field(default_factory=list)
    best_model: str | None = None
    best_score: float | None = None
    primary_metric: str | None = None
    report_uri: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    deleted: bool = False

    @classmethod
    def compose(
        cls,
        definition: ExperimentDefinition,
        control: ControlPlaneState | None,
        workflow: WorkflowState | None,
    ) -> "ExperimentRecord":
        """Merge the three documents.

        The status block (status, stage, failure) comes from whichever side wrote most
        recently, so a user restarting training is not overwritten by a stale workflow result
        and a finished workflow is not hidden by an older control-plane write.
        """
        control = control or ControlPlaneState(
            experiment_id=definition.experiment_id, updated_at=definition.created_at
        )
        # Falling back to the document's write time would let any control-plane write
        # reclaim the status, which is exactly what this field exists to prevent.
        control_claimed_at = control.status_updated_at or definition.created_at
        workflow_is_current = workflow is not None and workflow.updated_at >= control_claimed_at
        status_source = workflow if workflow_is_current else control

        return cls(
            experiment_id=definition.experiment_id,
            name=definition.name,
            created_by=definition.created_by,
            created_at=definition.created_at,
            updated_at=max(
                control.updated_at,
                workflow.updated_at if workflow else control.updated_at,
            ),
            dataset=definition.dataset,
            target_column=definition.target_column,
            requested_problem_type=definition.requested_problem_type,
            status=status_source.status,
            current_stage=status_source.current_stage or control.current_stage,
            artifact_prefix=definition.artifact_prefix,
            eda_execution_arn=control.eda_execution_arn,
            training_execution_arn=control.training_execution_arn,
            requested_models=control.requested_models,
            best_model=workflow.best_model if workflow else None,
            best_score=workflow.best_score if workflow else None,
            primary_metric=(workflow.primary_metric if workflow else None)
            or control.primary_metric,
            failure_code=workflow.failure_code if workflow_is_current else None,
            failure_message=workflow.failure_message if workflow_is_current else None,
            deleted=control.deleted,
        )
