"""The experiment record: small, queryable control-plane state (DynamoDB)."""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import ExperimentStatus, RequestedProblemType, StrictModel
from ml_engine.contracts.config import DatasetReference


class ExperimentRecord(StrictModel):
    """Metadata and state only. Large ML artifacts always live in the object store."""

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
    best_model: str | None = None
    best_score: float | None = None
    primary_metric: str | None = None
    report_uri: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    model_statuses: dict[str, str] = Field(
        default_factory=dict, description="model name -> ModelRunStatus, mirrored for cheap polling."
    )
    deleted: bool = False
