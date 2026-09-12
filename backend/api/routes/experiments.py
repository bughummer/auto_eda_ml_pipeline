"""Experiment endpoints: creation, artifacts, feature review and training."""

from typing import Annotated

from fastapi import APIRouter, Query, status

from backend.api.dependencies import CurrentUserDep, ExperimentServiceDep
from backend.schemas.experiments import (
    CreateExperimentRequest,
    CreateExperimentResponse,
    ExperimentListResponse,
    FeatureReviewResponse,
    StartTrainingResponse,
    TrainingConfigRequest,
    TrainingConfigResponse,
    TrainingStatusResponse,
    UpdateFeatureSelectionRequest,
)
from ml_engine.contracts.comparison import ComparisonReport, ExperimentSummary
from ml_engine.contracts.config import FeatureSelection
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.experiment import ExperimentRecord
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.model import ModelMetadata

router = APIRouter(prefix="/experiments", tags=["experiments"])

#: Optional artifact bucket to read from. Omitted means the platform's own bucket; supplied
#: means "show me what is in that approved bucket", which is how previous experiments from
#: another environment are browsed. Only configured roots are accepted.
ArtifactRoot = Annotated[
    str | None,
    Query(
        alias="root",
        description="Artifact bucket URI to read from. Defaults to this platform's bucket.",
        examples=["s3://my-ml-factory-artifacts"],
    ),
]


@router.post("", response_model=CreateExperimentResponse, status_code=status.HTTP_202_ACCEPTED)
def create_experiment(
    request: CreateExperimentRequest, service: ExperimentServiceDep, user: CurrentUserDep
) -> CreateExperimentResponse:
    """Create an experiment and start EDA. Returns immediately — AWS work runs asynchronously."""
    record = service.create(request, user)
    return CreateExperimentResponse(
        experiment_id=record.experiment_id,
        status=record.status,
        artifact_prefix=record.artifact_prefix,
    )


@router.get("", response_model=ExperimentListResponse)
def list_experiments(
    service: ExperimentServiceDep,
    root: ArtifactRoot = None,
    limit: int = Query(default=50, ge=1, le=500),
    mine: bool = Query(default=False, description="Restrict to experiments you created."),
    user: CurrentUserDep = None,
) -> ExperimentListResponse:
    """Experiments found under an artifact bucket, newest first."""
    records = service.list(limit=limit, created_by=user if mine else None, root=root)
    return ExperimentListResponse(
        experiments=records,
        count=len(records),
        root=root or service.artifact_roots()[0] if service.artifact_roots() else "",
        available_roots=service.artifact_roots(),
    )


@router.get("/{experiment_id}", response_model=ExperimentRecord)
def get_experiment(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> ExperimentRecord:
    return service.get(experiment_id, root)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(experiment_id: str, service: ExperimentServiceDep) -> None:
    """Soft delete: the record is hidden, the artifacts remain for audit."""
    service.delete(experiment_id)


@router.get("/{experiment_id}/eda", response_model=EdaReport)
def get_eda(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> EdaReport:
    return service.eda(experiment_id, root)


@router.get("/{experiment_id}/leakage", response_model=LeakageReport)
def get_leakage(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> LeakageReport:
    return service.leakage(experiment_id, root)


@router.get("/{experiment_id}/features", response_model=FeatureReviewResponse)
def get_features(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> FeatureReviewResponse:
    return service.feature_review(experiment_id, root)


@router.put("/{experiment_id}/features", response_model=FeatureSelection)
def put_features(
    experiment_id: str,
    request: UpdateFeatureSelectionRequest,
    service: ExperimentServiceDep,
    user: CurrentUserDep,
) -> FeatureSelection:
    return service.save_features(experiment_id, request, user)


@router.get("/{experiment_id}/training-config", response_model=TrainingConfigResponse)
def get_training_config(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> TrainingConfigResponse:
    return service.training_config(experiment_id, root)


@router.post(
    "/{experiment_id}/training",
    response_model=StartTrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_training(
    experiment_id: str,
    request: TrainingConfigRequest,
    service: ExperimentServiceDep,
    user: CurrentUserDep,
) -> StartTrainingResponse:
    record = service.start_training(experiment_id, request, user)
    return StartTrainingResponse(
        experiment_id=record.experiment_id,
        status=record.status,
        models=sorted(record.model_statuses),
    )


@router.get("/{experiment_id}/training-status", response_model=TrainingStatusResponse)
def get_training_status(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> TrainingStatusResponse:
    return service.training_status(experiment_id, root)


@router.get("/{experiment_id}/comparison", response_model=ComparisonReport)
def get_comparison(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> ComparisonReport:
    return service.comparison(experiment_id, root)


@router.get("/{experiment_id}/models/{model_name}", response_model=ModelMetadata)
def get_model(experiment_id: str, model_name: str, service: ExperimentServiceDep) -> ModelMetadata:
    return service.model_metadata(experiment_id, model_name)


@router.get("/{experiment_id}/summary", response_model=ExperimentSummary)
def get_summary(
    experiment_id: str, service: ExperimentServiceDep, root: ArtifactRoot = None
) -> ExperimentSummary:
    return service.summary(experiment_id, root)
