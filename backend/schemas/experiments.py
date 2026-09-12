"""API request and response models.

Artifact-shaped payloads reuse ``ml_engine.contracts`` verbatim; only genuinely API-specific
shapes live here.
"""

from datetime import datetime

from pydantic import Field, field_validator

from ml_engine.contracts.common import (
    ClassWeighting,
    ExperimentStatus,
    LeakageRiskLevel,
    ModelRunStatus,
    ProblemType,
    RecommendedAction,
    RequestedProblemType,
    SemanticType,
    Severity,
    StrictModel,
)
from ml_engine.contracts.config import PreprocessingConfig, SplitConfig
from ml_engine.contracts.experiment import ExperimentRecord
from ml_engine.contracts.warnings import AnalysisWarning


class CreateExperimentRequest(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    dataset_uri: str = Field(description="S3 URI of an approved dataset (s3://bucket/key).")
    target_column: str = Field(min_length=1, max_length=200)
    problem_type: RequestedProblemType = RequestedProblemType.AUTO
    file_format: str | None = Field(default=None, description="Override the inferred format.")
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name", "target_column")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned


class CreateExperimentResponse(StrictModel):
    experiment_id: str
    status: ExperimentStatus
    artifact_prefix: str


class ExperimentListResponse(StrictModel):
    experiments: list[ExperimentRecord]
    count: int


class FeatureReviewItem(StrictModel):
    """One row of the feature review table: profile, risk and the platform's recommendation."""

    feature: str
    selected: bool
    semantic_type: SemanticType
    dtype: str
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool
    is_high_cardinality: bool
    is_likely_id: bool
    leakage_risk: LeakageRiskLevel
    max_severity: Severity | None = None
    recommended_action: RecommendedAction
    reasons: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    documentation: str | None = None
    available_at_prediction_time: bool | None = None


class FeatureReviewResponse(StrictModel):
    experiment_id: str
    target_column: str
    problem_type: ProblemType | None
    features: list[FeatureReviewItem]
    selected_count: int
    excluded_count: int
    decided: bool = Field(
        default=False, description="True once the user has saved a feature selection."
    )


class UpdateFeatureSelectionRequest(StrictModel):
    selected_features: list[str] = Field(min_length=1)
    exclusion_reasons: dict[str, str] = Field(default_factory=dict)


class TrainingConfigRequest(StrictModel):
    """User-facing training configuration. Validated against the dataset before it is stored."""

    problem_type: RequestedProblemType = RequestedProblemType.AUTO
    primary_metric: str | None = None
    models: list[str] = Field(default_factory=list, description="Empty means every supported model.")
    model_parameters: dict[str, dict[str, float | int | str | bool | None]] = Field(
        default_factory=dict
    )
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = 42
    class_weighting: ClassWeighting = ClassWeighting.AUTO
    preprocessing: PreprocessingConfig | None = None

    def to_split_config(self) -> SplitConfig:
        return SplitConfig(
            validation_fraction=self.validation_fraction, random_seed=self.random_seed
        )


class TrainingConfigResponse(StrictModel):
    experiment_id: str
    problem_type: ProblemType
    requested_problem_type: RequestedProblemType
    primary_metric: str
    available_metrics: list[str]
    available_models: list[str]
    selected_models: list[str]
    validation_fraction: float
    random_seed: int
    class_weighting: ClassWeighting
    preprocessing: PreprocessingConfig
    selected_feature_count: int


class StartTrainingResponse(StrictModel):
    experiment_id: str
    status: ExperimentStatus
    models: list[str]


class ModelRunState(StrictModel):
    model_name: str
    display_name: str
    status: ModelRunStatus
    primary_score: float | None = None
    failure_message: str | None = None


class TrainingStatusResponse(StrictModel):
    experiment_id: str
    status: ExperimentStatus
    current_stage: str
    models: list[ModelRunState]
    best_model: str | None = None
    best_score: float | None = None
    primary_metric: str | None = None
    failure_message: str | None = None
    updated_at: datetime


class HealthResponse(StrictModel):
    status: str
    mode: str
    orchestrator: str
    artifact_root: str
    bedrock_enabled: bool
    configuration_problems: list[str] = Field(default_factory=list)
    available_models: list[str] = Field(default_factory=list)
