"""Per-model artifacts: metadata, feature importance and failure records."""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import ModelRunStatus, ProblemType, StrictModel
from ml_engine.contracts.metrics import MetricSet
from ml_engine.contracts.warnings import AnalysisWarning

MODEL_METADATA_SCHEMA_VERSION = "1.0"


class FeatureImportanceEntry(StrictModel):
    feature: str
    importance: float = Field(description="Normalized to sum to 1 across features.")
    raw_value: float
    rank: int


class FeatureImportance(StrictModel):
    method: str = Field(description="e.g. 'tree_gain', 'coefficient_magnitude'.")
    is_signed: bool = Field(
        default=False, description="True when raw values carry direction (linear coefficients)."
    )
    entries: list[FeatureImportanceEntry] = Field(default_factory=list)
    note: str | None = None


class PreprocessingMetadata(StrictModel):
    """What the fitted pipeline actually did. Written by the preparation job."""

    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    derived_datetime_features: list[str] = Field(default_factory=list)
    dropped_columns: dict[str, str] = Field(
        default_factory=dict, description="column -> reason it was dropped"
    )
    output_feature_names: list[str] = Field(default_factory=list)
    output_feature_count: int = 0
    strategy: str = Field(default="dense_numeric", description="dense_numeric | native_categorical")
    fitted_on: str = Field(default="train", description="Learned transforms fit on the train fold only.")
    config_digest: str | None = None


class ModelArtifacts(StrictModel):
    model_uri: str | None = None
    metadata_uri: str | None = None
    preprocessor_uri: str | None = None


class ModelMetadata(StrictModel):
    schema_version: str = MODEL_METADATA_SCHEMA_VERSION
    experiment_id: str
    model_name: str
    display_name: str
    status: ModelRunStatus
    problem_type: ProblemType
    library: str
    library_version: str
    hyperparameters: dict[str, float | int | str | bool | None | list[float]] = Field(
        default_factory=dict
    )
    features_used: list[str] = Field(default_factory=list)
    feature_count: int = 0
    training_row_count: int = 0
    validation_row_count: int = 0
    training_started_at: datetime | None = None
    training_completed_at: datetime | None = None
    training_duration_seconds: float | None = None
    primary_metric: str | None = None
    primary_score: float | None = None
    metrics: MetricSet = Field(default_factory=MetricSet)
    train_metrics: MetricSet | None = Field(
        default=None, description="Training-fold metrics, for overfitting inspection only."
    )
    feature_importance: FeatureImportance | None = None
    preprocessing: PreprocessingMetadata | None = None
    artifacts: ModelArtifacts = Field(default_factory=ModelArtifacts)
    package_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[AnalysisWarning] = Field(default_factory=list)


class ModelFailure(StrictModel):
    """Written to ``models/<name>/failure.json``; surfaced to the user without a stack trace."""

    experiment_id: str
    model_name: str
    status: ModelRunStatus = ModelRunStatus.FAILED
    error_code: str
    message: str = Field(description="User-safe message. No stack traces.")
    failed_at: datetime
    details: dict[str, str] = Field(default_factory=dict)
