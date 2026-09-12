"""Cross-model comparison and the experiment-level summary artifact."""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import (
    ExperimentStatus,
    MetricDirection,
    ModelRunStatus,
    ProblemType,
    StrictModel,
)
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.model import FeatureImportance
from ml_engine.contracts.warnings import AnalysisWarning

COMPARISON_SCHEMA_VERSION = "1.0"
SUMMARY_SCHEMA_VERSION = "1.0"


class ModelComparisonEntry(StrictModel):
    model_name: str
    display_name: str
    status: ModelRunStatus
    primary_score: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    training_duration_seconds: float | None = None
    feature_count: int | None = None
    rank: int | None = None
    warning_count: int = 0
    failure_message: str | None = None


class ComparisonReport(StrictModel):
    schema_version: str = COMPARISON_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    problem_type: ProblemType
    primary_metric: str
    direction: MetricDirection
    best_model: str | None = None
    best_score: float | None = None
    models: list[ModelComparisonEntry] = Field(default_factory=list)
    succeeded_count: int = 0
    failed_count: int = 0
    warnings: list[AnalysisWarning] = Field(default_factory=list)


class ExperimentSummary(StrictModel):
    """Everything a report or the reasoning layer needs, in one artifact."""

    schema_version: str = SUMMARY_SCHEMA_VERSION
    experiment_id: str
    name: str
    generated_at: datetime
    status: ExperimentStatus
    config: ExperimentConfig
    dataset_row_count: int | None = None
    dataset_column_count: int | None = None
    selected_feature_count: int = 0
    excluded_feature_count: int = 0
    comparison: ComparisonReport | None = None
    best_model_importance: FeatureImportance | None = None
    eda_warnings: list[AnalysisWarning] = Field(default_factory=list)
    leakage_warning_count: int = 0
    warnings: list[AnalysisWarning] = Field(default_factory=list)
