"""Deterministic EDA output schema.

Produced by ``ml_engine.profiling`` inside the profiling job, written to
``eda/eda.json`` and served verbatim by ``GET /api/v1/experiments/{id}/eda``.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import ProblemType, SemanticType, StrictModel
from ml_engine.contracts.warnings import AnalysisWarning

EDA_SCHEMA_VERSION = "1.0"


class DatasetSummary(StrictModel):
    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_percentage: float
    memory_usage_bytes: int = Field(description="Deep pandas memory usage of the loaded frame.")
    memory_usage_is_exact: bool = True
    file_format: str
    source_uri: str
    source_size_bytes: int | None = None
    sampled: bool = False
    sample_row_count: int | None = Field(
        default=None, description="Rows actually profiled when the dataset was sampled."
    )


class NumericStats(StrictModel):
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    q01: float | None
    q05: float | None
    q25: float | None
    q75: float | None
    q95: float | None
    q99: float | None
    zero_count: int
    zero_percentage: float
    negative_count: int
    skewness: float | None = None


class CategoryFrequency(StrictModel):
    value: str
    count: int
    percentage: float


class CategoricalStats(StrictModel):
    top_values: list[CategoryFrequency]
    truncated: bool = Field(
        default=False, description="True when more distinct values exist than were returned."
    )
    mean_length: float | None = None
    max_length: int | None = None


class DatetimeStats(StrictModel):
    min: datetime | None
    max: datetime | None
    range_days: float | None


class ColumnProfile(StrictModel):
    name: str
    semantic_type: SemanticType
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool
    is_high_cardinality: bool
    is_likely_id: bool
    is_text_like: bool = False
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None
    sample_values: list[str] = Field(default_factory=list)


class ClassDistributionEntry(StrictModel):
    label: str
    count: int
    percentage: float


class TargetAnalysis(StrictModel):
    column: str
    exists: bool
    inferred_problem_type: ProblemType | None
    dtype: str | None = None
    missing_count: int = 0
    missing_percentage: float = 0.0
    unique_count: int = 0
    class_distribution: list[ClassDistributionEntry] | None = None
    imbalance_ratio: float | None = Field(
        default=None,
        description="Majority class count divided by minority class count. None for regression.",
    )
    positive_class: str | None = None
    numeric: NumericStats | None = None


class EdaReport(StrictModel):
    schema_version: str = EDA_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    duration_seconds: float
    dataset: DatasetSummary
    target: TargetAnalysis
    columns: list[ColumnProfile]
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    profiling_config: dict[str, int | float | bool | str] = Field(default_factory=dict)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)
