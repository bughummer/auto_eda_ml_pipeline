"""Profiling thresholds. Everything that decides a warning lives here, not in the rules."""

from pydantic import Field

from ml_engine.contracts.common import StrictModel


class ProfilingConfig(StrictModel):
    """Deterministic thresholds for EDA. Persisted into the EDA report for reproducibility."""

    max_rows: int | None = Field(
        default=None, description="Profile at most this many rows. None profiles everything."
    )
    top_values: int = Field(default=20, ge=1, description="Cap on returned categorical values.")
    sample_values: int = Field(default=5, ge=0)

    high_cardinality_absolute: int = Field(default=50, ge=2)
    high_cardinality_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    identifier_unique_ratio: float = Field(default=0.95, gt=0.0, le=1.0)
    identifier_min_rows: int = Field(default=20, ge=1)

    missing_warn_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    missing_high_fraction: float = Field(default=0.60, ge=0.0, le=1.0)

    text_mean_length: float = Field(default=40.0, gt=0.0)
    text_min_word_count: float = Field(default=4.0, gt=0.0)

    datetime_parse_fraction: float = Field(default=0.9, gt=0.0, le=1.0)
    discrete_max_unique: int = Field(default=20, ge=2)

    max_classes_for_classification: int = Field(default=50, ge=2)
    imbalance_warn_ratio: float = Field(default=3.0, gt=1.0)
    imbalance_high_ratio: float = Field(default=10.0, gt=1.0)
    small_dataset_rows: int = Field(default=100, ge=1)
    duplicate_rows_warn_fraction: float = Field(default=0.01, ge=0.0, le=1.0)

    def as_metadata(self) -> dict[str, int | float | bool | str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}
