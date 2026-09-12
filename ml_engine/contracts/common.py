"""Shared enums and primitives used across every ML Factory contract."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every contract: unknown fields are a bug, not a silent no-op."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False, validate_assignment=True)


class ProblemType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self is not ProblemType.REGRESSION


class RequestedProblemType(StrEnum):
    """What the user may ask for. ``auto`` is resolved during profiling."""

    AUTO = "auto"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class ExperimentStatus(StrEnum):
    CREATED = "CREATED"
    EDA_RUNNING = "EDA_RUNNING"
    EDA_COMPLETED = "EDA_COMPLETED"
    FEATURE_REVIEW = "FEATURE_REVIEW"
    READY_FOR_TRAINING = "READY_FOR_TRAINING"
    PREPARING = "PREPARING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.COMPLETED_WITH_WARNINGS,
        ExperimentStatus.FAILED,
    }
)


class ModelRunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WarningCategory(StrEnum):
    """Warning taxonomy. Stable — consumers filter and group on these values."""

    DATA_QUALITY = "data_quality"
    LEAKAGE = "leakage"
    ANOMALY = "anomaly"
    SCHEMA = "schema"
    TARGET = "target"
    FEATURE = "feature"
    PREPROCESSING = "preprocessing"
    METRIC = "metric"
    MODEL = "model"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class SemanticType(StrEnum):
    """Inferred meaning of a column, independent of its pandas dtype."""

    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    HIGH_CARDINALITY_CATEGORICAL = "high_cardinality_categorical"
    TEXT = "text"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ClassWeighting(StrEnum):
    AUTO = "auto"
    NONE = "none"


class LeakageRiskLevel(StrEnum):
    """Deterministic evidence strength. ``CONFIRMED_DUPLICATE`` needs structural proof."""

    NONE = "none"
    REQUIRES_REVIEW = "requires_review"
    POTENTIAL_LEAKAGE = "potential_leakage"
    CONFIRMED_DUPLICATE = "confirmed_duplicate"


LEAKAGE_RISK_ORDER: dict["LeakageRiskLevel", int] = {}


class RecommendedAction(StrEnum):
    """What the platform suggests. The user always decides."""

    KEEP = "keep"
    REVIEW = "review"
    CONSIDER_EXCLUDING = "consider_excluding"
    STRONGLY_CONSIDER_EXCLUDING = "strongly_consider_excluding"
    EXCLUDED_AUTOMATICALLY = "excluded_automatically"


LEAKAGE_RISK_ORDER.update(
    {
        LeakageRiskLevel.NONE: 0,
        LeakageRiskLevel.REQUIRES_REVIEW: 1,
        LeakageRiskLevel.POTENTIAL_LEAKAGE: 2,
        LeakageRiskLevel.CONFIRMED_DUPLICATE: 3,
    }
)

ACTION_ORDER: dict[RecommendedAction, int] = {
    RecommendedAction.KEEP: 0,
    RecommendedAction.REVIEW: 1,
    RecommendedAction.CONSIDER_EXCLUDING: 2,
    RecommendedAction.STRONGLY_CONSIDER_EXCLUDING: 3,
    RecommendedAction.EXCLUDED_AUTOMATICALLY: 4,
}
