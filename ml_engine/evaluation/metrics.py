"""Metric definitions and optimization directions.

Getting a direction wrong silently ranks models backwards, so directions are declared once,
here, and every consumer reads them from this table.
"""

from ml_engine.contracts.common import MetricDirection, ProblemType
from ml_engine.contracts.metrics import MetricDefinition

BINARY_METRICS = (
    MetricDefinition(
        name="roc_auc",
        display_name="ROC AUC",
        direction=MetricDirection.MAXIMIZE,
        description="Ranking quality across all thresholds; insensitive to class balance.",
    ),
    MetricDefinition(
        name="pr_auc",
        display_name="PR AUC",
        direction=MetricDirection.MAXIMIZE,
        description="Average precision. More informative than ROC AUC on rare positives.",
    ),
    MetricDefinition(
        name="accuracy",
        display_name="Accuracy",
        direction=MetricDirection.MAXIMIZE,
        description="Share of correct predictions. Misleading under class imbalance.",
    ),
    MetricDefinition(
        name="precision",
        display_name="Precision",
        direction=MetricDirection.MAXIMIZE,
        description="Share of predicted positives that are actually positive.",
    ),
    MetricDefinition(
        name="recall",
        display_name="Recall",
        direction=MetricDirection.MAXIMIZE,
        description="Share of actual positives that were found.",
    ),
    MetricDefinition(
        name="f1",
        display_name="F1",
        direction=MetricDirection.MAXIMIZE,
        description="Harmonic mean of precision and recall at the 0.5 threshold.",
    ),
    MetricDefinition(
        name="log_loss",
        display_name="Log Loss",
        direction=MetricDirection.MINIMIZE,
        description="Probability calibration quality. Lower is better.",
    ),
)

MULTICLASS_METRICS = (
    MetricDefinition(
        name="accuracy",
        display_name="Accuracy",
        direction=MetricDirection.MAXIMIZE,
        description="Share of correct predictions across all classes.",
    ),
    MetricDefinition(
        name="macro_f1",
        display_name="Macro F1",
        direction=MetricDirection.MAXIMIZE,
        description="Unweighted mean F1 across classes; treats rare classes as equally important.",
    ),
    MetricDefinition(
        name="weighted_f1",
        display_name="Weighted F1",
        direction=MetricDirection.MAXIMIZE,
        description="Support-weighted mean F1 across classes.",
    ),
    MetricDefinition(
        name="macro_precision",
        display_name="Macro Precision",
        direction=MetricDirection.MAXIMIZE,
        description="Unweighted mean precision across classes.",
    ),
    MetricDefinition(
        name="macro_recall",
        display_name="Macro Recall",
        direction=MetricDirection.MAXIMIZE,
        description="Unweighted mean recall across classes.",
    ),
    MetricDefinition(
        name="log_loss",
        display_name="Log Loss",
        direction=MetricDirection.MINIMIZE,
        description="Multiclass probability calibration quality. Lower is better.",
    ),
)

REGRESSION_METRICS = (
    MetricDefinition(
        name="rmse",
        display_name="RMSE",
        direction=MetricDirection.MINIMIZE,
        description="Root mean squared error, in target units. Penalizes large errors.",
    ),
    MetricDefinition(
        name="mae",
        display_name="MAE",
        direction=MetricDirection.MINIMIZE,
        description="Mean absolute error, in target units. Robust to outliers.",
    ),
    MetricDefinition(
        name="r2",
        display_name="R²",
        direction=MetricDirection.MAXIMIZE,
        description="Share of variance explained. Negative means worse than predicting the mean.",
    ),
    MetricDefinition(
        name="mape",
        display_name="MAPE",
        direction=MetricDirection.MINIMIZE,
        description="Mean absolute percentage error; rows with a zero target are excluded.",
    ),
    MetricDefinition(
        name="smape",
        display_name="SMAPE",
        direction=MetricDirection.MINIMIZE,
        description="Symmetric MAPE; bounded and defined when the target is zero.",
    ),
    MetricDefinition(
        name="bias",
        display_name="Bias (mean error)",
        direction=MetricDirection.MINIMIZE,
        description="Mean signed error. Positive means the model over-predicts on average.",
    ),
)

METRICS_BY_PROBLEM: dict[ProblemType, tuple[MetricDefinition, ...]] = {
    ProblemType.BINARY_CLASSIFICATION: BINARY_METRICS,
    ProblemType.MULTICLASS_CLASSIFICATION: MULTICLASS_METRICS,
    ProblemType.REGRESSION: REGRESSION_METRICS,
}

DEFAULT_PRIMARY_METRIC: dict[ProblemType, str] = {
    ProblemType.BINARY_CLASSIFICATION: "roc_auc",
    ProblemType.MULTICLASS_CLASSIFICATION: "macro_f1",
    ProblemType.REGRESSION: "rmse",
}

# Bias is signed: a model can be "best" at it by cancelling errors out. Never rank on it.
NON_RANKING_METRICS = frozenset({"bias"})


class UnknownMetricError(ValueError):
    """Raised when a configuration requests a metric that does not exist for the problem."""


def metrics_for(problem_type: ProblemType) -> tuple[MetricDefinition, ...]:
    return METRICS_BY_PROBLEM[problem_type]


def metric_names(problem_type: ProblemType) -> list[str]:
    return [m.name for m in metrics_for(problem_type)]


def selectable_primary_metrics(problem_type: ProblemType) -> list[str]:
    return [m.name for m in metrics_for(problem_type) if m.name not in NON_RANKING_METRICS]


def metric_definition(problem_type: ProblemType, name: str) -> MetricDefinition:
    for definition in metrics_for(problem_type):
        if definition.name == name:
            return definition
    raise UnknownMetricError(
        f"Metric {name!r} is not defined for {problem_type.value}. "
        f"Available: {', '.join(metric_names(problem_type))}."
    )


def resolve_primary_metric(problem_type: ProblemType, requested: str | None) -> str:
    """Validate a requested primary metric, or fall back to the problem-type default."""
    if requested is None:
        return DEFAULT_PRIMARY_METRIC[problem_type]
    definition = metric_definition(problem_type, requested)
    if definition.name in NON_RANKING_METRICS:
        raise UnknownMetricError(
            f"Metric {requested!r} cannot be used to rank models because it is signed. "
            f"Choose one of: {', '.join(selectable_primary_metrics(problem_type))}."
        )
    return definition.name


def metric_direction(problem_type: ProblemType, name: str) -> MetricDirection:
    return metric_definition(problem_type, name).direction


def is_better(problem_type: ProblemType, name: str, candidate: float, incumbent: float) -> bool:
    """Direction-aware comparison. The only place model ranking is decided."""
    if metric_direction(problem_type, name) is MetricDirection.MAXIMIZE:
        return candidate > incumbent
    return candidate < incumbent
