"""Metric computation.

Every metric is computed independently and defensively: a metric that cannot be computed is
omitted and explained with a warning. A metric failure never fails an experiment.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from ml_engine.contracts.common import ProblemType, Severity, WarningCategory
from ml_engine.contracts.metrics import ConfusionMatrix, MetricSet
from ml_engine.contracts.warnings import AnalysisWarning

_EPSILON = 1e-12


def evaluate(
    problem_type: ProblemType,
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    class_labels: list[str] | None = None,
    positive_class: str | None = None,
) -> MetricSet:
    """Compute every metric defined for ``problem_type``."""
    if problem_type is ProblemType.REGRESSION:
        return _evaluate_regression(y_true, y_pred)
    return _evaluate_classification(
        problem_type,
        y_true,
        y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
        positive_class=positive_class,
    )


def _safe(
    result: MetricSet,
    name: str,
    compute: Callable[[], float],
    *,
    severity: Severity = Severity.LOW,
) -> None:
    """Run one metric. Failure is recorded as a warning, never raised."""
    try:
        value = float(compute())
    except Exception as exc:
        result.warnings.append(
            AnalysisWarning(
                rule="metric_not_computed",
                category=WarningCategory.METRIC,
                severity=severity,
                message=f"Metric '{name}' could not be computed: {type(exc).__name__}: {exc}",
                details={"metric": name},
            )
        )
        return
    if not np.isfinite(value):
        result.warnings.append(
            AnalysisWarning(
                rule="metric_not_finite",
                category=WarningCategory.METRIC,
                severity=severity,
                message=f"Metric '{name}' evaluated to a non-finite value and was omitted.",
                details={"metric": name},
            )
        )
        return
    result.values[name] = round(value, 6)


def _evaluate_classification(
    problem_type: ProblemType,
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None,
    class_labels: list[str] | None,
    positive_class: str | None,
) -> MetricSet:
    result = MetricSet()
    true_labels = y_true.astype("string").to_numpy()
    predicted_labels = np.asarray(y_pred).astype(str)
    labels = class_labels or sorted(set(true_labels.tolist()) | set(predicted_labels.tolist()))

    _safe(result, "accuracy", lambda: skm.accuracy_score(true_labels, predicted_labels))

    if problem_type is ProblemType.BINARY_CLASSIFICATION:
        positive = positive_class or (labels[-1] if labels else None)
        if positive is None:
            return result
        _safe(
            result,
            "precision",
            lambda: skm.precision_score(
                true_labels, predicted_labels, pos_label=positive, zero_division=0
            ),
        )
        _safe(
            result,
            "recall",
            lambda: skm.recall_score(
                true_labels, predicted_labels, pos_label=positive, zero_division=0
            ),
        )
        _safe(
            result,
            "f1",
            lambda: skm.f1_score(
                true_labels, predicted_labels, pos_label=positive, zero_division=0
            ),
        )
        result.values["positive_class_rate"] = round(float((true_labels == positive).mean()), 6)
        result.values["predicted_positive_rate"] = round(
            float((predicted_labels == positive).mean()), 6
        )

        scores = _positive_scores(y_proba, labels, positive)
        if scores is None:
            result.warnings.append(
                AnalysisWarning(
                    rule="probabilities_unavailable",
                    category=WarningCategory.METRIC,
                    severity=Severity.MEDIUM,
                    message=(
                        "The model does not expose probabilities, so ROC AUC, PR AUC and log "
                        "loss were not computed."
                    ),
                )
            )
        else:
            binary_true = (true_labels == positive).astype(int)
            _safe(result, "roc_auc", lambda: skm.roc_auc_score(binary_true, scores))
            _safe(result, "pr_auc", lambda: skm.average_precision_score(binary_true, scores))
            _safe(
                result,
                "log_loss",
                lambda: skm.log_loss(binary_true, np.clip(scores, _EPSILON, 1 - _EPSILON)),
            )
    else:
        _safe(
            result,
            "macro_f1",
            lambda: skm.f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
        )
        _safe(
            result,
            "weighted_f1",
            lambda: skm.f1_score(
                true_labels, predicted_labels, average="weighted", zero_division=0
            ),
        )
        _safe(
            result,
            "macro_precision",
            lambda: skm.precision_score(
                true_labels, predicted_labels, average="macro", zero_division=0
            ),
        )
        _safe(
            result,
            "macro_recall",
            lambda: skm.recall_score(
                true_labels, predicted_labels, average="macro", zero_division=0
            ),
        )
        if y_proba is not None and class_labels:
            _safe(
                result,
                "log_loss",
                lambda: skm.log_loss(true_labels, y_proba, labels=class_labels),
                severity=Severity.LOW,
            )

    try:
        matrix = skm.confusion_matrix(true_labels, predicted_labels, labels=labels)
        result.confusion_matrix = ConfusionMatrix(
            labels=[str(label) for label in labels],
            matrix=[[int(cell) for cell in row] for row in matrix],
        )
    except Exception as exc:
        result.warnings.append(
            AnalysisWarning(
                rule="confusion_matrix_not_computed",
                category=WarningCategory.METRIC,
                severity=Severity.LOW,
                message=f"Confusion matrix could not be computed: {exc}",
            )
        )
    return result


def _positive_scores(
    y_proba: np.ndarray | None, labels: list[str], positive: str
) -> np.ndarray | None:
    if y_proba is None:
        return None
    array = np.asarray(y_proba, dtype="float64")
    if array.ndim == 1:
        return array
    if array.shape[1] == 1:
        return array[:, 0]
    if positive in labels:
        return array[:, labels.index(positive)]
    return array[:, -1]


def _evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> MetricSet:
    result = MetricSet()
    truth = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype="float64")
    prediction = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(truth) & np.isfinite(prediction)
    dropped = int((~mask).sum())
    if dropped:
        result.warnings.append(
            AnalysisWarning(
                rule="non_finite_predictions_excluded",
                category=WarningCategory.METRIC,
                severity=Severity.MEDIUM,
                message=(
                    f"{dropped} row(s) had a non-finite target or prediction and were excluded "
                    "from the regression metrics."
                ),
                details={"excluded_rows": dropped},
            )
        )
    truth, prediction = truth[mask], prediction[mask]
    if truth.size == 0:
        result.warnings.append(
            AnalysisWarning(
                rule="no_evaluable_rows",
                category=WarningCategory.METRIC,
                severity=Severity.HIGH,
                message="No rows remained with finite targets and predictions.",
            )
        )
        return result

    errors = prediction - truth
    _safe(result, "mae", lambda: np.abs(errors).mean())
    _safe(result, "rmse", lambda: float(np.sqrt(np.mean(errors**2))))
    _safe(result, "r2", lambda: skm.r2_score(truth, prediction))
    _safe(result, "bias", lambda: errors.mean())

    non_zero = np.abs(truth) > _EPSILON
    excluded = int((~non_zero).sum())
    if non_zero.any():
        _safe(
            result,
            "mape",
            lambda: float(np.mean(np.abs(errors[non_zero] / truth[non_zero])) * 100.0),
        )
    if excluded:
        result.warnings.append(
            AnalysisWarning(
                rule="mape_excluded_zero_targets",
                category=WarningCategory.METRIC,
                severity=Severity.LOW,
                message=(
                    f"MAPE excluded {excluded} row(s) with a zero target, where the percentage "
                    "error is undefined. Use SMAPE for a zero-safe alternative."
                ),
                details={"excluded_rows": excluded},
            )
        )

    denominator = (np.abs(truth) + np.abs(prediction)) / 2.0
    smape_mask = denominator > _EPSILON
    if smape_mask.any():
        _safe(
            result,
            "smape",
            lambda: float(np.mean(np.abs(errors[smape_mask]) / denominator[smape_mask]) * 100.0),
        )
    return result
