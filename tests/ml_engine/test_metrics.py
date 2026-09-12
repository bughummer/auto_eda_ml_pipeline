"""Metrics must be correct, direction-aware, and non-fatal when they cannot be computed."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.contracts.common import MetricDirection, ProblemType
from ml_engine.evaluation import (
    evaluate,
    is_better,
    metric_direction,
    resolve_primary_metric,
    selectable_primary_metrics,
)
from ml_engine.evaluation.metrics import UnknownMetricError


def test_perfect_binary_predictions_score_perfectly():
    truth = pd.Series(["0", "1"] * 50)
    proba = np.array([[1.0, 0.0] if value == "0" else [0.0, 1.0] for value in truth])
    metrics = evaluate(
        ProblemType.BINARY_CLASSIFICATION,
        truth,
        truth.to_numpy(),
        y_proba=proba,
        class_labels=["0", "1"],
        positive_class="1",
    )
    assert metrics.get("roc_auc") == pytest.approx(1.0)
    assert metrics.get("accuracy") == pytest.approx(1.0)
    assert metrics.get("f1") == pytest.approx(1.0)
    assert metrics.confusion_matrix.matrix == [[50, 0], [0, 50]]


def test_binary_metrics_include_the_rate_pair():
    truth = pd.Series(["0"] * 80 + ["1"] * 20)
    predictions = np.array(["0"] * 90 + ["1"] * 10)
    metrics = evaluate(
        ProblemType.BINARY_CLASSIFICATION,
        truth,
        predictions,
        class_labels=["0", "1"],
        positive_class="1",
    )
    assert metrics.get("positive_class_rate") == pytest.approx(0.2)
    assert metrics.get("predicted_positive_rate") == pytest.approx(0.1)


def test_missing_probabilities_are_reported_not_fatal():
    truth = pd.Series(["0", "1"] * 50)
    metrics = evaluate(
        ProblemType.BINARY_CLASSIFICATION,
        truth,
        truth.to_numpy(),
        class_labels=["0", "1"],
        positive_class="1",
    )
    assert metrics.get("roc_auc") is None
    assert any(w.rule == "probabilities_unavailable" for w in metrics.warnings)
    assert metrics.get("accuracy") is not None


def test_multiclass_metrics_are_computed():
    truth = pd.Series(["a", "b", "c"] * 40)
    metrics = evaluate(
        ProblemType.MULTICLASS_CLASSIFICATION,
        truth,
        truth.to_numpy(),
        class_labels=["a", "b", "c"],
    )
    assert metrics.get("macro_f1") == pytest.approx(1.0)
    assert metrics.get("weighted_f1") == pytest.approx(1.0)
    assert metrics.confusion_matrix.labels == ["a", "b", "c"]


def test_regression_metrics_match_hand_computed_values():
    truth = pd.Series([10.0, 20.0, 30.0, 40.0])
    predictions = np.array([12.0, 18.0, 33.0, 39.0])
    metrics = evaluate(ProblemType.REGRESSION, truth, predictions)
    assert metrics.get("mae") == pytest.approx(2.0)
    assert metrics.get("rmse") == pytest.approx(np.sqrt(4.5))
    assert metrics.get("bias") == pytest.approx(0.5)


def test_mape_excludes_zero_targets_and_says_so():
    truth = pd.Series([0.0, 10.0, 20.0])
    metrics = evaluate(ProblemType.REGRESSION, truth, np.array([1.0, 11.0, 18.0]))
    assert metrics.get("mape") is not None
    assert metrics.get("smape") is not None
    assert any(w.rule == "mape_excluded_zero_targets" for w in metrics.warnings)


def test_non_finite_predictions_are_excluded_with_a_warning():
    truth = pd.Series([1.0, 2.0, 3.0])
    metrics = evaluate(ProblemType.REGRESSION, truth, np.array([1.0, np.nan, 3.0]))
    assert metrics.get("mae") is not None
    assert any(w.rule == "non_finite_predictions_excluded" for w in metrics.warnings)


@pytest.mark.parametrize(
    ("problem_type", "expected"),
    [
        (ProblemType.BINARY_CLASSIFICATION, "roc_auc"),
        (ProblemType.MULTICLASS_CLASSIFICATION, "macro_f1"),
        (ProblemType.REGRESSION, "rmse"),
    ],
)
def test_default_primary_metrics(problem_type, expected):
    assert resolve_primary_metric(problem_type, None) == expected


def test_metric_directions_are_declared_correctly():
    assert (
        metric_direction(ProblemType.BINARY_CLASSIFICATION, "roc_auc") is MetricDirection.MAXIMIZE
    )
    assert (
        metric_direction(ProblemType.BINARY_CLASSIFICATION, "log_loss") is MetricDirection.MINIMIZE
    )
    assert metric_direction(ProblemType.REGRESSION, "rmse") is MetricDirection.MINIMIZE
    assert metric_direction(ProblemType.REGRESSION, "r2") is MetricDirection.MAXIMIZE


def test_ranking_respects_direction():
    assert is_better(ProblemType.REGRESSION, "rmse", 1.0, 2.0) is True
    assert is_better(ProblemType.REGRESSION, "r2", 0.9, 0.8) is True
    assert is_better(ProblemType.BINARY_CLASSIFICATION, "log_loss", 0.7, 0.3) is False


def test_signed_metrics_cannot_be_used_for_ranking():
    assert "bias" not in selectable_primary_metrics(ProblemType.REGRESSION)
    with pytest.raises(UnknownMetricError):
        resolve_primary_metric(ProblemType.REGRESSION, "bias")


def test_unknown_metric_is_rejected():
    with pytest.raises(UnknownMetricError):
        resolve_primary_metric(ProblemType.BINARY_CLASSIFICATION, "not_a_metric")
