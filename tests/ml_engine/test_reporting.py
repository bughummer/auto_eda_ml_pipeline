"""Comparison must rank in the right direction and represent failures honestly."""

from datetime import UTC, datetime

import pytest

from ml_engine.contracts.common import ExperimentStatus, ModelRunStatus, ProblemType
from ml_engine.contracts.metrics import MetricSet
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.reporting import build_comparison, final_status


def _metadata(name: str, **metrics) -> ModelMetadata:
    return ModelMetadata(
        experiment_id="exp-1",
        model_name=name,
        display_name=name.title(),
        status=ModelRunStatus.COMPLETED,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        library="test",
        library_version="1.0",
        metrics=MetricSet(values=metrics),
        training_duration_seconds=1.0,
        feature_count=5,
    )


def _failure(name: str) -> ModelFailure:
    return ModelFailure(
        experiment_id="exp-1",
        model_name=name,
        error_code="MODEL_TRAINING_FAILED",
        message="the library ran out of memory",
        failed_at=datetime.now(UTC),
    )


def test_maximized_metric_ranks_highest_first():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=0.7), _metadata("b", roc_auc=0.9)],
        failures=[],
    )
    assert comparison.best_model == "b"
    assert [entry.model_name for entry in comparison.models] == ["b", "a"]


def test_minimized_metric_ranks_lowest_first():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.REGRESSION,
        primary_metric="rmse",
        successes=[
            _metadata("a", rmse=10.0).model_copy(update={"problem_type": ProblemType.REGRESSION}),
            _metadata("b", rmse=4.0).model_copy(update={"problem_type": ProblemType.REGRESSION}),
        ],
        failures=[],
    )
    assert comparison.best_model == "b"
    assert comparison.direction.value == "minimize"


def test_failed_models_appear_without_a_rank():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=0.8)],
        failures=[_failure("b")],
    )
    failed = next(entry for entry in comparison.models if entry.model_name == "b")
    assert failed.status is ModelRunStatus.FAILED
    assert failed.rank is None
    assert failed.failure_message
    assert comparison.succeeded_count == 1 and comparison.failed_count == 1


def test_partial_failure_completes_with_warnings():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=0.8)],
        failures=[_failure("b")],
    )
    assert final_status(comparison) is ExperimentStatus.COMPLETED_WITH_WARNINGS


def test_total_failure_fails_the_experiment():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[],
        failures=[_failure("a")],
    )
    assert final_status(comparison) is ExperimentStatus.FAILED
    assert comparison.best_model is None


def test_clean_run_completes():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=0.8), _metadata("b", roc_auc=0.7)],
        failures=[],
    )
    assert final_status(comparison) is ExperimentStatus.COMPLETED


def test_model_without_the_primary_metric_cannot_be_best():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", accuracy=0.99)],
        failures=[],
    )
    assert comparison.best_model is None
    assert any(w.rule == "no_rankable_model" for w in comparison.warnings)


def test_tied_models_are_called_out():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=0.8), _metadata("b", roc_auc=0.8)],
        failures=[],
    )
    assert any(w.rule == "models_statistically_indistinguishable" for w in comparison.warnings)


def test_best_score_matches_the_best_model():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=0.71), _metadata("b", roc_auc=0.93)],
        failures=[],
    )
    assert comparison.best_score == pytest.approx(0.93)


def test_a_perfect_score_is_flagged_as_probable_leakage():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=1.0)],
        failures=[],
    )
    warning = next(w for w in comparison.warnings if w.rule == "suspiciously_perfect_score")
    assert warning.category.value == "leakage"


def test_a_normal_score_is_not_flagged():
    comparison = build_comparison(
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        successes=[_metadata("a", roc_auc=0.87)],
        failures=[],
    )
    assert not any(w.rule == "suspiciously_perfect_score" for w in comparison.warnings)
