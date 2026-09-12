"""The full vertical slice: create -> EDA -> feature review -> training -> comparison.

This is the test that must never go red. It exercises the real job code, the real contracts,
the real S3-addressed artifact layout and the real experiment records; only the bytes and the
compute are substituted (an in-memory store and an in-process orchestrator).
"""

import pytest

from backend.schemas.experiments import (
    CreateExperimentRequest,
    TrainingConfigRequest,
    UpdateFeatureSelectionRequest,
)
from ml_engine.contracts.common import (
    ExperimentStatus,
    LeakageRiskLevel,
    ModelRunStatus,
    ProblemType,
    RecommendedAction,
)
from ml_engine.io import ExperimentLayout

pytestmark = pytest.mark.slow

USER = "analyst@corp.example"


def _run_to_feature_review(container, dataset_csv, target="churned", name="e2e"):
    service = container.experiments
    record = service.create(
        CreateExperimentRequest(name=name, dataset_uri=dataset_csv, target_column=target),
        USER,
    )
    container.orchestrator.wait_for_idle(timeout=300)
    return service.get(record.experiment_id)


def test_full_classification_experiment(container, dataset_csv):
    service = container.experiments
    record = _run_to_feature_review(container, dataset_csv)
    assert record.status is ExperimentStatus.FEATURE_REVIEW

    eda = service.eda(record.experiment_id)
    assert eda.dataset.row_count == 400
    assert eda.target.inferred_problem_type is ProblemType.BINARY_CLASSIFICATION

    review = service.feature_review(record.experiment_id)
    leaked = next(f for f in review.features if f.feature == "churn_copy")
    assert leaked.leakage_risk is LeakageRiskLevel.CONFIRMED_DUPLICATE
    assert leaked.selected is False  # proposed for exclusion, still visible and reversible

    keep = [
        f.feature
        for f in review.features
        if f.selected
        and f.recommended_action
        not in {RecommendedAction.CONSIDER_EXCLUDING, RecommendedAction.STRONGLY_CONSIDER_EXCLUDING}
    ]
    service.save_features(
        record.experiment_id, UpdateFeatureSelectionRequest(selected_features=keep), USER
    )
    assert service.get(record.experiment_id).status is ExperimentStatus.READY_FOR_TRAINING

    service.start_training(
        record.experiment_id,
        TrainingConfigRequest(models=["logistic_regression", "random_forest"]),
        USER,
    )
    container.orchestrator.wait_for_idle(timeout=900)

    status = service.training_status(record.experiment_id)
    assert status.status in {ExperimentStatus.COMPLETED, ExperimentStatus.COMPLETED_WITH_WARNINGS}
    assert {m.status for m in status.models} == {ModelRunStatus.COMPLETED}

    comparison = service.comparison(record.experiment_id)
    assert comparison.primary_metric == "roc_auc"
    assert comparison.direction.value == "maximize"
    assert comparison.best_model in {"logistic_regression", "random_forest"}
    assert comparison.succeeded_count == 2
    ranked = [m for m in comparison.models if m.rank]
    assert ranked[0].primary_score >= ranked[-1].primary_score

    metadata = service.model_metadata(record.experiment_id, comparison.best_model)
    assert metadata.feature_importance is not None
    assert metadata.training_row_count + metadata.validation_row_count == 400
    assert metadata.package_versions["scikit-learn"]

    summary = service.summary(record.experiment_id)
    assert summary.config.feature_selection.selected_features == keep
    assert summary.config.dataset.uri == dataset_csv
    assert summary.comparison.best_model == comparison.best_model


def test_leakage_left_in_place_produces_a_suspiciously_perfect_model(container, dataset_csv):
    """The platform does not remove leakage for you — but it told you, and the score shows it."""
    service = container.experiments
    record = _run_to_feature_review(container, dataset_csv, name="with-leak")
    review = service.feature_review(record.experiment_id)
    service.save_features(
        record.experiment_id,
        UpdateFeatureSelectionRequest(selected_features=["churn_copy", "tenure_months"]),
        USER,
    )
    service.start_training(
        record.experiment_id, TrainingConfigRequest(models=["logistic_regression"]), USER
    )
    container.orchestrator.wait_for_idle(timeout=600)

    comparison = service.comparison(record.experiment_id)
    assert comparison.best_score > 0.99
    assert any(f.feature == "churn_copy" for f in service.leakage(record.experiment_id).findings)
    assert review.features  # the warning existed before training started


def test_regression_experiment(container, regression_csv):
    service = container.experiments
    record = service.create(
        CreateExperimentRequest(name="prices", dataset_uri=regression_csv, target_column="price"),
        USER,
    )
    container.orchestrator.wait_for_idle(timeout=300)

    review = service.feature_review(record.experiment_id)
    service.save_features(
        record.experiment_id,
        UpdateFeatureSelectionRequest(selected_features=[f.feature for f in review.features]),
        USER,
    )
    service.start_training(
        record.experiment_id, TrainingConfigRequest(models=["elastic_net"]), USER
    )
    container.orchestrator.wait_for_idle(timeout=600)

    comparison = service.comparison(record.experiment_id)
    assert comparison.problem_type is ProblemType.REGRESSION
    assert comparison.primary_metric == "rmse"
    assert comparison.direction.value == "minimize"
    assert comparison.best_score > 0


def test_artifacts_are_written_where_the_architecture_says(container, dataset_csv):
    service = container.experiments
    record = _run_to_feature_review(container, dataset_csv, name="layout")
    review = service.feature_review(record.experiment_id)
    service.save_features(
        record.experiment_id,
        UpdateFeatureSelectionRequest(
            selected_features=[f.feature for f in review.features if f.selected]
        ),
        USER,
    )
    service.start_training(
        record.experiment_id, TrainingConfigRequest(models=["logistic_regression"]), USER
    )
    container.orchestrator.wait_for_idle(timeout=600)

    layout = ExperimentLayout(base=record.artifact_prefix)
    for uri in (
        layout.eda,
        layout.leakage,
        layout.experiment_config,
        layout.selected_features,
        layout.preparation,
        layout.preprocessor("dense_numeric"),
        layout.train_dataset,
        layout.validation_dataset,
        layout.model_artifact("logistic_regression"),
        layout.model_metadata("logistic_regression"),
        layout.comparison,
        layout.summary,
    ):
        assert container.store.exists(uri), f"missing artifact: {uri}"


def test_failed_model_does_not_fail_the_experiment(container, dataset_csv, monkeypatch):
    service = container.experiments
    record = _run_to_feature_review(container, dataset_csv, name="partial-failure")
    review = service.feature_review(record.experiment_id)
    service.save_features(
        record.experiment_id,
        UpdateFeatureSelectionRequest(
            selected_features=[f.feature for f in review.features if f.selected]
        ),
        USER,
    )

    import ml_engine.models.random_forest as rf

    def explode(self, params, context):
        raise RuntimeError("simulated library failure")

    monkeypatch.setattr(rf.RandomForestClassifierPlugin, "build", explode)

    service.start_training(
        record.experiment_id,
        TrainingConfigRequest(models=["logistic_regression", "random_forest"]),
        USER,
    )
    container.orchestrator.wait_for_idle(timeout=600)

    status = service.training_status(record.experiment_id)
    assert status.status is ExperimentStatus.COMPLETED_WITH_WARNINGS
    failed = next(m for m in status.models if m.model_name == "random_forest")
    assert failed.status is ModelRunStatus.FAILED
    assert failed.failure_message and "simulated library failure" in failed.failure_message
    comparison = service.comparison(record.experiment_id)
    assert comparison.best_model == "logistic_regression"


def test_a_dataset_without_a_usable_target_fails_cleanly(container, tmp_path):
    import pandas as pd

    local = tmp_path / "constant_target.csv"
    pd.DataFrame({"x": range(50), "y": [1] * 50}).to_csv(local, index=False)
    uri = container.store.put_file("s3://ml-factory-test-data/curated/constant_target.csv", local)

    service = container.experiments
    record = service.create(
        CreateExperimentRequest(name="bad target", dataset_uri=uri, target_column="y"), USER
    )
    container.orchestrator.wait_for_idle(timeout=300)

    eda = service.eda(record.experiment_id)
    assert eda.target.inferred_problem_type is None
    assert any(w.rule == "target_not_learnable" for w in eda.warnings)
