"""Job entrypoints: same code in SageMaker and locally, failures always become artifacts."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from jobs.evaluation.main import run_evaluation
from jobs.preprocessing.main import main as preparation_main
from jobs.preprocessing.main import required_strategies, run_preparation
from jobs.profiling.main import main as profiling_main
from jobs.profiling.main import run_profiling
from jobs.training.main import main as training_main
from jobs.training.main import run_training
from ml_engine.contracts.common import ClassWeighting, ProblemType, RequestedProblemType
from ml_engine.contracts.config import (
    ComputeConfig,
    DatasetReference,
    EnvironmentCapture,
    ExperimentConfig,
    FeatureSelection,
    ModelSpec,
    PreprocessingConfig,
    SplitConfig,
)
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.io import read_model, write_model

FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]


def _config(dataset_csv, layout, models: list[str]) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp-test",
        name="test",
        created_at=datetime.now(UTC),
        dataset=DatasetReference(uri=str(dataset_csv), file_format="csv"),
        target_column="churned",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        requested_problem_type=RequestedProblemType.AUTO,
        primary_metric="roc_auc",
        feature_selection=FeatureSelection(selected_features=FEATURES),
        split=SplitConfig(),
        preprocessing=PreprocessingConfig(),
        models=[ModelSpec(name=name) for name in models],
        class_weighting=ClassWeighting.AUTO,
        compute=ComputeConfig(),
        environment=EnvironmentCapture(
            python_version="3.12",
            platform="test",
            ml_factory_version="1.0.0",
            captured_at=datetime.now(UTC),
        ),
        artifact_prefix=layout.base,
    )


def test_profiling_job_writes_both_artifacts(store, layout, dataset_csv):
    eda, leakage = run_profiling(
        store,
        layout,
        experiment_id="exp-test",
        dataset_uri=str(dataset_csv),
        target_column="churned",
    )
    assert store.exists(layout.eda) and store.exists(layout.leakage)
    assert eda.dataset.row_count == 400
    assert any(f.rule == "exact_target_duplicate" for f in leakage.findings)
    assert read_model(store, layout.eda, EdaReport).experiment_id == "exp-test"


def test_profiling_cli_reports_a_missing_target_as_an_artifact(tmp_path, store, dataset_csv):
    base = str(tmp_path / "exp-missing")
    code = profiling_main(
        [
            "--experiment-id",
            "exp-missing",
            "--artifact-base",
            base,
            "--dataset-uri",
            str(dataset_csv),
            "--target-column",
            "does_not_exist",
        ]
    )
    assert code == 1
    failure = read_model(store, f"{base}/eda/failure.json", ModelFailure)
    assert failure.error_code == "TARGET_COLUMN_MISSING"
    assert "does_not_exist" in failure.message


def test_preparation_job_splits_and_fits(store, layout, dataset_csv):
    run_profiling(
        store,
        layout,
        experiment_id="exp-test",
        dataset_uri=str(dataset_csv),
        target_column="churned",
    )
    write_model(
        store, layout.experiment_config, _config(dataset_csv, layout, ["logistic_regression"])
    )

    report = run_preparation(store, layout, experiment_id="exp-test")
    assert report.split.train_row_count + report.split.validation_row_count == report.rows_after
    assert store.exists(layout.train_dataset) and store.exists(layout.validation_dataset)
    assert store.exists(layout.preprocessor("dense_numeric"))
    assert "dense_numeric" in report.preprocessing
    assert report.preprocessing["dense_numeric"].fitted_on == "train"


def test_preparation_fits_a_pipeline_per_required_strategy(store, layout, dataset_csv):
    run_profiling(
        store,
        layout,
        experiment_id="exp-test",
        dataset_uri=str(dataset_csv),
        target_column="churned",
    )
    config = _config(dataset_csv, layout, ["logistic_regression", "catboost"])
    write_model(store, layout.experiment_config, config)
    assert set(required_strategies(config)) == {"dense_numeric", "native_categorical"}

    report = run_preparation(store, layout, experiment_id="exp-test")
    assert set(report.preprocessing) == {"dense_numeric", "native_categorical"}
    assert store.exists(layout.preprocessor("native_categorical"))


def test_preparation_refuses_an_unusable_selection(store, layout, dataset_csv):
    run_profiling(
        store,
        layout,
        experiment_id="exp-test",
        dataset_uri=str(dataset_csv),
        target_column="churned",
    )
    config = _config(dataset_csv, layout, ["logistic_regression"])
    config.feature_selection = FeatureSelection(selected_features=["region_code"])
    write_model(store, layout.experiment_config, config)

    code = preparation_main(["--experiment-id", "exp-test", "--artifact-base", layout.base])
    assert code == 1
    failure = read_model(store, layout.path("validation/failure.json"), ModelFailure)
    assert failure.error_code == "NO_USABLE_FEATURES"


def test_training_job_writes_model_and_metadata(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv, ["logistic_regression"])
    metadata = run_training(
        store, layout, experiment_id="exp-test", model_name="logistic_regression"
    )
    assert store.exists(layout.model_artifact("logistic_regression"))
    stored = read_model(store, layout.model_metadata("logistic_regression"), ModelMetadata)
    assert stored.primary_score == metadata.primary_score
    assert stored.artifacts.model_uri == layout.model_artifact("logistic_regression")
    assert stored.preprocessing is not None


def test_training_job_reports_an_unconfigured_model(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv, ["logistic_regression"])
    code = training_main(
        ["--experiment-id", "exp-test", "--artifact-base", layout.base, "--model-name", "xgboost"]
    )
    assert code == 1
    failure = read_model(store, layout.model_failure("xgboost"), ModelFailure)
    assert failure.error_code == "MODEL_NOT_CONFIGURED"


def test_evaluation_job_builds_comparison_and_summary(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv, ["logistic_regression", "random_forest"])
    for name in ("logistic_regression", "random_forest"):
        run_training(store, layout, experiment_id="exp-test", model_name=name)

    comparison = run_evaluation(store, layout, experiment_id="exp-test")
    assert comparison.succeeded_count == 2
    assert comparison.best_model in {"logistic_regression", "random_forest"}
    assert store.exists(layout.comparison) and store.exists(layout.summary)


def test_one_failed_model_does_not_lose_the_others(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv, ["logistic_regression", "random_forest"])
    run_training(store, layout, experiment_id="exp-test", model_name="logistic_regression")
    write_model(
        store,
        layout.model_failure("random_forest"),
        ModelFailure(
            experiment_id="exp-test",
            model_name="random_forest",
            error_code="MODEL_TRAINING_FAILED",
            message="out of memory",
            failed_at=datetime.now(UTC),
        ),
    )
    comparison = run_evaluation(store, layout, experiment_id="exp-test")
    assert comparison.succeeded_count == 1 and comparison.failed_count == 1
    assert comparison.best_model == "logistic_regression"


def test_a_model_that_produced_nothing_is_still_represented(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv, ["logistic_regression", "random_forest"])
    run_training(store, layout, experiment_id="exp-test", model_name="logistic_regression")
    comparison = run_evaluation(store, layout, experiment_id="exp-test")
    missing = next(m for m in comparison.models if m.model_name == "random_forest")
    assert missing.status.value == "FAILED"
    assert "terminated abnormally" in missing.failure_message


def test_preparation_drops_rows_without_a_target(store, layout, tmp_path, classification_frame):
    frame = classification_frame.copy()
    frame.loc[frame.index[:20], "churned"] = pd.NA
    path = tmp_path / "with_missing_target.csv"
    frame.to_csv(path, index=False)

    run_profiling(
        store, layout, experiment_id="exp-test", dataset_uri=str(path), target_column="churned"
    )
    write_model(store, layout.experiment_config, _config(path, layout, ["logistic_regression"]))
    report = run_preparation(store, layout, experiment_id="exp-test")
    assert report.rows_dropped_missing_target == 20
    assert any(w.rule == "rows_dropped_missing_target" for w in report.warnings)


def _prepare(store, layout, dataset_csv, models: list[str]) -> PreparationReport:
    run_profiling(
        store,
        layout,
        experiment_id="exp-test",
        dataset_uri=str(dataset_csv),
        target_column="churned",
    )
    write_model(store, layout.experiment_config, _config(dataset_csv, layout, models))
    return run_preparation(store, layout, experiment_id="exp-test")


@pytest.fixture(autouse=True)
def _quiet_logs(caplog):
    caplog.set_level("WARNING")
