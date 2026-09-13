"""Executes every recipe in docs/notebook_usage.md.

The document promises that a notebook can drive the engine directly, with the same contracts
and the same artifact layout the SageMaker jobs use. These tests are that promise: if an
engine signature moves, the doc is wrong and this file says so.
"""

from datetime import UTC, datetime

from backend.repositories import ObjectStoreExperimentRepository
from jobs._common.experiment_state import ArtifactExperimentStateWriter
from jobs.evaluation.main import run_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.training.main import run_training
from ml_engine.contracts.common import (
    ClassWeighting,
    ExperimentStatus,
    ProblemType,
    RequestedProblemType,
)
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
from ml_engine.contracts.experiment import ControlPlaneState, ExperimentDefinition
from ml_engine.io import write_model

EXPERIMENT_ID = "exp-test"
FEATURES = ["tenure_months", "monthly_charges", "contract"]
MODELS = ["logistic_regression", "random_forest"]


def _definition(dataset_csv: str, layout) -> ExperimentDefinition:
    """Recipe B."""
    return ExperimentDefinition(
        experiment_id=EXPERIMENT_ID,
        name="Churn baseline (notebook)",
        created_by="your.name",
        created_at=datetime.now(UTC),
        dataset=DatasetReference(uri=dataset_csv, file_format="csv"),
        target_column="churned",
        artifact_prefix=layout.base,
    )


def _config(dataset_csv: str, layout) -> ExperimentConfig:
    """Recipe C."""
    return ExperimentConfig(
        experiment_id=EXPERIMENT_ID,
        name="Churn baseline (notebook)",
        created_at=datetime.now(UTC),
        dataset=DatasetReference(uri=dataset_csv, file_format="csv"),
        target_column="churned",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        requested_problem_type=RequestedProblemType.AUTO,
        primary_metric="roc_auc",
        feature_selection=FeatureSelection(selected_features=FEATURES),
        split=SplitConfig(),
        preprocessing=PreprocessingConfig(),
        models=[ModelSpec(name=name) for name in MODELS],
        class_weighting=ClassWeighting.AUTO,
        compute=ComputeConfig(),
        environment=EnvironmentCapture(
            python_version="3.12",
            platform="notebook",
            ml_factory_version="1.0.0",
            captured_at=datetime.now(UTC),
        ),
        artifact_prefix=layout.base,
    )


def test_recipe_a_profiles_and_screens_without_any_platform_scaffolding(store, layout, dataset_csv):
    eda, leakage = run_profiling(
        store,
        layout,
        experiment_id=EXPERIMENT_ID,
        dataset_uri=dataset_csv,
        target_column="churned",
    )

    assert eda.dataset.row_count == 400
    assert eda.target.inferred_problem_type is ProblemType.BINARY_CLASSIFICATION
    assert leakage.findings, "the fixture carries a duplicated target, which must be found"
    assert store.exists(layout.eda) and store.exists(layout.leakage)


def test_recipe_b_makes_a_notebook_run_visible_to_the_experiment_list(
    store, layout, artifact_root, dataset_csv
):
    write_model(store, layout.definition, _definition(dataset_csv, layout))
    write_model(
        store,
        layout.control_state,
        ControlPlaneState(experiment_id=EXPERIMENT_ID, updated_at=datetime.now(UTC)),
    )

    listed = ObjectStoreExperimentRepository(store, artifact_root).list()

    assert [record.experiment_id for record in listed] == [EXPERIMENT_ID]
    assert listed[0].name == "Churn baseline (notebook)"


def test_recipe_c_runs_the_whole_pipeline_and_reports_completion(
    store, layout, artifact_root, dataset_csv
):
    write_model(store, layout.definition, _definition(dataset_csv, layout))
    write_model(store, layout.experiment_config, _config(dataset_csv, layout))

    preparation = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    assert preparation.usable_features == FEATURES
    assert preparation.rows_after > 0

    for name in MODELS:
        metadata = run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=name)
        assert metadata.primary_metric == "roc_auc"
        assert metadata.primary_score is not None

    comparison = run_evaluation(
        store,
        layout,
        experiment_id=EXPERIMENT_ID,
        state_writer=ArtifactExperimentStateWriter(store, layout),
    )

    assert comparison.succeeded_count == len(MODELS)
    assert comparison.best_model in MODELS

    record = ObjectStoreExperimentRepository(store, artifact_root).get(EXPERIMENT_ID)
    assert record.status is ExperimentStatus.COMPLETED
    assert record.best_model == comparison.best_model
