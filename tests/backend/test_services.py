"""Control-plane behaviour: validation, state transitions and artifact access."""

import pytest

from backend.config import Settings
from backend.container import build_container
from backend.errors import ConflictError, FeatureDisabledError, ValidationError
from backend.repositories import ObjectStoreExperimentRepository
from backend.schemas.experiments import (
    CreateExperimentRequest,
    TrainingConfigRequest,
    UpdateFeatureSelectionRequest,
)
from backend.schemas.reasoning import ReasoningRequest
from ml_engine.contracts.common import ExperimentStatus, ProblemType

USER = "analyst@corp.example"


@pytest.fixture
def ready_experiment(container, dataset_csv):
    record = container.experiments.create(
        CreateExperimentRequest(
            name="churn", dataset_uri=str(dataset_csv), target_column="churned"
        ),
        USER,
    )
    container.orchestrator.wait_for_idle(timeout=300)
    return container.experiments.get(record.experiment_id)


def test_dataset_allow_list_is_enforced(tmp_path):
    settings = Settings(
        artifact_bucket="s3://artifacts",
        allowed_dataset_prefixes=["s3://approved/curated"],
        eda_state_machine_arn="arn:eda",
        training_state_machine_arn="arn:train",
    )
    assert settings.dataset_uri_allowed("s3://approved/curated/a.csv") is True
    assert settings.dataset_uri_allowed("s3://approved/curated-other/a.csv") is False
    assert settings.dataset_uri_allowed("s3://elsewhere/a.csv") is False


def test_missing_aws_configuration_is_reported():
    problems = Settings().validate_configuration()
    assert any("ARTIFACT_BUCKET" in p for p in problems)
    assert any("ALLOWED_DATASET_PREFIXES" in p for p in problems)
    assert any("STATE_MACHINE_ARN" in p for p in problems)


def test_artifact_roots_are_allow_listed():
    settings = Settings(
        artifact_bucket="s3://artifacts",
        additional_artifact_roots=["s3://other-team-artifacts"],
    )
    assert settings.artifact_roots == ["s3://artifacts", "s3://other-team-artifacts"]
    assert settings.artifact_root_allowed("s3://other-team-artifacts/") is True
    assert settings.artifact_root_allowed("s3://somewhere-else") is False


def test_creation_records_dataset_identity(container, dataset_csv):
    record = container.experiments.create(
        CreateExperimentRequest(name="x", dataset_uri=str(dataset_csv), target_column="churned"),
        USER,
    )
    assert record.dataset.file_format == "csv"
    assert record.dataset.size_bytes > 0
    assert record.created_by == USER
    assert record.artifact_prefix.endswith(record.experiment_id)


def test_eda_completion_moves_to_feature_review(ready_experiment):
    assert ready_experiment.status is ExperimentStatus.FEATURE_REVIEW


def test_feature_review_joins_profile_and_risk(container, ready_experiment):
    review = container.experiments.feature_review(ready_experiment.experiment_id)
    assert review.problem_type is ProblemType.BINARY_CLASSIFICATION
    assert "churned" not in {f.feature for f in review.features}
    identifier = next(f for f in review.features if f.feature == "customer_id")
    assert identifier.is_likely_id is True
    assert identifier.reasons
    assert review.decided is False


def test_saving_features_records_the_exact_decision(container, ready_experiment):
    selection = container.experiments.save_features(
        ready_experiment.experiment_id,
        UpdateFeatureSelectionRequest(
            selected_features=["tenure_months", "contract"],
            exclusion_reasons={"customer_id": "identifier"},
        ),
        USER,
    )
    assert selection.selected_features == ["tenure_months", "contract"]
    assert "customer_id" in selection.excluded_features
    assert selection.exclusion_reasons["customer_id"] == "identifier"
    assert selection.decided_by == USER
    review = container.experiments.feature_review(ready_experiment.experiment_id)
    assert review.decided is True
    assert review.selected_count == 2


def test_the_target_cannot_be_selected_as_a_feature(container, ready_experiment):
    with pytest.raises(ValidationError, match="target column"):
        container.experiments.save_features(
            ready_experiment.experiment_id,
            UpdateFeatureSelectionRequest(selected_features=["churned"]),
            USER,
        )


def test_unknown_features_are_rejected(container, ready_experiment):
    with pytest.raises(ValidationError, match="not in the dataset"):
        container.experiments.save_features(
            ready_experiment.experiment_id,
            UpdateFeatureSelectionRequest(selected_features=["ghost"]),
            USER,
        )


def test_training_config_defaults_come_from_the_resolved_problem_type(container, ready_experiment):
    config = container.experiments.training_config(ready_experiment.experiment_id)
    assert config.problem_type is ProblemType.BINARY_CLASSIFICATION
    assert config.primary_metric == "roc_auc"
    assert "logistic_regression" in config.available_models
    assert "bias" not in config.available_metrics


def test_invalid_primary_metric_is_rejected(container, ready_experiment):
    container.experiments.save_features(
        ready_experiment.experiment_id,
        UpdateFeatureSelectionRequest(selected_features=["tenure_months"]),
        USER,
    )
    with pytest.raises(ValidationError):
        container.experiments.start_training(
            ready_experiment.experiment_id, TrainingConfigRequest(primary_metric="rmse"), USER
        )


def test_model_not_supporting_the_problem_type_is_rejected(container, ready_experiment):
    container.experiments.save_features(
        ready_experiment.experiment_id,
        UpdateFeatureSelectionRequest(selected_features=["tenure_months"]),
        USER,
    )
    with pytest.raises(ValidationError, match="does not support"):
        container.experiments.start_training(
            ready_experiment.experiment_id, TrainingConfigRequest(models=["elastic_net"]), USER
        )


def test_frozen_config_captures_reproducibility_metadata(container, ready_experiment):
    from ml_engine.contracts.config import ExperimentConfig
    from ml_engine.io import ExperimentLayout, read_model

    container.experiments.save_features(
        ready_experiment.experiment_id,
        UpdateFeatureSelectionRequest(selected_features=["tenure_months", "contract"]),
        USER,
    )
    container.experiments.start_training(
        ready_experiment.experiment_id,
        TrainingConfigRequest(models=["logistic_regression"], random_seed=99),
        USER,
    )
    container.orchestrator.wait_for_idle(timeout=600)

    layout = ExperimentLayout(base=ready_experiment.artifact_prefix)
    config = read_model(container.store, layout.experiment_config, ExperimentConfig)
    assert config.split.random_seed == 99
    assert config.environment.package_versions["scikit-learn"]
    assert config.environment.ml_factory_version == "1.0.0"
    assert config.dataset.uri.endswith("churn.csv")
    assert config.feature_selection.selected_features == ["tenure_months", "contract"]


def test_reasoning_requires_bedrock(container, ready_experiment):
    with pytest.raises(FeatureDisabledError):
        container.reasoning.run(ready_experiment.experiment_id, ReasoningRequest())


def test_reasoning_runs_against_deterministic_artifacts(artifact_root, dataset_csv, store):
    import json

    from tests.support.inline_orchestrator import InlineOrchestrator

    settings = Settings(
        artifact_bucket=artifact_root,
        allowed_dataset_prefixes=["s3://ml-factory-test-data"],
        bedrock_enabled=True,
        bedrock_model_id="fake.model",
    )
    container = build_container(
        settings,
        store=store,
        repository=ObjectStoreExperimentRepository(store, settings.artifact_root),
        orchestrator=InlineOrchestrator(store),
    )
    try:
        record = container.experiments.create(
            CreateExperimentRequest(name="x", dataset_uri=dataset_csv, target_column="churned"),
            USER,
        )
        container.orchestrator.wait_for_idle(timeout=300)

        class FakeClient:
            model_id = "fake.model"

            def invoke(self, system_prompt, user_prompt):
                assert "customer_id" in user_prompt  # deterministic context was supplied
                return json.dumps(
                    {
                        "executive_summary": "Reviewed.",
                        "semantic_leakage": [],
                        "data_quality_concerns": [],
                        "assumptions": [],
                        "limitations": [],
                        "proposed_experiments": [],
                    }
                )

        container.reasoning._client = FakeClient()
        report = container.reasoning.run(record.experiment_id, ReasoningRequest())
        assert report.experiment_id == record.experiment_id
        assert report.input_artifacts
        assert container.reasoning.get(record.experiment_id).executive_summary == "Reviewed."
    finally:
        container.shutdown()


def test_dictionary_upload_reconciles_with_the_dataset(container, ready_experiment):
    payload = b"column,definition,available at prediction time\ncontract,Contract type,yes\nghost,Unknown,no\n"
    dictionary = container.dictionary.upload(
        ready_experiment.experiment_id, filename="dict.csv", payload=payload
    )
    assert dictionary.unmatched_columns == ["ghost"]
    assert "customer_id" in dictionary.undocumented_columns
    review = container.experiments.feature_review(ready_experiment.experiment_id)
    contract = next(f for f in review.features if f.feature == "contract")
    assert contract.documentation == "Contract type"
    assert contract.available_at_prediction_time is True


def test_oversized_dictionary_is_rejected(container, ready_experiment):
    container.settings.max_dictionary_upload_bytes = 10
    with pytest.raises(ValidationError, match="maximum upload size"):
        container.dictionary.upload(
            ready_experiment.experiment_id, filename="d.csv", payload=b"column,definition\na,b\n"
        )


def test_repository_soft_delete_hides_the_record(container, ready_experiment):
    from backend.errors import NotFoundError

    container.experiments.delete(ready_experiment.experiment_id)
    with pytest.raises(NotFoundError):
        container.experiments.get(ready_experiment.experiment_id)
    assert container.experiments.list() == []


def test_repository_round_trips_a_record_through_the_store(artifact_root, store):
    from datetime import UTC, datetime

    from ml_engine.contracts.config import DatasetReference
    from ml_engine.contracts.experiment import ExperimentDefinition
    from ml_engine.io import ExperimentLayout

    repository = ObjectStoreExperimentRepository(store, str(artifact_root))
    layout = ExperimentLayout.for_experiment(str(artifact_root), "exp-1")
    now = datetime.now(UTC)
    repository.create(
        ExperimentDefinition(
            experiment_id="exp-1",
            name="x",
            created_at=now,
            dataset=DatasetReference(uri="s3://b/k.csv", file_format="csv"),
            target_column="y",
            artifact_prefix=layout.base,
        )
    )
    assert store.exists(layout.definition)
    assert store.exists(layout.control_state)

    updated = repository.update("exp-1", status=ExperimentStatus.READY_FOR_TRAINING)
    assert updated.status is ExperimentStatus.READY_FOR_TRAINING
    assert updated.updated_at >= now
    assert repository.get("exp-1").status is ExperimentStatus.READY_FOR_TRAINING
    assert [record.experiment_id for record in repository.list()] == ["exp-1"]


def test_the_control_plane_cannot_write_workflow_fields(artifact_root, store):
    from datetime import UTC, datetime

    from ml_engine.contracts.config import DatasetReference
    from ml_engine.contracts.experiment import ExperimentDefinition
    from ml_engine.io import ExperimentLayout

    repository = ObjectStoreExperimentRepository(store, str(artifact_root))
    repository.create(
        ExperimentDefinition(
            experiment_id="exp-2",
            name="x",
            created_at=datetime.now(UTC),
            dataset=DatasetReference(uri="s3://b/k.csv", file_format="csv"),
            target_column="y",
            artifact_prefix=ExperimentLayout.for_experiment(str(artifact_root), "exp-2").base,
        )
    )
    with pytest.raises(ValueError, match="does not own"):
        repository.update("exp-2", best_model="xgboost")


def test_training_cannot_start_twice(container, ready_experiment):
    container.experiments.save_features(
        ready_experiment.experiment_id,
        UpdateFeatureSelectionRequest(selected_features=["tenure_months"]),
        USER,
    )
    container.repository.update(ready_experiment.experiment_id, status=ExperimentStatus.TRAINING)
    with pytest.raises(ConflictError, match="already running"):
        container.experiments.start_training(
            ready_experiment.experiment_id, TrainingConfigRequest(), USER
        )


def test_review_recommendation_reflects_eda_warnings_too(container, ready_experiment):
    """A text or constant column is worth excluding even with no leakage finding."""
    review = container.experiments.feature_review(ready_experiment.experiment_id)
    text_column = next(f for f in review.features if f.feature == "notes")
    assert text_column.leakage_risk.value == "none"
    assert text_column.recommended_action.value == "consider_excluding"
    assert text_column.reasons
