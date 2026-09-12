"""Experiment records live in the artifact bucket, with one writer per document."""

from datetime import UTC, datetime, timedelta

import pytest

from backend.errors import NotFoundError, ValidationError
from backend.repositories import ObjectStoreExperimentRepository
from backend.schemas.experiments import CreateExperimentRequest
from ml_engine.contracts.common import ExperimentStatus
from ml_engine.contracts.config import DatasetReference
from ml_engine.contracts.experiment import (
    ControlPlaneState,
    ExperimentDefinition,
    ExperimentRecord,
    WorkflowState,
)
from ml_engine.io import ExperimentLayout, read_model, write_model

USER = "analyst@corp.example"


def _definition(root: str, experiment_id: str = "exp-1") -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id=experiment_id,
        name=f"experiment {experiment_id}",
        created_by=USER,
        created_at=datetime.now(UTC),
        dataset=DatasetReference(uri="s3://data/curated/x.csv", file_format="csv"),
        target_column="y",
        artifact_prefix=ExperimentLayout.for_experiment(root, experiment_id).base,
    )


def test_a_record_is_three_objects_under_the_experiment_prefix(store, artifact_root):
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    definition = _definition(artifact_root)
    repository.create(definition)
    layout = ExperimentLayout(base=definition.artifact_prefix)

    assert store.exists(layout.definition)
    assert store.exists(layout.control_state)
    assert not store.exists(layout.workflow_state)  # written by the workflow, not by us
    assert layout.definition.startswith(artifact_root)


def test_the_workflow_state_is_what_the_state_machine_writes(store, artifact_root):
    """A record composed from what Step Functions puts in S3, with no other source of truth."""
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    definition = _definition(artifact_root)
    repository.create(definition)
    layout = ExperimentLayout(base=definition.artifact_prefix)

    write_model(
        store,
        layout.workflow_state,
        WorkflowState(
            experiment_id="exp-1",
            updated_at=datetime.now(UTC) + timedelta(seconds=1),
            status=ExperimentStatus.COMPLETED,
            current_stage="completed",
            best_model="xgboost",
            best_score=0.83,
            primary_metric="roc_auc",
        ),
    )
    record = repository.get("exp-1")
    assert record.status is ExperimentStatus.COMPLETED
    assert record.best_model == "xgboost"
    assert record.best_score == pytest.approx(0.83)


def test_recording_an_execution_arn_does_not_take_the_status_back(store, artifact_root):
    """The bug this guards: a bookkeeping write must not undo a stage the workflow reported."""
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    definition = _definition(artifact_root)
    repository.create(definition)
    layout = ExperimentLayout(base=definition.artifact_prefix)

    write_model(
        store,
        layout.workflow_state,
        WorkflowState(
            experiment_id="exp-1",
            updated_at=datetime.now(UTC),
            status=ExperimentStatus.FEATURE_REVIEW,
            current_stage="feature_review",
        ),
    )
    record = repository.update("exp-1", eda_execution_arn="arn:aws:states:::execution:x")
    assert record.status is ExperimentStatus.FEATURE_REVIEW
    assert record.eda_execution_arn == "arn:aws:states:::execution:x"


def test_the_control_plane_reclaims_the_status_when_it_sets_one(store, artifact_root):
    """Starting a new run must not be masked by the previous run's terminal state."""
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    definition = _definition(artifact_root)
    repository.create(definition)
    layout = ExperimentLayout(base=definition.artifact_prefix)
    write_model(
        store,
        layout.workflow_state,
        WorkflowState(
            experiment_id="exp-1",
            updated_at=datetime.now(UTC),
            status=ExperimentStatus.COMPLETED,
            current_stage="completed",
            best_model="xgboost",
        ),
    )
    record = repository.update("exp-1", status=ExperimentStatus.READY_FOR_TRAINING)
    assert record.status is ExperimentStatus.READY_FOR_TRAINING


def test_listing_reads_one_prefix_per_experiment(store, artifact_root):
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    for index in range(3):
        repository.create(_definition(artifact_root, f"exp-{index}"))
    records = repository.list()
    assert {record.experiment_id for record in records} == {"exp-0", "exp-1", "exp-2"}
    assert [record.created_at for record in records] == sorted(
        (record.created_at for record in records), reverse=True
    )


def test_a_prefix_without_a_definition_is_ignored(store, artifact_root):
    """Stray objects under the namespace must not break the listing."""
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    repository.create(_definition(artifact_root))
    store.write_bytes(f"{artifact_root}/ml-factory/experiments/not-an-experiment/junk.txt", b"x")
    assert [record.experiment_id for record in repository.list()] == ["exp-1"]


def test_soft_delete_hides_the_record_but_keeps_the_artifacts(store, artifact_root):
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    definition = _definition(artifact_root)
    repository.create(definition)
    repository.delete("exp-1")

    with pytest.raises(NotFoundError):
        repository.get("exp-1")
    assert repository.list() == []
    assert store.exists(ExperimentLayout(base=definition.artifact_prefix).definition)


def test_an_unknown_experiment_is_not_found(store, artifact_root):
    repository = ObjectStoreExperimentRepository(store, artifact_root)
    with pytest.raises(NotFoundError):
        repository.get("exp-missing")


def test_records_can_be_read_from_another_artifact_bucket(store, artifact_root):
    """What the artifact browser does: read experiments produced by another environment."""
    other_root = "s3://another-teams-artifacts"
    other = ObjectStoreExperimentRepository(store, other_root)
    other.create(_definition(other_root, "exp-elsewhere"))

    repository = ObjectStoreExperimentRepository(store, artifact_root)
    assert repository.list() == []
    assert [r.experiment_id for r in repository.list(root=other_root)] == ["exp-elsewhere"]
    assert repository.get("exp-elsewhere", root=other_root).name == "experiment exp-elsewhere"


def test_the_service_refuses_an_artifact_root_that_is_not_configured(container):
    with pytest.raises(ValidationError, match="not configured"):
        container.experiments.list(root="s3://somewhere-the-operator-did-not-approve")


def test_the_service_browses_a_configured_additional_root(settings, store, dataset_csv):
    from backend.container import build_container
    from tests.support.inline_orchestrator import InlineOrchestrator

    other_root = "s3://another-teams-artifacts"
    ObjectStoreExperimentRepository(store, other_root).create(_definition(other_root, "exp-old"))

    browsing = settings.model_copy(update={"additional_artifact_roots": [other_root]})
    container = build_container(
        browsing,
        store=store,
        repository=ObjectStoreExperimentRepository(store, browsing.artifact_root),
        orchestrator=InlineOrchestrator(store),
    )
    try:
        container.experiments.create(
            CreateExperimentRequest(name="new", dataset_uri=dataset_csv, target_column="churned"),
            USER,
        )
        assert [r.experiment_id for r in container.experiments.list()] != []
        assert [r.experiment_id for r in container.experiments.list(root=other_root)] == ["exp-old"]
        assert container.experiments.artifact_roots() == [browsing.artifact_root, other_root]
    finally:
        container.shutdown()


def test_a_record_written_by_the_workflow_alone_still_composes(store, artifact_root):
    """If the control-plane document were ever lost, the definition plus workflow state suffice."""
    definition = _definition(artifact_root)
    layout = ExperimentLayout(base=definition.artifact_prefix)
    write_model(store, layout.definition, definition)
    write_model(
        store,
        layout.workflow_state,
        WorkflowState(
            experiment_id="exp-1",
            updated_at=datetime.now(UTC),
            status=ExperimentStatus.TRAINING,
            current_stage="training",
        ),
    )
    record = ObjectStoreExperimentRepository(store, artifact_root).get("exp-1")
    assert record.status is ExperimentStatus.TRAINING
    assert record.name == "experiment exp-1"


def test_the_composed_record_matches_the_documents(store, artifact_root):
    definition = _definition(artifact_root)
    layout = ExperimentLayout(base=definition.artifact_prefix)
    write_model(store, layout.definition, definition)
    control = ControlPlaneState(
        experiment_id="exp-1",
        updated_at=datetime.now(UTC),
        status=ExperimentStatus.READY_FOR_TRAINING,
        status_updated_at=datetime.now(UTC),
        requested_models=["logistic_regression", "xgboost"],
    )
    write_model(store, layout.control_state, control)

    record = ExperimentRecord.compose(
        definition, read_model(store, layout.control_state, ControlPlaneState), None
    )
    assert record.requested_models == ["logistic_regression", "xgboost"]
    assert record.status is ExperimentStatus.READY_FOR_TRAINING
    assert record.created_by == USER
