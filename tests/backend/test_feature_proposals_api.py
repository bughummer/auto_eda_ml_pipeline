"""Derived-feature proposals through the control plane.

The model suggests, validation decides what may be offered, a person decides what is used, and
only that last decision reaches the frozen configuration. Each of those boundaries is tested
here, with a fake model whose answers are exactly the ones a real one gets wrong.
"""

import json

import pytest

from backend.config import Settings
from backend.container import build_container
from backend.errors import ArtifactNotReadyError, FeatureDisabledError, ValidationError
from backend.repositories import ObjectStoreExperimentRepository
from backend.schemas.experiments import CreateExperimentRequest, UpdateFeatureSelectionRequest
from backend.schemas.proposals import ApproveProposalsRequest, ProposeFeaturesRequest
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.proposals import ProposalRejection
from ml_engine.io import ExperimentLayout, read_model
from tests.support.inline_orchestrator import InlineOrchestrator

USER = "reviewer"

GOOD_PROPOSAL = {
    "name": "charges_per_tenure_month",
    "op": "ratio",
    "numerator": "monthly_charges",
    "denominator": "tenure_months",
    "rationale": "Spend normalised by relationship length.",
    "available_at_prediction_time": True,
}
LEAKY_PROPOSAL = {
    "name": "target_restated",
    "op": "difference",
    "left": "churned",
    "right": "tenure_months",
    "rationale": "Looks innocent.",
    "available_at_prediction_time": True,
}
HALLUCINATED_PROPOSAL = {
    "name": "spend_per_visit",
    "op": "ratio",
    "numerator": "monthly_charges",
    "denominator": "visit_count",
    "rationale": "There is no visit_count column.",
    "available_at_prediction_time": True,
}
UNSUPPORTED_PROPOSAL = {
    "name": "charges_squared",
    "op": "polynomial",
    "degree": 2,
    "rationale": "Not in the vocabulary.",
    "available_at_prediction_time": True,
}


class FakeModel:
    """Returns whatever the test tells it to, and records the prompt it was given."""

    model_id = "fake.model"

    def __init__(self, proposals: list[dict] | str) -> None:
        self._proposals = proposals
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if isinstance(self._proposals, str):
            return self._proposals
        return json.dumps({"proposals": self._proposals})


@pytest.fixture
def bedrock_container(artifact_root, store):
    settings = Settings(
        artifact_bucket=artifact_root,
        allowed_dataset_prefixes=["s3://ml-factory-test-data"],
        bedrock_enabled=True,
        bedrock_model_id="fake.model",
        eda_state_machine_arn="arn:aws:states:eu-central-1:000000000000:stateMachine:t-eda",
        training_state_machine_arn="arn:aws:states:eu-central-1:000000000000:stateMachine:t-train",
    )
    built = build_container(
        settings,
        store=store,
        repository=ObjectStoreExperimentRepository(store, settings.artifact_root),
        orchestrator=InlineOrchestrator(store),
    )
    yield built
    built.shutdown()


@pytest.fixture
def profiled(bedrock_container, dataset_csv):
    record = bedrock_container.experiments.create(
        CreateExperimentRequest(name="churn", dataset_uri=dataset_csv, target_column="churned"),
        USER,
    )
    bedrock_container.orchestrator.wait_for_idle(timeout=300)
    return record


def _with_model(container, proposals) -> FakeModel:
    model = FakeModel(proposals)
    container.proposals._client = model
    return model


def test_the_prompt_carries_the_vocabulary_and_the_deterministic_context(
    bedrock_container, profiled
):
    model = _with_model(bedrock_container, [GOOD_PROPOSAL])

    bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    assert "NEVER use the target column as an input" in model.system_prompt
    assert "map_categories" in model.system_prompt
    assert "monthly_charges" in model.user_prompt, "the profile was supplied as evidence"


def test_a_usable_proposal_is_offered_for_review(bedrock_container, profiled):
    _with_model(bedrock_container, [GOOD_PROPOSAL])

    report = bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    assert [c.name for c in report.candidates] == ["charges_per_tenure_month"]
    assert report.rejected == []
    assert report.approved_names == [], "proposing is not approving"
    assert report.proposed_by == "fake.model"


def test_a_proposal_reading_the_target_never_reaches_the_reviewer(bedrock_container, profiled):
    _with_model(bedrock_container, [GOOD_PROPOSAL, LEAKY_PROPOSAL])

    report = bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    assert [c.name for c in report.candidates] == ["charges_per_tenure_month"]
    refusal = next(r for r in report.rejected if r.name == "target_restated")
    assert refusal.reason is ProposalRejection.TARGET_REFERENCE


def test_an_invented_column_is_refused(bedrock_container, profiled):
    _with_model(bedrock_container, [HALLUCINATED_PROPOSAL])

    report = bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    assert report.candidates == []
    assert report.rejected[0].reason is ProposalRejection.UNKNOWN_COLUMN


def test_an_operation_outside_the_vocabulary_is_refused(bedrock_container, profiled):
    _with_model(bedrock_container, [UNSUPPORTED_PROPOSAL, GOOD_PROPOSAL])

    report = bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    assert [c.name for c in report.candidates] == ["charges_per_tenure_month"]
    refusal = next(r for r in report.rejected if r.name == "charges_squared")
    assert refusal.reason is ProposalRejection.MALFORMED_SPECIFICATION
    assert refusal.op is None


def test_prose_around_the_json_is_tolerated(bedrock_container, profiled):
    _with_model(
        bedrock_container,
        "Here are my suggestions:\n```json\n"
        + json.dumps({"proposals": [GOOD_PROPOSAL]})
        + "\n```",
    )

    report = bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    assert [c.name for c in report.candidates] == ["charges_per_tenure_month"]


def test_a_response_that_is_not_json_is_an_upstream_error(bedrock_container, profiled):
    from backend.errors import UpstreamError

    _with_model(bedrock_container, "I would suggest a few ratios.")

    with pytest.raises(UpstreamError):
        bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())


def test_proposals_are_unavailable_without_bedrock(container, dataset_csv):
    record = container.experiments.create(
        CreateExperimentRequest(name="x", dataset_uri=dataset_csv, target_column="churned"), USER
    )
    with pytest.raises(FeatureDisabledError):
        container.proposals.generate(record.experiment_id, ProposeFeaturesRequest())


def test_reading_before_proposing_says_so(bedrock_container, profiled):
    with pytest.raises(ArtifactNotReadyError):
        bedrock_container.proposals.get(profiled.experiment_id)


# --- approval -------------------------------------------------------------------


def test_approving_records_exactly_what_was_chosen(bedrock_container, profiled):
    _with_model(bedrock_container, [GOOD_PROPOSAL])
    bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    report = bedrock_container.proposals.approve(
        profiled.experiment_id,
        ApproveProposalsRequest(approved_names=["charges_per_tenure_month"]),
        USER,
    )

    assert report.approved_names == ["charges_per_tenure_month"]
    assert [p.name for p in report.approved] == ["charges_per_tenure_month"]
    assert report.decided_by == USER
    assert report.decided_at is not None


def test_approving_nothing_is_a_valid_answer(bedrock_container, profiled):
    _with_model(bedrock_container, [GOOD_PROPOSAL])
    bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    report = bedrock_container.proposals.approve(
        profiled.experiment_id, ApproveProposalsRequest(approved_names=[]), USER
    )

    assert report.approved == []
    assert report.decided_at is not None, "deciding against is still a decision"


def test_a_name_that_was_never_offered_cannot_be_approved(bedrock_container, profiled):
    """Otherwise the approval endpoint would be a way to inject an unvalidated feature."""
    _with_model(bedrock_container, [GOOD_PROPOSAL])
    bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    with pytest.raises(ValidationError, match="offered"):
        bedrock_container.proposals.approve(
            profiled.experiment_id,
            ApproveProposalsRequest(approved_names=["target_restated"]),
            USER,
        )


def test_a_refused_proposal_cannot_be_approved(bedrock_container, profiled):
    _with_model(bedrock_container, [LEAKY_PROPOSAL])
    bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    with pytest.raises(ValidationError):
        bedrock_container.proposals.approve(
            profiled.experiment_id,
            ApproveProposalsRequest(approved_names=["target_restated"]),
            USER,
        )


# --- what reaches the run -------------------------------------------------------


def _start_training(container, experiment_id):
    from backend.schemas.experiments import TrainingConfigRequest

    review = container.experiments.feature_review(experiment_id)
    container.experiments.save_features(
        experiment_id,
        UpdateFeatureSelectionRequest(
            selected_features=[f.feature for f in review.features if f.selected][:4]
        ),
        USER,
    )
    container.experiments.start_training(
        experiment_id, TrainingConfigRequest(models=["logistic_regression"]), USER
    )


def test_only_approved_proposals_are_frozen_into_the_configuration(
    bedrock_container, profiled, store
):
    _with_model(bedrock_container, [GOOD_PROPOSAL])
    bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())
    bedrock_container.proposals.approve(
        profiled.experiment_id,
        ApproveProposalsRequest(approved_names=["charges_per_tenure_month"]),
        USER,
    )

    _start_training(bedrock_container, profiled.experiment_id)

    layout = ExperimentLayout(base=profiled.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert [p.name for p in config.derived_features] == ["charges_per_tenure_month"]


def test_an_unapproved_proposal_leaves_no_trace_in_the_run(bedrock_container, profiled, store):
    _with_model(bedrock_container, [GOOD_PROPOSAL])
    bedrock_container.proposals.generate(profiled.experiment_id, ProposeFeaturesRequest())

    _start_training(bedrock_container, profiled.experiment_id)

    layout = ExperimentLayout(base=profiled.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert config.derived_features == []


def test_an_experiment_that_never_saw_a_proposal_is_unaffected(bedrock_container, profiled, store):
    _start_training(bedrock_container, profiled.experiment_id)

    layout = ExperimentLayout(base=profiled.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert config.derived_features == []
