"""Derived features inside the preparation job.

The specifications live in the frozen configuration, so preparation is where they turn into
columns. Three things have to hold: an approved feature reaches the model without having to be
listed twice, a specification that stopped being valid is refused loudly rather than silently,
and the columns derivation invents are screened for leakage like any other.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from jobs._common.runtime import JobError
from jobs.preprocessing.main import run_preparation
from ml_engine.contracts.common import (
    ClassWeighting,
    LeakageRiskLevel,
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
from ml_engine.contracts.proposals import (
    DifferenceProposal,
    IsMissingProposal,
    ProposalRejection,
    RatioProposal,
)
from ml_engine.io import read_parquet, write_model

EXPERIMENT_ID = "exp-test"
BASE_FEATURES = ["monthly_charges", "tenure_months"]


@pytest.fixture
def derived_dataset(store) -> str:
    """A frame with everything the derived-feature tests need, including a deliberate leak."""
    rows = 200
    frame = pd.DataFrame(
        {
            "monthly_charges": [40.0 + (i % 50) for i in range(rows)],
            "tenure_months": [float(i % 24) for i in range(rows)],
            # A constant column, for a proposal that is refused before types are checked.
            "zeros": [0] * rows,
            # Two ordinary-looking numeric columns whose difference happens to be the
            # outcome. Neither is binary, so neither is suspicious on its own.
            "score_after": [(i % 5) + (i % 2) for i in range(rows)],
            "score_before": [i % 5 for i in range(rows)],
            "notes": [None if i % 5 == 0 else f"note {i}" for i in range(rows)],
            "churned": [i % 2 for i in range(rows)],
        }
    )
    uri = "s3://ml-factory-test-data/curated/derived.csv"
    store.write_bytes(uri, frame.to_csv(index=False).encode())
    return uri


def _config(dataset_uri: str, layout, proposals, *, selected=None) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=EXPERIMENT_ID,
        name="derived features",
        created_at=datetime.now(UTC),
        dataset=DatasetReference(uri=dataset_uri, file_format="csv"),
        target_column="churned",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        requested_problem_type=RequestedProblemType.AUTO,
        primary_metric="roc_auc",
        feature_selection=FeatureSelection(selected_features=selected or list(BASE_FEATURES)),
        derived_features=proposals,
        split=SplitConfig(),
        preprocessing=PreprocessingConfig(),
        models=[ModelSpec(name="logistic_regression")],
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


def _ratio() -> RatioProposal:
    return RatioProposal(
        name="charges_per_tenure_month",
        numerator="monthly_charges",
        denominator="tenure_months",
        rationale="spend normalised by relationship length",
        available_at_prediction_time=True,
    )


def test_an_approved_feature_is_computed_and_used_without_being_selected_again(
    store, layout, derived_dataset
):
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [_ratio()]))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert [f.name for f in report.derived_features] == ["charges_per_tenure_month"]
    assert "charges_per_tenure_month" in report.requested_features
    assert "charges_per_tenure_month" in report.usable_features
    assert report.rejected_proposals == []


def test_the_derived_column_reaches_the_materialized_folds(store, layout, derived_dataset):
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [_ratio()]))

    run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    train = read_parquet(store, layout.train_dataset)
    assert "charges_per_tenure_month" in train.columns


def test_a_specification_that_reads_the_target_is_refused_during_preparation(
    store, layout, derived_dataset
):
    """Defence in depth: the validator ran when the proposal was approved, and runs again here."""
    leak = DifferenceProposal(
        name="target_in_disguise",
        left="churned",
        right="zeros",
        rationale="looks innocent, is not",
        available_at_prediction_time=True,
    )
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [leak]))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert report.derived_features == []
    assert [r.reason for r in report.rejected_proposals] == [ProposalRejection.TARGET_REFERENCE]
    assert "target_in_disguise" not in report.usable_features
    rejected = [w for w in report.warnings if w.rule == "derived_feature_rejected"]
    assert rejected and rejected[0].column == "target_in_disguise"


def test_a_specification_naming_a_column_that_no_longer_exists_is_refused(
    store, layout, derived_dataset
):
    stale = RatioProposal(
        name="charges_per_visit",
        numerator="monthly_charges",
        denominator="visit_count",
        rationale="the dataset used to have this column",
        available_at_prediction_time=True,
    )
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [stale]))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert [r.reason for r in report.rejected_proposals] == [ProposalRejection.UNKNOWN_COLUMN]


def test_one_refused_specification_does_not_stop_the_others(store, layout, derived_dataset):
    stale = RatioProposal(
        name="charges_per_visit",
        numerator="monthly_charges",
        denominator="visit_count",
        rationale="stale",
        available_at_prediction_time=True,
    )
    write_model(
        store, layout.experiment_config, _config(derived_dataset, layout, [stale, _ratio()])
    )

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert [f.name for f in report.derived_features] == ["charges_per_tenure_month"]
    assert [r.name for r in report.rejected_proposals] == ["charges_per_visit"]


def test_a_derived_column_that_reproduces_the_target_is_caught_by_screening(
    store, layout, derived_dataset
):
    leak = DifferenceProposal(
        name="reconstructed_target",
        left="score_after",
        right="score_before",
        rationale="a plausible-looking difference that happens to be the outcome",
        available_at_prediction_time=True,
    )
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [leak]))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert report.derived_leakage is not None
    findings = report.derived_leakage.findings
    assert any(
        f.feature == "reconstructed_target"
        and f.risk_level
        in {LeakageRiskLevel.CONFIRMED_DUPLICATE, LeakageRiskLevel.POTENTIAL_LEAKAGE}
        for f in findings
    )
    assert any(w.rule.startswith("derived_") for w in report.warnings)


def test_a_derived_feature_is_not_excluded_on_its_own(store, layout, derived_dataset):
    """Screening reports; it never silently drops. The decision stays with the data scientist."""
    leak = DifferenceProposal(
        name="reconstructed_target",
        left="score_after",
        right="score_before",
        rationale="r",
        available_at_prediction_time=True,
    )
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [leak]))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert "reconstructed_target" in report.usable_features


def test_a_feature_unavailable_at_prediction_time_is_warned_about(store, layout, derived_dataset):
    proposal = IsMissingProposal(
        name="notes_missing",
        column="notes",
        rationale="captured only after the outcome is known",
        available_at_prediction_time=False,
    )
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [proposal]))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    rules = {w.rule for w in report.warnings}
    assert "derived_feature_unavailable_at_prediction_time" in rules


def test_an_experiment_without_proposals_is_unchanged(store, layout, derived_dataset):
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, []))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert report.derived_features == []
    assert report.rejected_proposals == []
    assert report.derived_leakage is None
    assert report.usable_features == BASE_FEATURES


def test_every_proposal_being_refused_still_prepares_the_experiment(store, layout, derived_dataset):
    stale = RatioProposal(
        name="charges_per_visit",
        numerator="monthly_charges",
        denominator="visit_count",
        rationale="stale",
        available_at_prediction_time=True,
    )
    write_model(store, layout.experiment_config, _config(derived_dataset, layout, [stale]))

    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert report.usable_features == BASE_FEATURES
    assert report.split.train_row_count > 0


def test_a_proposal_cannot_rescue_an_experiment_with_no_usable_features(
    store, layout, derived_dataset
):
    """A stale proposal plus an unusable selection is still a failed preparation, not a crash."""
    config = _config(derived_dataset, layout, [], selected=["visit_count"])
    write_model(store, layout.experiment_config, config)

    with pytest.raises(JobError) as error:
        run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert error.value.code == "NO_USABLE_FEATURES"
