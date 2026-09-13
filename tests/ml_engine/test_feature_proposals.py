"""Proposed derived features: what is refused, and what each operation computes.

The refusals matter more than the arithmetic. A proposer will eventually suggest a feature
built on the target, or on a column that does not exist, and the platform must decline and say
so rather than compute it.
"""

import numpy as np
import pandas as pd
import pytest

from ml_engine.contracts.proposals import (
    MAX_PROPOSALS,
    DateDifferenceProposal,
    DateUnit,
    DifferenceProposal,
    FeatureOp,
    IsMissingProposal,
    MapCategoriesProposal,
    ProposalRejection,
    RatioProposal,
    ZeroDenominatorPolicy,
)
from ml_engine.features import (
    apply_proposals,
    derivation_warnings,
    source_columns,
    validate_proposals,
)

TARGET = "churned"


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "monthly_charges": [70.0, 40.0, 55.0, np.nan],
            "tenure_months": [10.0, 0.0, 5.0, 2.0],
            "credit_limit": [1000.0, 500.0, 800.0, 200.0],
            "balance": [400.0, 500.0, 100.0, 50.0],
            "signup_date": pd.to_datetime(["2021-01-01", "2021-06-01", "2022-01-01", "2022-03-01"]),
            "first_order_date": pd.to_datetime(
                ["2021-01-31", "2021-07-01", "2022-01-11", "2022-03-02"]
            ),
            "country": ["DE", "FR", "US", "BR"],
            "is_business": [True, False, True, False],
            "notes": ["a", "b", None, "d"],
            TARGET: ["no", "yes", "no", "yes"],
        }
    )


def _ratio(
    name="charges_per_month", numerator="monthly_charges", denominator="tenure_months", **kw
):
    return RatioProposal(
        name=name,
        numerator=numerator,
        denominator=denominator,
        rationale="spend normalised by relationship length",
        available_at_prediction_time=True,
        **kw,
    )


# --- refusals -------------------------------------------------------------------


def test_a_feature_built_on_the_target_is_refused(frame):
    accepted, rejected = validate_proposals(
        [_ratio(numerator=TARGET, denominator="tenure_months")], frame, target_column=TARGET
    )

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.TARGET_REFERENCE
    assert TARGET in rejected[0].message


def test_the_target_is_refused_in_every_input_position(frame):
    proposals = [
        _ratio(name="a", denominator=TARGET),
        DifferenceProposal(
            name="b",
            left=TARGET,
            right="balance",
            rationale="r",
            available_at_prediction_time=True,
        ),
        IsMissingProposal(
            name="c", column=TARGET, rationale="r", available_at_prediction_time=True
        ),
    ]

    accepted, rejected = validate_proposals(proposals, frame, target_column=TARGET)

    assert accepted == []
    assert {r.reason for r in rejected} == {ProposalRejection.TARGET_REFERENCE}


def test_an_unknown_column_is_refused(frame):
    accepted, rejected = validate_proposals(
        [_ratio(denominator="lifetime_value")], frame, target_column=TARGET
    )

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.UNKNOWN_COLUMN


def test_arithmetic_on_a_non_numeric_column_is_refused(frame):
    accepted, rejected = validate_proposals(
        [_ratio(numerator="country")], frame, target_column=TARGET
    )

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.WRONG_COLUMN_TYPE


def test_a_boolean_is_not_treated_as_a_measurement(frame):
    """Booleans are numeric to pandas. Dividing by one is not arithmetic anybody wanted."""
    accepted, rejected = validate_proposals(
        [_ratio(denominator="is_business")], frame, target_column=TARGET
    )

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.WRONG_COLUMN_TYPE


def test_a_date_difference_over_non_dates_is_refused(frame):
    proposal = DateDifferenceProposal(
        name="days_to_order",
        start="balance",
        end="first_order_date",
        rationale="r",
        available_at_prediction_time=True,
    )

    accepted, rejected = validate_proposals([proposal], frame, target_column=TARGET)

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.WRONG_COLUMN_TYPE


def test_mapping_a_numeric_column_is_refused(frame):
    proposal = MapCategoriesProposal(
        name="balance_group",
        column="balance",
        mapping={"400.0": "high"},
        rationale="r",
        available_at_prediction_time=True,
    )

    accepted, rejected = validate_proposals([proposal], frame, target_column=TARGET)

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.WRONG_COLUMN_TYPE


def test_an_empty_mapping_is_refused(frame):
    proposal = MapCategoriesProposal(
        name="region",
        column="country",
        mapping={},
        rationale="r",
        available_at_prediction_time=True,
    )

    accepted, rejected = validate_proposals([proposal], frame, target_column=TARGET)

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.EMPTY_MAPPING


def test_a_name_that_already_exists_is_refused(frame):
    accepted, rejected = validate_proposals([_ratio(name="balance")], frame, target_column=TARGET)

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.NAME_ALREADY_IN_DATASET


def test_two_proposals_cannot_claim_the_same_name(frame):
    accepted, rejected = validate_proposals(
        [_ratio(), _ratio(numerator="balance")], frame, target_column=TARGET
    )

    assert len(accepted) == 1
    assert rejected[0].reason is ProposalRejection.DUPLICATE_NAME


@pytest.mark.parametrize("name", ["", "2_fast", "spend rate", "spend-rate", "drop table"])
def test_an_unusable_column_name_is_refused(frame, name):
    accepted, rejected = validate_proposals([_ratio(name=name)], frame, target_column=TARGET)

    assert accepted == []
    assert rejected[0].reason is ProposalRejection.INVALID_NAME


def test_the_proposal_count_is_capped(frame):
    proposals = [_ratio(name=f"feature_{i}") for i in range(MAX_PROPOSALS + 3)]

    accepted, rejected = validate_proposals(proposals, frame, target_column=TARGET)

    assert len(accepted) == MAX_PROPOSALS
    assert len(rejected) == 3
    assert {r.reason for r in rejected} == {ProposalRejection.LIMIT_EXCEEDED}


def test_one_bad_proposal_does_not_discard_the_good_ones(frame):
    accepted, rejected = validate_proposals(
        [_ratio(), _ratio(name="leak", numerator=TARGET)], frame, target_column=TARGET
    )

    assert [p.name for p in accepted] == ["charges_per_month"]
    assert [r.name for r in rejected] == ["leak"]


# --- computation ----------------------------------------------------------------


def test_ratio_leaves_a_zero_denominator_null_by_default(frame):
    result, derived = apply_proposals(frame, [_ratio()])

    assert result["charges_per_month"].tolist()[0] == pytest.approx(7.0)
    assert pd.isna(result["charges_per_month"].iloc[1]), "1/0 must not become infinity"
    assert derived[0].null_count == 2, "the zero denominator and the missing numerator"


def test_ratio_can_be_asked_to_treat_a_zero_denominator_as_zero(frame):
    proposal = _ratio(on_zero_denominator=ZeroDenominatorPolicy.ZERO)

    result, _ = apply_proposals(frame, [proposal])

    assert result["charges_per_month"].iloc[1] == 0.0
    assert pd.isna(result["charges_per_month"].iloc[3]), "a missing numerator stays missing"


def test_difference_computes_headroom(frame):
    proposal = DifferenceProposal(
        name="available_credit",
        left="credit_limit",
        right="balance",
        rationale="unused allowance",
        available_at_prediction_time=True,
    )

    result, _ = apply_proposals(frame, [proposal])

    assert result["available_credit"].tolist() == [600.0, 0.0, 700.0, 150.0]


@pytest.mark.parametrize(
    ("unit", "expected_first"),
    [
        (DateUnit.DAYS, 30.0),
        (DateUnit.MONTHS, 30.0 / (365.25 / 12)),
        (DateUnit.YEARS, 30.0 / 365.25),
    ],
)
def test_date_difference_converts_units(frame, unit, expected_first):
    proposal = DateDifferenceProposal(
        name="days_to_first_order",
        start="signup_date",
        end="first_order_date",
        unit=unit,
        rationale="activation delay",
        available_at_prediction_time=True,
    )

    result, _ = apply_proposals(frame, [proposal])

    assert result["days_to_first_order"].iloc[0] == pytest.approx(expected_first)


def test_date_difference_accepts_dates_stored_as_strings(frame):
    """A CSV loads dates as text. The profiler recognises them; so must this."""
    as_text = frame.assign(
        signup_date=frame["signup_date"].dt.strftime("%Y-%m-%d"),
        first_order_date=frame["first_order_date"].dt.strftime("%Y-%m-%d"),
    )
    proposal = DateDifferenceProposal(
        name="days_to_first_order",
        start="signup_date",
        end="first_order_date",
        rationale="activation delay",
        available_at_prediction_time=True,
    )

    accepted, rejected = validate_proposals([proposal], as_text, target_column=TARGET)
    result, _ = apply_proposals(as_text, accepted)

    assert rejected == []
    assert result["days_to_first_order"].iloc[0] == pytest.approx(30.0)


def test_map_categories_groups_documented_values(frame):
    proposal = MapCategoriesProposal(
        name="region",
        column="country",
        mapping={"DE": "EU", "FR": "EU", "US": "NA"},
        default="other",
        rationale="country to sales region, from the data dictionary",
        available_at_prediction_time=True,
    )

    result, _ = apply_proposals(frame, [proposal])

    assert result["region"].tolist() == ["EU", "EU", "NA", "other"]


def test_map_categories_without_a_default_leaves_unknown_values_missing(frame):
    proposal = MapCategoriesProposal(
        name="region",
        column="country",
        mapping={"DE": "EU"},
        rationale="r",
        available_at_prediction_time=True,
    )

    result, _ = apply_proposals(frame, [proposal])

    assert result["region"].iloc[0] == "EU"
    assert result["region"].isna().sum() == 3


def test_is_missing_flags_the_gaps(frame):
    proposal = IsMissingProposal(
        name="notes_missing",
        column="notes",
        rationale="an absent note may itself be informative",
        available_at_prediction_time=True,
    )

    result, derived = apply_proposals(frame, [proposal])

    assert result["notes_missing"].tolist() == [False, False, True, False]
    assert derived[0].op is FeatureOp.IS_MISSING


# --- bookkeeping ----------------------------------------------------------------


def test_the_source_frame_is_never_mutated(frame):
    before = frame.copy()

    apply_proposals(frame, [_ratio()])

    pd.testing.assert_frame_equal(frame, before)


def test_each_derived_feature_records_where_it_came_from(frame):
    _, derived = apply_proposals(frame, [_ratio()])

    assert derived[0].source_columns == ["monthly_charges", "tenure_months"]
    assert derived[0].rationale.startswith("spend")
    assert derived[0].available_at_prediction_time is True


def test_source_columns_covers_every_operation(frame):
    proposals = [
        _ratio(),
        DifferenceProposal(
            name="d",
            left="balance",
            right="credit_limit",
            rationale="r",
            available_at_prediction_time=True,
        ),
        DateDifferenceProposal(
            name="e",
            start="signup_date",
            end="first_order_date",
            rationale="r",
            available_at_prediction_time=True,
        ),
        MapCategoriesProposal(
            name="f",
            column="country",
            mapping={"DE": "EU"},
            rationale="r",
            available_at_prediction_time=True,
        ),
        IsMissingProposal(
            name="g", column="notes", rationale="r", available_at_prediction_time=True
        ),
    ]

    for proposal in proposals:
        columns = source_columns(proposal)
        assert columns and all(column in frame.columns for column in columns)


def test_a_feature_unavailable_at_prediction_time_is_warned_about(frame):
    proposal = _ratio()
    proposal.available_at_prediction_time = False

    _, derived = apply_proposals(frame, [proposal])
    warnings = derivation_warnings(derived, row_count=len(frame))

    assert [w.rule for w in warnings] == ["derived_feature_unavailable_at_prediction_time"]
    assert warnings[0].column == "charges_per_month"


def test_an_all_null_derived_column_is_warned_about(frame):
    empty = frame.assign(tenure_months=0.0)

    _, derived = apply_proposals(empty, [_ratio()])
    warnings = derivation_warnings(derived, row_count=len(empty))

    assert [w.rule for w in warnings] == ["derived_feature_all_null"]
