"""Validate and apply proposed derived features.

Two steps, deliberately separate. ``validate_proposals`` decides what may run and explains
every refusal; ``apply_proposals`` computes only what survived. Nothing here evaluates a
string, imports a module or calls back into a model: each operation is a small, tested
function selected by an enum.

Derivation happens on the full frame *before* the split, which is safe precisely because every
operation is row-wise — a derived value depends only on its own row, so no information crosses
the fold boundary. The moment an operation needs to be fitted, it stops belonging here.
"""

import re
from collections.abc import Iterable, Sequence

import pandas as pd

from ml_engine.contracts.common import Severity, WarningCategory
from ml_engine.contracts.proposals import (
    MAX_PROPOSALS,
    DateDifferenceProposal,
    DateUnit,
    DerivedFeature,
    DifferenceProposal,
    FeatureProposal,
    IsMissingProposal,
    MapCategoriesProposal,
    ProposalRejection,
    RatioProposal,
    RejectedProposal,
    ZeroDenominatorPolicy,
)
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.features.columns import ColumnTypes

_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_DAYS_PER = {DateUnit.DAYS: 1.0, DateUnit.MONTHS: 365.25 / 12.0, DateUnit.YEARS: 365.25}


def source_columns(proposal: FeatureProposal) -> tuple[str, ...]:
    """Every dataset column a proposal reads. One place, so validation cannot miss one."""
    match proposal:
        case RatioProposal():
            return (proposal.numerator, proposal.denominator)
        case DifferenceProposal():
            return (proposal.left, proposal.right)
        case DateDifferenceProposal():
            return (proposal.start, proposal.end)
        case MapCategoriesProposal() | IsMissingProposal():
            return (proposal.column,)
    raise ValueError(f"Unhandled proposal type {type(proposal).__name__}")


def validate_proposals(
    proposals: Sequence[FeatureProposal],
    columns: ColumnTypes,
    *,
    target_column: str,
    max_proposals: int = MAX_PROPOSALS,
) -> tuple[list[FeatureProposal], list[RejectedProposal]]:
    """Split proposals into those that may be computed and those that may not.

    Refusals are returned, not raised: a proposer that produces some unusable specifications is
    expected, and the reviewer should see what was discarded and why.

    ``columns`` comes from the EDA report when a proposal is offered and from the dataframe
    when preparation runs, so both checks reach the same verdict.
    """
    accepted: list[FeatureProposal] = []
    rejected: list[RejectedProposal] = []
    claimed: set[str] = set()

    for proposal in proposals:
        if len(accepted) >= max_proposals:
            rejected.append(
                _reject(
                    proposal,
                    ProposalRejection.LIMIT_EXCEEDED,
                    f"At most {max_proposals} derived features are accepted per experiment.",
                )
            )
            continue

        refusal = _check(proposal, columns, claimed, target_column)
        if refusal is not None:
            rejected.append(refusal)
            continue

        claimed.add(proposal.name)
        accepted.append(proposal)

    return accepted, rejected


def apply_proposals(
    frame: pd.DataFrame, proposals: Iterable[FeatureProposal]
) -> tuple[pd.DataFrame, list[DerivedFeature]]:
    """Compute every proposal onto a copy of the frame. Validate first; this does not re-check."""
    result = frame.copy()
    derived: list[DerivedFeature] = []

    for proposal in proposals:
        values = _compute(proposal, result)
        result[proposal.name] = values
        derived.append(
            DerivedFeature(
                name=proposal.name,
                op=proposal.op,
                source_columns=list(source_columns(proposal)),
                rationale=proposal.rationale,
                available_at_prediction_time=proposal.available_at_prediction_time,
                null_count=int(values.isna().sum()),
            )
        )

    return result, derived


def derivation_warnings(derived: Sequence[DerivedFeature], row_count: int) -> list[AnalysisWarning]:
    """Flag derived columns a reviewer should look at twice."""
    warnings: list[AnalysisWarning] = []
    for feature in derived:
        if not feature.available_at_prediction_time:
            warnings.append(
                AnalysisWarning(
                    rule="derived_feature_unavailable_at_prediction_time",
                    category=WarningCategory.FEATURE,
                    severity=Severity.HIGH,
                    message=(
                        f"Derived feature '{feature.name}' was proposed as not available when a "
                        "prediction would be made. It will inflate validation scores without "
                        "helping in production."
                    ),
                    column=feature.name,
                )
            )
        if row_count and feature.null_count == row_count:
            warnings.append(
                AnalysisWarning(
                    rule="derived_feature_all_null",
                    category=WarningCategory.FEATURE,
                    severity=Severity.MEDIUM,
                    message=f"Derived feature '{feature.name}' is null for every row.",
                    column=feature.name,
                )
            )
    return warnings


# --- validation -----------------------------------------------------------------


def _check(
    proposal: FeatureProposal,
    columns: ColumnTypes,
    claimed: set[str],
    target_column: str,
) -> RejectedProposal | None:
    if not _VALID_NAME.match(proposal.name):
        return _reject(
            proposal,
            ProposalRejection.INVALID_NAME,
            f"'{proposal.name}' is not a usable column name.",
        )
    if proposal.name in claimed:
        return _reject(
            proposal,
            ProposalRejection.DUPLICATE_NAME,
            f"Another accepted proposal already defines '{proposal.name}'.",
        )
    if columns.exists(proposal.name):
        return _reject(
            proposal,
            ProposalRejection.NAME_ALREADY_IN_DATASET,
            f"The dataset already has a column named '{proposal.name}'.",
        )

    sources = source_columns(proposal)
    for column in sources:
        if column == target_column:
            return _reject(
                proposal,
                ProposalRejection.TARGET_REFERENCE,
                f"'{proposal.name}' reads the target column '{target_column}'. A feature "
                "derived from the target is leakage by construction.",
            )
        if not columns.exists(column):
            return _reject(
                proposal,
                ProposalRejection.UNKNOWN_COLUMN,
                f"Column '{column}' is not in the dataset.",
            )

    return _check_types(proposal, columns, sources)


def _check_types(
    proposal: FeatureProposal, columns: ColumnTypes, sources: tuple[str, ...]
) -> RejectedProposal | None:
    match proposal:
        case RatioProposal() | DifferenceProposal():
            for column in sources:
                if not columns.is_arithmetic(column):
                    return _reject(
                        proposal,
                        ProposalRejection.WRONG_COLUMN_TYPE,
                        f"Column '{column}' is not numeric, so {proposal.op.value} cannot be "
                        "computed from it.",
                    )
        case DateDifferenceProposal():
            for column in sources:
                if not columns.is_date_like(column):
                    return _reject(
                        proposal,
                        ProposalRejection.WRONG_COLUMN_TYPE,
                        f"Column '{column}' does not hold dates.",
                    )
        case MapCategoriesProposal():
            if not proposal.mapping:
                return _reject(
                    proposal, ProposalRejection.EMPTY_MAPPING, "The category mapping is empty."
                )
            if columns.is_arithmetic(proposal.column):
                return _reject(
                    proposal,
                    ProposalRejection.WRONG_COLUMN_TYPE,
                    f"Column '{proposal.column}' is numeric; map_categories groups category "
                    "values, not measurements.",
                )
        case IsMissingProposal():
            pass
    return None


def _reject(proposal: FeatureProposal, reason: ProposalRejection, message: str) -> RejectedProposal:
    return RejectedProposal(name=proposal.name, op=proposal.op, reason=reason, message=message)


# --- computation ----------------------------------------------------------------


def _compute(proposal: FeatureProposal, frame: pd.DataFrame) -> pd.Series:
    match proposal:
        case RatioProposal():
            return _ratio(frame[proposal.numerator], frame[proposal.denominator], proposal)
        case DifferenceProposal():
            return frame[proposal.left] - frame[proposal.right]
        case DateDifferenceProposal():
            return _date_difference(frame[proposal.start], frame[proposal.end], proposal.unit)
        case MapCategoriesProposal():
            return _map_categories(frame[proposal.column], proposal)
        case IsMissingProposal():
            return frame[proposal.column].isna()
    raise ValueError(f"Unhandled proposal type {type(proposal).__name__}")


def _ratio(numerator: pd.Series, denominator: pd.Series, proposal: RatioProposal) -> pd.Series:
    zeros = denominator == 0
    safe = denominator.where(~zeros)
    values = numerator / safe
    if proposal.on_zero_denominator is ZeroDenominatorPolicy.ZERO:
        values = values.mask(zeros & numerator.notna(), 0.0)
    return values


def _date_difference(start: pd.Series, end: pd.Series, unit: DateUnit) -> pd.Series:
    parsed_start = pd.to_datetime(start, errors="coerce", format="mixed")
    parsed_end = pd.to_datetime(end, errors="coerce", format="mixed")
    days = (parsed_end - parsed_start).dt.total_seconds() / 86400.0
    return days / _DAYS_PER[unit]


def _map_categories(series: pd.Series, proposal: MapCategoriesProposal) -> pd.Series:
    keys = series.where(series.isna(), series.astype("string"))
    mapped = keys.map(proposal.mapping)
    if proposal.default is not None:
        mapped = mapped.where(series.isna() | mapped.notna(), proposal.default)
    return mapped.astype("object")
