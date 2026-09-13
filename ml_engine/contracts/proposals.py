"""Proposed derived features: what may be proposed, and what was decided about it.

A proposal is a *specification*, never code. The operation vocabulary below is the whole
vocabulary — anything outside it cannot be expressed — so a proposal can be checked before it
runs, reproduced exactly from the frozen experiment configuration, and read six months later by
someone asking where a column came from.

Every operation here is row-wise and stateless: the value of a derived column depends only on
the values in its own row. Nothing is fitted, so nothing can leak across the train/validation
split. Stateful operations (group aggregates, quantile bins, target encoding) are deliberately
absent; they need fold-aware fitting and belong with the preprocessing pipelines, not here.

Inputs must be columns of the source dataset. A proposal may not consume another proposal's
output: that keeps the target-reference check a single lookup rather than a graph walk, and one
derivation step is enough for the features this vocabulary can express.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from ml_engine.contracts.common import StrictModel

FEATURE_PROPOSAL_SCHEMA_VERSION = "1.0"

#: Proposals offered for one experiment. A model asked for domain features should return a
#: handful of considered ones, not an exhaustive sweep of column pairs.
MAX_PROPOSALS = 10


class FeatureOp(StrEnum):
    RATIO = "ratio"
    DIFFERENCE = "difference"
    DATE_DIFFERENCE = "date_difference"
    MAP_CATEGORIES = "map_categories"
    IS_MISSING = "is_missing"


class ZeroDenominatorPolicy(StrEnum):
    """What a ratio does where the denominator is zero. Never silently infinite."""

    NULL = "null"
    ZERO = "zero"


class DateUnit(StrEnum):
    DAYS = "days"
    MONTHS = "months"
    YEARS = "years"


class _BaseProposal(StrictModel):
    name: str = Field(description="Name of the derived column. Must not already exist.")
    rationale: str = Field(description="Why this feature should help. Shown to the reviewer.")
    available_at_prediction_time: bool = Field(
        description=(
            "Whether every input is known when a prediction would actually be made. The "
            "proposer states its assumption; a human confirms it. No dictionary can settle "
            "this, and getting it wrong is how a good validation score fails in production."
        )
    )


class RatioProposal(_BaseProposal):
    """``numerator / denominator`` — per-unit, per-month, per-capita quantities."""

    op: Literal[FeatureOp.RATIO] = FeatureOp.RATIO
    numerator: str
    denominator: str
    on_zero_denominator: ZeroDenominatorPolicy = ZeroDenominatorPolicy.NULL


class DifferenceProposal(_BaseProposal):
    """``left - right`` — gaps, headroom, unused allowance."""

    op: Literal[FeatureOp.DIFFERENCE] = FeatureOp.DIFFERENCE
    left: str
    right: str


class DateDifferenceProposal(_BaseProposal):
    """Elapsed time between two date columns of the dataset.

    Both endpoints are columns. There is deliberately no "difference from today": that would
    produce a different value on every run and make the experiment irreproducible.
    """

    op: Literal[FeatureOp.DATE_DIFFERENCE] = FeatureOp.DATE_DIFFERENCE
    start: str
    end: str
    unit: DateUnit = DateUnit.DAYS


class MapCategoriesProposal(_BaseProposal):
    """Group category values by documented meaning — country to region, code to family.

    This is the operation a data dictionary actually unlocks: the mapping comes from what the
    values *mean*, which no statistic in the profile can supply.
    """

    op: Literal[FeatureOp.MAP_CATEGORIES] = FeatureOp.MAP_CATEGORIES
    column: str
    mapping: dict[str, str]
    default: str | None = Field(
        default=None,
        description="Group for values absent from the mapping; null leaves them missing.",
    )


class IsMissingProposal(_BaseProposal):
    """Flag whether a column is missing.

    Missingness that predicts the outcome is already a leakage finding; this turns the pattern
    into an explicit feature, so the derived column is screened like any other.
    """

    op: Literal[FeatureOp.IS_MISSING] = FeatureOp.IS_MISSING
    column: str


FeatureProposal = Annotated[
    RatioProposal
    | DifferenceProposal
    | DateDifferenceProposal
    | MapCategoriesProposal
    | IsMissingProposal,
    Field(discriminator="op"),
]


class ProposalRejection(StrEnum):
    INVALID_NAME = "invalid_name"
    DUPLICATE_NAME = "duplicate_name"
    NAME_ALREADY_IN_DATASET = "name_already_in_dataset"
    UNKNOWN_COLUMN = "unknown_column"
    TARGET_REFERENCE = "target_reference"
    WRONG_COLUMN_TYPE = "wrong_column_type"
    EMPTY_MAPPING = "empty_mapping"
    LIMIT_EXCEEDED = "limit_exceeded"


class RejectedProposal(StrictModel):
    """A proposal that will not be computed, and why. Shown to the reviewer, never silent."""

    name: str
    op: FeatureOp
    reason: ProposalRejection
    message: str


class DerivedFeature(StrictModel):
    """One accepted proposal as applied, recorded so the column can be rebuilt exactly."""

    name: str
    op: FeatureOp
    source_columns: list[str]
    rationale: str
    available_at_prediction_time: bool
    null_count: int
    approved_by: str | None = None
    approved_at: datetime | None = None


class FeatureProposalReport(StrictModel):
    """Everything proposed for one experiment and what happened to it."""

    schema_version: str = FEATURE_PROPOSAL_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    proposed_by: str | None = Field(
        default=None,
        description="Model id that produced the proposals; null when they were written by hand.",
    )
    accepted: list[FeatureProposal] = Field(default_factory=list)
    rejected: list[RejectedProposal] = Field(default_factory=list)
