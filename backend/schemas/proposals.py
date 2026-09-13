"""Derived-feature proposal API schemas."""

from pydantic import Field

from ml_engine.contracts.common import StrictModel


class ProposeFeaturesRequest(StrictModel):
    """Business context the profile cannot supply. Both fields are optional but both help."""

    target_definition: str | None = Field(
        default=None,
        max_length=2000,
        description="What the target actually means in business terms.",
    )
    prediction_timing: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "When the prediction is made relative to the outcome. This is what lets the model "
            "judge whether a proposed feature would be known at that moment."
        ),
    )


class ApproveProposalsRequest(StrictModel):
    """The candidates a person chose. An empty list means 'none of them', and is respected."""

    approved_names: list[str] = Field(default_factory=list)
