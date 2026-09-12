"""Reasoning API schemas."""

from pydantic import Field

from ml_engine.contracts.common import StrictModel


class ReasoningRequest(StrictModel):
    """Business context the deterministic artifacts cannot supply."""

    target_definition: str | None = Field(
        default=None,
        max_length=2000,
        description="What the target actually means in business terms.",
    )
    prediction_timing: str | None = Field(
        default=None,
        max_length=2000,
        description="When the prediction is made relative to the outcome being predicted.",
    )
    question: str | None = Field(
        default=None, max_length=2000, description="An optional focused question."
    )
