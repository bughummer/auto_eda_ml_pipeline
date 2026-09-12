"""The single warning shape used by EDA, leakage, preprocessing, training and evaluation."""

from pydantic import Field

from ml_engine.contracts.common import (
    RecommendedAction,
    Severity,
    StrictModel,
    WarningCategory,
)


class AnalysisWarning(StrictModel):
    """A deterministic finding. Never an instruction — always a recommendation."""

    rule: str = Field(description="Stable rule identifier, e.g. 'high_missingness'.")
    category: WarningCategory
    severity: Severity
    message: str = Field(description="Human-readable explanation of what was observed.")
    column: str | None = Field(default=None, description="Column the finding refers to.")
    recommended_action: RecommendedAction = RecommendedAction.REVIEW
    details: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


def sort_warnings(warnings: list[AnalysisWarning]) -> list[AnalysisWarning]:
    """Most severe first, then by rule then column, so output is deterministic."""
    from ml_engine.contracts.common import SEVERITY_ORDER

    return sorted(
        warnings,
        key=lambda w: (-SEVERITY_ORDER[w.severity], w.rule, w.column or ""),
    )
