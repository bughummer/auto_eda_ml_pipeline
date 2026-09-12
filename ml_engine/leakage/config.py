"""Thresholds for deterministic leakage detection.

Deliberately conservative: a strong relationship is evidence for *review*, never a verdict.
Only structural equality with the target is ever called a confirmed duplicate.
"""

from pydantic import Field

from ml_engine.contracts.common import StrictModel


class LeakageConfig(StrictModel):
    max_rows: int = Field(
        default=50_000, ge=100, description="Rows sampled for relationship checks."
    )
    max_categorical_levels: int = Field(default=200, ge=2)

    perfect_relationship: float = Field(
        default=0.999,
        ge=0.5,
        le=1.0,
        description="Above this a single feature reproduces the target.",
    )
    strong_relationship: float = Field(default=0.95, ge=0.5, le=1.0)
    regression_perfect_correlation: float = Field(default=0.999, ge=0.5, le=1.0)
    regression_strong_correlation: float = Field(default=0.98, ge=0.5, le=1.0)
    missingness_signal_auc: float = Field(default=0.8, ge=0.5, le=1.0)

    suspicious_name_tokens: list[str] = Field(
        default_factory=lambda: [
            "target",
            "label",
            "outcome",
            "response",
            "ground_truth",
            "y_true",
            "actual",
            "result",
            "final",
            "closed",
            "resolved",
            "settled",
            "repaid",
            "refund",
            "cancelled",
            "canceled",
            "churned",
            "default_flag",
            "post_",
            "_post",
            "after_",
            "_after",
            "future_",
            "prediction",
            "predicted",
            "probability",
            "score_final",
        ]
    )
    post_outcome_tokens: list[str] = Field(
        default_factory=lambda: [
            "closed",
            "resolved",
            "settled",
            "repaid",
            "refund",
            "cancelled",
            "canceled",
            "final",
            "post_",
            "_post",
            "after_",
            "_after",
            "future_",
        ]
    )
