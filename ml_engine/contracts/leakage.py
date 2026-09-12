"""Deterministic leakage and feature-risk output schema.

Deterministic evidence only. Semantic (business-meaning) leakage lives in
``ml_engine.contracts.reasoning`` and is produced later, by the Bedrock layer.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import (
    LeakageRiskLevel,
    RecommendedAction,
    Severity,
    StrictModel,
)

LEAKAGE_SCHEMA_VERSION = "1.0"


class LeakageFinding(StrictModel):
    feature: str
    rule: str
    severity: Severity
    risk_level: LeakageRiskLevel
    explanation: str
    recommended_action: RecommendedAction
    evidence: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class FeatureRisk(StrictModel):
    """Aggregated per-feature verdict, consumed directly by the feature review UI."""

    feature: str
    risk_level: LeakageRiskLevel
    max_severity: Severity | None
    recommended_action: RecommendedAction
    reasons: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)


class LeakageReport(StrictModel):
    schema_version: str = LEAKAGE_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    target_column: str
    findings: list[LeakageFinding] = Field(default_factory=list)
    feature_risks: list[FeatureRisk] = Field(default_factory=list)
    checks_executed: list[str] = Field(default_factory=list)
    checks_skipped: dict[str, str] = Field(
        default_factory=dict, description="rule id -> reason the check could not run"
    )

    def risk_for(self, feature: str) -> FeatureRisk | None:
        return next((r for r in self.feature_risks if r.feature == feature), None)
