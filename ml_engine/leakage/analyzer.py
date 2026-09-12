"""Runs the deterministic leakage rules and aggregates them into a per-feature verdict."""

from datetime import UTC, datetime

import pandas as pd

from ml_engine.contracts.common import (
    ACTION_ORDER,
    LEAKAGE_RISK_ORDER,
    SEVERITY_ORDER,
    LeakageRiskLevel,
    ProblemType,
    RecommendedAction,
)
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import FeatureRisk, LeakageFinding, LeakageReport
from ml_engine.leakage.config import LeakageConfig
from ml_engine.leakage.rules import RULES, build_context


def analyze_leakage(
    frame: pd.DataFrame,
    *,
    experiment_id: str,
    target_column: str,
    problem_type: ProblemType,
    eda: EdaReport,
    config: LeakageConfig | None = None,
) -> LeakageReport:
    """Execute every registered rule. A failing rule is recorded, never fatal."""
    config = config or LeakageConfig()
    executed: list[str] = []
    skipped: dict[str, str] = {}
    findings: list[LeakageFinding] = []

    if target_column not in frame.columns:
        return LeakageReport(
            experiment_id=experiment_id,
            generated_at=datetime.now(UTC),
            target_column=target_column,
            checks_skipped={r.__name__: "target column is not present in the dataset" for r in RULES},
        )

    context = build_context(
        frame,
        target_column=target_column,
        problem_type=problem_type,
        eda=eda,
        config=config,
    )

    if context.frame.empty:
        return LeakageReport(
            experiment_id=experiment_id,
            generated_at=datetime.now(UTC),
            target_column=target_column,
            checks_skipped={r.__name__: "no rows with a non-missing target" for r in RULES},
        )

    for rule in RULES:
        try:
            findings.extend(rule(context))
            executed.append(rule.__name__)
        except Exception as exc:  # noqa: BLE001 - a broken rule must not fail an experiment
            skipped[rule.__name__] = f"{type(exc).__name__}: {exc}"

    return LeakageReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        target_column=target_column,
        findings=_sorted_findings(findings),
        feature_risks=aggregate_feature_risks(findings),
        checks_executed=executed,
        checks_skipped=skipped,
    )


def _sorted_findings(findings: list[LeakageFinding]) -> list[LeakageFinding]:
    return sorted(
        findings,
        key=lambda f: (
            -LEAKAGE_RISK_ORDER[f.risk_level],
            -SEVERITY_ORDER[f.severity],
            f.feature,
            f.rule,
        ),
    )


def aggregate_feature_risks(findings: list[LeakageFinding]) -> list[FeatureRisk]:
    """One row per flagged feature — exactly what the feature review table renders."""
    by_feature: dict[str, list[LeakageFinding]] = {}
    for finding in findings:
        by_feature.setdefault(finding.feature, []).append(finding)

    risks: list[FeatureRisk] = []
    for feature, feature_findings in by_feature.items():
        risk_level = max(
            (f.risk_level for f in feature_findings), key=lambda level: LEAKAGE_RISK_ORDER[level]
        )
        severity = max((f.severity for f in feature_findings), key=lambda s: SEVERITY_ORDER[s])
        action = max(
            (f.recommended_action for f in feature_findings), key=lambda a: ACTION_ORDER[a]
        )
        risks.append(
            FeatureRisk(
                feature=feature,
                risk_level=risk_level,
                max_severity=severity,
                recommended_action=action,
                reasons=[f.explanation for f in feature_findings],
                rules=sorted({f.rule for f in feature_findings}),
            )
        )
    return sorted(
        risks,
        key=lambda r: (-LEAKAGE_RISK_ORDER[r.risk_level], r.feature),
    )


def default_feature_selection(
    eda: EdaReport, leakage: LeakageReport, target_column: str
) -> tuple[list[str], dict[str, str]]:
    """Propose an initial selection for the review UI.

    Only the target is excluded automatically. Everything else is *proposed* for exclusion
    with a reason; the user sees and controls the final list.
    """
    proposed_exclusions: dict[str, str] = {}
    for profile in eda.columns:
        if profile.name == target_column:
            continue
        risk = leakage.risk_for(profile.name)
        if risk and risk.risk_level is LeakageRiskLevel.CONFIRMED_DUPLICATE:
            proposed_exclusions[profile.name] = risk.reasons[0]
        elif risk and risk.recommended_action in {
            RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
        }:
            proposed_exclusions[profile.name] = risk.reasons[0]
        elif profile.is_constant:
            proposed_exclusions[profile.name] = "Column is constant and carries no information."
        elif profile.semantic_type.value == "empty":
            proposed_exclusions[profile.name] = "Column is entirely missing."

    selected = [
        p.name
        for p in eda.columns
        if p.name != target_column and p.name not in proposed_exclusions
    ]
    return selected, proposed_exclusions
