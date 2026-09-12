"""Leakage screening must find structural leaks and stay conservative about the rest."""

import numpy as np
import pandas as pd

from ml_engine.contracts.common import (
    LeakageRiskLevel,
    ProblemType,
    RecommendedAction,
)
from ml_engine.leakage import analyze_leakage, default_feature_selection
from ml_engine.profiling import profile_dataset


def _analyze(frame: pd.DataFrame, target: str = "churned"):
    eda = profile_dataset(
        frame,
        experiment_id="exp-1",
        target_column=target,
        source_uri="memory://test",
        file_format="csv",
    )
    report = analyze_leakage(
        frame,
        experiment_id="exp-1",
        target_column=target,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        eda=eda,
    )
    return eda, report


def test_exact_target_duplicate_is_confirmed(classification_frame):
    _eda, report = _analyze(classification_frame)
    finding = next(
        f
        for f in report.findings
        if f.feature == "churn_copy" and f.rule == "exact_target_duplicate"
    )
    assert finding.risk_level is LeakageRiskLevel.CONFIRMED_DUPLICATE
    assert finding.recommended_action is RecommendedAction.STRONGLY_CONSIDER_EXCLUDING


def test_inverse_binary_target_is_detected(classification_frame):
    frame = classification_frame.copy()
    frame["retained"] = 1 - frame["churned"]
    _eda, report = _analyze(frame)
    assert any(
        f.rule == "inverse_binary_target" and f.feature == "retained" for f in report.findings
    )


def test_duplicate_columns_are_flagged(classification_frame):
    frame = classification_frame.copy()
    frame["tenure_copy"] = frame["tenure_months"]
    _eda, report = _analyze(frame)
    finding = next(f for f in report.findings if f.rule == "duplicate_feature")
    assert finding.feature == "tenure_copy"
    assert finding.evidence["duplicate_of"] == "tenure_months"


def test_identifier_is_flagged_for_review(classification_frame):
    _eda, report = _analyze(classification_frame)
    risk = report.risk_for("customer_id")
    assert risk is not None
    assert risk.risk_level is LeakageRiskLevel.REQUIRES_REVIEW


def test_post_outcome_names_are_flagged(classification_frame):
    frame = classification_frame.copy()
    frame["account_closed_date"] = "2024-01-01"
    _eda, report = _analyze(frame)
    assert any(
        f.feature == "account_closed_date" and f.rule == "post_outcome_name"
        for f in report.findings
    )


def test_missingness_that_predicts_the_target_is_flagged(classification_frame):
    frame = classification_frame.copy()
    frame["closure_note"] = np.where(frame["churned"] == 1, "closed", None)
    _eda, report = _analyze(frame)
    assert any(f.rule == "missingness_predicts_target" for f in report.findings)


def test_strong_but_legitimate_feature_is_review_not_verdict():
    """A dominant feature is surfaced for review, never labelled confirmed leakage."""
    rng = np.random.default_rng(3)
    rows = 400
    signal = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=rows),
            "churned": (signal + rng.normal(0, 0.05, rows) > 0).astype(int),
        }
    )
    _eda, report = _analyze(frame)
    risk = report.risk_for("signal")
    assert risk is not None
    assert risk.risk_level is not LeakageRiskLevel.CONFIRMED_DUPLICATE


def test_unrelated_features_are_not_flagged():
    rng = np.random.default_rng(5)
    rows = 400
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.choice(["x", "y", "z"], rows),
            "churned": rng.choice([0, 1], rows),
        }
    )
    _eda, report = _analyze(frame)
    assert report.findings == []


def test_only_structural_problems_are_excluded_by_default(classification_frame):
    eda, report = _analyze(classification_frame)
    selected, proposed = default_feature_selection(eda, report, "churned")
    assert "churn_copy" in proposed  # exact duplicate of the target
    assert "region_code" in proposed  # constant
    assert "customer_id" in selected  # flagged for review, but the user decides
    assert "churned" not in selected  # the target is never a feature


def test_rule_failures_are_recorded_not_raised(classification_frame, monkeypatch):
    import ml_engine.leakage.analyzer as analyzer

    def broken(_ctx):
        raise RuntimeError("rule exploded")

    broken.__name__ = "broken_rule"
    monkeypatch.setattr(analyzer, "RULES", (broken,))
    eda = profile_dataset(
        classification_frame,
        experiment_id="exp-1",
        target_column="churned",
        source_uri="memory://test",
        file_format="csv",
    )
    report = analyzer.analyze_leakage(
        classification_frame,
        experiment_id="exp-1",
        target_column="churned",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        eda=eda,
    )
    assert "broken_rule" in report.checks_skipped
