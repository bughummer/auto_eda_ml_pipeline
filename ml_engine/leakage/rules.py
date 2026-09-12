"""Deterministic leakage rules.

Every rule is a pure function ``(LeakageContext) -> list[LeakageFinding]``. Rules report
evidence; they never mutate the dataset or the feature selection.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ml_engine.contracts.common import (
    LeakageRiskLevel,
    ProblemType,
    RecommendedAction,
    SemanticType,
    Severity,
)
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageFinding
from ml_engine.leakage.config import LeakageConfig


@dataclass(slots=True)
class LeakageContext:
    """Everything the rules need, prepared once."""

    frame: pd.DataFrame
    target: pd.Series
    target_column: str
    problem_type: ProblemType
    eda: EdaReport
    config: LeakageConfig

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c != self.target_column]

    def profile(self, column: str):
        return self.eda.column(column)


def build_context(
    frame: pd.DataFrame,
    *,
    target_column: str,
    problem_type: ProblemType,
    eda: EdaReport,
    config: LeakageConfig | None = None,
) -> LeakageContext:
    """Drop rows without a target and sample deterministically for the relationship checks."""
    config = config or LeakageConfig()
    usable = frame[frame[target_column].notna()]
    if len(usable) > config.max_rows:
        usable = usable.sample(n=config.max_rows, random_state=0).sort_index()
    return LeakageContext(
        frame=usable,
        target=usable[target_column],
        target_column=target_column,
        problem_type=problem_type,
        eda=eda,
        config=config,
    )


def exact_target_duplicate(ctx: LeakageContext) -> list[LeakageFinding]:
    """A feature that *is* the target, possibly relabelled or inverted."""
    findings: list[LeakageFinding] = []
    target_text = ctx.target.astype("string")
    target_numeric = pd.to_numeric(ctx.target, errors="coerce")
    binary_target = target_numeric.dropna().isin([0, 1]).all() and target_numeric.nunique() == 2

    for column in ctx.feature_columns:
        series = ctx.frame[column]
        if series.astype("string").equals(target_text):
            findings.append(
                LeakageFinding(
                    feature=column,
                    rule="exact_target_duplicate",
                    severity=Severity.CRITICAL,
                    risk_level=LeakageRiskLevel.CONFIRMED_DUPLICATE,
                    explanation=(
                        f"Column '{column}' is identical to the target column "
                        f"'{ctx.target_column}' for every row. Training on it measures nothing."
                    ),
                    recommended_action=RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
                    evidence={"match_fraction": 1.0},
                )
            )
            continue

        if binary_target:
            numeric = pd.to_numeric(series, errors="coerce")
            is_binary_feature = numeric.notna().all() and numeric.isin([0, 1]).all()
            if is_binary_feature and (numeric == (1 - target_numeric)).all():
                findings.append(
                    LeakageFinding(
                        feature=column,
                        rule="inverse_binary_target",
                        severity=Severity.CRITICAL,
                        risk_level=LeakageRiskLevel.CONFIRMED_DUPLICATE,
                        explanation=(
                            f"Column '{column}' is the exact inverse of the binary target "
                            f"'{ctx.target_column}'."
                        ),
                        recommended_action=RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
                        evidence={"match_fraction": 1.0},
                    )
                )
    return findings


def duplicate_features(ctx: LeakageContext) -> list[LeakageFinding]:
    """Columns identical to another column: redundant, and they distort importance."""
    findings: list[LeakageFinding] = []
    seen: dict[str, str] = {}
    for column in ctx.feature_columns:
        try:
            digest = pd.util.hash_pandas_object(ctx.frame[column].astype("string"), index=False)
            key = str(int(digest.sum())) + f":{len(digest)}"
        except (TypeError, ValueError):
            continue
        if key in seen:
            first = seen[key]
            if ctx.frame[column].astype("string").equals(ctx.frame[first].astype("string")):
                findings.append(
                    LeakageFinding(
                        feature=column,
                        rule="duplicate_feature",
                        severity=Severity.LOW,
                        risk_level=LeakageRiskLevel.REQUIRES_REVIEW,
                        explanation=(
                            f"Column '{column}' holds the same values as '{first}'. Keeping both "
                            "adds no information and splits feature importance between them."
                        ),
                        recommended_action=RecommendedAction.CONSIDER_EXCLUDING,
                        evidence={"duplicate_of": first},
                    )
                )
        else:
            seen[key] = column
    return findings


def near_deterministic_relationship(ctx: LeakageContext) -> list[LeakageFinding]:
    """A single feature that reproduces the target almost perfectly on its own."""
    findings: list[LeakageFinding] = []
    for column in ctx.feature_columns:
        profile = ctx.profile(column)
        if profile is None or profile.semantic_type in {
            SemanticType.IDENTIFIER,
            SemanticType.CONSTANT,
            SemanticType.EMPTY,
            SemanticType.TEXT,
        }:
            continue
        strength, method = _relationship_strength(ctx, column)
        if strength is None:
            continue
        if strength >= ctx.config.perfect_relationship:
            findings.append(
                LeakageFinding(
                    feature=column,
                    rule="near_deterministic_relationship",
                    severity=Severity.HIGH,
                    risk_level=LeakageRiskLevel.POTENTIAL_LEAKAGE,
                    explanation=(
                        f"Column '{column}' alone reproduces the target almost perfectly "
                        f"({method}={strength:.4f}). This is usually leakage, but it can be a "
                        "legitimately dominant feature — confirm it is available when the "
                        "prediction is made."
                    ),
                    recommended_action=RecommendedAction.CONSIDER_EXCLUDING,
                    evidence={"metric": method, "value": round(strength, 6)},
                )
            )
        elif strength >= ctx.config.strong_relationship:
            findings.append(
                LeakageFinding(
                    feature=column,
                    rule="strong_single_feature_relationship",
                    severity=Severity.MEDIUM,
                    risk_level=LeakageRiskLevel.REQUIRES_REVIEW,
                    explanation=(
                        f"Column '{column}' predicts the target unusually well on its own "
                        f"({method}={strength:.4f}). Verify it is not derived from the outcome."
                    ),
                    recommended_action=RecommendedAction.REVIEW,
                    evidence={"metric": method, "value": round(strength, 6)},
                )
            )
    return findings


def _relationship_strength(ctx: LeakageContext, column: str) -> tuple[float | None, str]:
    series = ctx.frame[column]
    if series.isna().all():
        return None, ""
    if ctx.problem_type is ProblemType.REGRESSION:
        return _regression_strength(ctx, series)
    return _classification_strength(ctx, series)


def _regression_strength(ctx: LeakageContext, series: pd.Series) -> tuple[float | None, str]:
    target = pd.to_numeric(ctx.target, errors="coerce")
    if pdt.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        mask = numeric.notna() & target.notna()
        if mask.sum() < 10 or numeric[mask].nunique() < 2:
            return None, ""
        correlation = float(np.abs(np.corrcoef(numeric[mask], target[mask])[0, 1]))
        return (correlation if np.isfinite(correlation) else None), "abs_pearson_r"

    levels = series.dropna().nunique()
    if levels < 2 or levels > ctx.config.max_categorical_levels:
        return None, ""
    mask = series.notna() & target.notna()
    if mask.sum() < 10:
        return None, ""
    grouped = target[mask].groupby(series[mask].astype("string"), observed=True)
    residual = float(((target[mask] - grouped.transform("mean")) ** 2).sum())
    total = float(((target[mask] - target[mask].mean()) ** 2).sum())
    if total <= 0:
        return None, ""
    return max(0.0, 1.0 - residual / total), "group_mean_r2"


def _classification_strength(ctx: LeakageContext, series: pd.Series) -> tuple[float | None, str]:
    """Best accuracy a lookup table on this single feature could achieve, baseline-corrected."""
    target = ctx.target.astype("string")
    mask = series.notna() & target.notna()
    if mask.sum() < 10:
        return None, ""
    values = series[mask]
    labels = target[mask]

    levels = values.nunique()
    if levels < 2:
        return None, ""
    if levels > ctx.config.max_categorical_levels:
        if not pdt.is_numeric_dtype(values):
            return None, ""
        try:
            values = pd.qcut(values, q=10, duplicates="drop")
        except (ValueError, TypeError):
            return None, ""
        if values.nunique() < 2:
            return None, ""

    frame = pd.DataFrame({"value": values.astype("string"), "label": labels})
    per_group_best = frame.groupby("value", observed=True)["label"].agg(
        lambda group: group.value_counts().iloc[0]
    )
    accuracy = float(per_group_best.sum()) / float(len(frame))
    baseline = float(labels.value_counts().iloc[0]) / float(len(labels))
    if baseline >= 1.0:
        return None, ""
    # Normalize against the majority-class baseline so a 99%-imbalanced target does not make
    # every column look deterministic.
    return (accuracy - baseline) / (1.0 - baseline), "baseline_corrected_purity"


def suspicious_feature_name(ctx: LeakageContext) -> list[LeakageFinding]:
    """Name-based heuristics. Always ``requires_review`` — a name is never proof."""
    findings: list[LeakageFinding] = []
    target_token = ctx.target_column.strip().lower()
    for column in ctx.feature_columns:
        lowered = column.strip().lower()
        matched = [t for t in ctx.config.suspicious_name_tokens if t in lowered]
        is_post_outcome = any(t in lowered for t in ctx.config.post_outcome_tokens)
        contains_target = target_token and target_token in lowered

        if contains_target:
            findings.append(
                LeakageFinding(
                    feature=column,
                    rule="target_name_in_feature",
                    severity=Severity.HIGH,
                    risk_level=LeakageRiskLevel.REQUIRES_REVIEW,
                    explanation=(
                        f"Column name '{column}' contains the target name "
                        f"'{ctx.target_column}', which often indicates a derived or "
                        "post-outcome variable."
                    ),
                    recommended_action=RecommendedAction.REVIEW,
                    evidence={"target_column": ctx.target_column},
                )
            )
        elif matched:
            findings.append(
                LeakageFinding(
                    feature=column,
                    rule="post_outcome_name" if is_post_outcome else "suspicious_feature_name",
                    severity=Severity.MEDIUM if is_post_outcome else Severity.LOW,
                    risk_level=LeakageRiskLevel.REQUIRES_REVIEW,
                    explanation=(
                        f"Column name '{column}' matches the naming pattern "
                        f"{matched[0]!r}, which suggests a value recorded after the outcome is "
                        "known. Confirm it exists at prediction time."
                    ),
                    recommended_action=RecommendedAction.REVIEW,
                    evidence={"matched_token": matched[0]},
                )
            )
    return findings


def identifier_like(ctx: LeakageContext) -> list[LeakageFinding]:
    """Identifiers memorize rows; they also frequently encode collection order."""
    findings: list[LeakageFinding] = []
    for column in ctx.feature_columns:
        profile = ctx.profile(column)
        if profile is None or not profile.is_likely_id:
            continue
        findings.append(
            LeakageFinding(
                feature=column,
                rule="identifier_like_feature",
                severity=Severity.MEDIUM,
                risk_level=LeakageRiskLevel.REQUIRES_REVIEW,
                explanation=(
                    f"Column '{column}' is almost entirely unique "
                    f"({profile.unique_percentage:.2f}%), which is characteristic of an "
                    "identifier. Identifiers cannot generalize and may encode record order."
                ),
                recommended_action=RecommendedAction.CONSIDER_EXCLUDING,
                evidence={"unique_percentage": profile.unique_percentage},
            )
        )
    return findings


def missingness_predicts_target(ctx: LeakageContext) -> list[LeakageFinding]:
    """When *whether* a value exists predicts the outcome, the column is often post-outcome."""
    if ctx.problem_type is ProblemType.REGRESSION:
        return []
    findings: list[LeakageFinding] = []
    labels = ctx.target.astype("string")
    baseline = float(labels.value_counts().iloc[0]) / float(len(labels)) if len(labels) else 1.0
    if baseline >= 1.0:
        return []

    for column in ctx.feature_columns:
        missing = ctx.frame[column].isna()
        missing_fraction = float(missing.mean())
        if missing_fraction < 0.02 or missing_fraction > 0.98:
            continue
        frame = pd.DataFrame({"missing": missing, "label": labels})
        best = frame.groupby("missing", observed=True)["label"].agg(
            lambda group: group.value_counts().iloc[0]
        )
        accuracy = float(best.sum()) / float(len(frame))
        lift = (accuracy - baseline) / (1.0 - baseline)
        if lift >= ctx.config.missingness_signal_auc:
            findings.append(
                LeakageFinding(
                    feature=column,
                    rule="missingness_predicts_target",
                    severity=Severity.HIGH,
                    risk_level=LeakageRiskLevel.POTENTIAL_LEAKAGE,
                    explanation=(
                        f"Whether '{column}' is populated predicts the target on its own "
                        f"(baseline-corrected accuracy {lift:.3f}). Values that only exist for "
                        "certain outcomes are usually recorded after the fact."
                    ),
                    recommended_action=RecommendedAction.CONSIDER_EXCLUDING,
                    evidence={
                        "missing_fraction": round(missing_fraction, 6),
                        "baseline_corrected_accuracy": round(lift, 6),
                    },
                )
            )
    return findings


RULES = (
    exact_target_duplicate,
    duplicate_features,
    near_deterministic_relationship,
    suspicious_feature_name,
    identifier_like,
    missingness_predicts_target,
)
