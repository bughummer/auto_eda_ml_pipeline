"""Decide what happens to each selected feature before any transformer is built.

The plan is computed once, recorded in the preparation artifact, and reused by every model
so that all models in an experiment see exactly the same feature space.
"""

from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as pdt

from ml_engine.contracts.common import (
    RecommendedAction,
    SemanticType,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.config import PreprocessingConfig
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.profiling.config import ProfilingConfig
from ml_engine.profiling.inference import infer_semantic_type


@dataclass(slots=True)
class ColumnPlan:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    warnings: list[AnalysisWarning] = field(default_factory=list)

    @property
    def usable(self) -> list[str]:
        return [*self.numeric, *self.boolean, *self.categorical, *self.datetime]

    @property
    def is_empty(self) -> bool:
        return not self.usable


_DROPPED_TYPES = {
    SemanticType.TEXT: "free-text column; text features are not supported in this version",
    SemanticType.EMPTY: "column is entirely missing",
}


def plan_columns(
    frame: pd.DataFrame,
    features: list[str],
    config: PreprocessingConfig,
    eda: EdaReport | None = None,
) -> ColumnPlan:
    """Assign every selected feature to a transformation group, or drop it with a reason."""
    plan = ColumnPlan()
    profiling_config = ProfilingConfig()

    for feature in features:
        if feature not in frame.columns:
            plan.dropped[feature] = "column is not present in the dataset"
            plan.warnings.append(
                AnalysisWarning(
                    rule="selected_feature_missing",
                    category=WarningCategory.SCHEMA,
                    severity=Severity.HIGH,
                    message=f"Selected feature '{feature}' is not present in the dataset.",
                    column=feature,
                )
            )
            continue

        series = frame[feature]
        semantic_type = _semantic_type(series, feature, eda, profiling_config)

        if semantic_type in _DROPPED_TYPES and (
            semantic_type is not SemanticType.TEXT or config.drop_text_columns
        ):
            reason = _DROPPED_TYPES[semantic_type]
            plan.dropped[feature] = reason
            plan.warnings.append(
                AnalysisWarning(
                    rule="feature_dropped_by_preprocessing",
                    category=WarningCategory.PREPROCESSING,
                    severity=Severity.MEDIUM,
                    message=f"Feature '{feature}' was not used: {reason}.",
                    column=feature,
                    recommended_action=RecommendedAction.REVIEW,
                )
            )
            continue

        if semantic_type is SemanticType.CONSTANT and config.drop_constant_columns:
            plan.dropped[feature] = "column is constant in the training data"
            plan.warnings.append(
                AnalysisWarning(
                    rule="feature_dropped_by_preprocessing",
                    category=WarningCategory.PREPROCESSING,
                    severity=Severity.LOW,
                    message=f"Feature '{feature}' was not used: it is constant.",
                    column=feature,
                )
            )
            continue

        match semantic_type:
            case SemanticType.BOOLEAN:
                plan.boolean.append(feature)
            case SemanticType.NUMERIC_CONTINUOUS | SemanticType.NUMERIC_DISCRETE:
                plan.numeric.append(feature)
            case SemanticType.DATETIME:
                if config.derive_datetime_features:
                    plan.datetime.append(feature)
                else:
                    plan.dropped[feature] = "datetime feature derivation is disabled"
            case SemanticType.IDENTIFIER:
                # Kept only when the user explicitly selected it; numeric ids stay numeric.
                if pdt.is_numeric_dtype(series):
                    plan.numeric.append(feature)
                else:
                    plan.categorical.append(feature)
                plan.warnings.append(
                    AnalysisWarning(
                        rule="identifier_used_as_feature",
                        category=WarningCategory.FEATURE,
                        severity=Severity.MEDIUM,
                        message=(
                            f"Feature '{feature}' looks like an identifier but was selected for "
                            "training. It is very unlikely to generalize."
                        ),
                        column=feature,
                        recommended_action=RecommendedAction.REVIEW,
                    )
                )
            case _:
                plan.categorical.append(feature)

    return plan


def _semantic_type(
    series: pd.Series,
    feature: str,
    eda: EdaReport | None,
    profiling_config: ProfilingConfig,
) -> SemanticType:
    """Prefer the semantic type computed during EDA; recompute when it is unavailable."""
    if eda is not None:
        profile = eda.column(feature)
        if profile is not None:
            return profile.semantic_type
    row_count = len(series)
    return infer_semantic_type(
        series,
        profiling_config,
        row_count=row_count,
        unique_count=int(series.dropna().nunique()),
        missing_count=int(series.isna().sum()),
    )
