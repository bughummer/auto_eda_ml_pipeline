"""Deterministic EDA warning rules.

Every rule id is stable and is part of the platform's contract with its users: never reuse
an id with different semantics. Warnings recommend; they never change behaviour.
"""

from ml_engine.contracts.common import (
    RecommendedAction,
    SemanticType,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.eda import ColumnProfile, DatasetSummary, TargetAnalysis
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.profiling.config import ProfilingConfig


def dataset_warnings(summary: DatasetSummary, config: ProfilingConfig) -> list[AnalysisWarning]:
    found: list[AnalysisWarning] = []
    if summary.row_count == 0:
        found.append(
            AnalysisWarning(
                rule="empty_dataset",
                category=WarningCategory.SCHEMA,
                severity=Severity.CRITICAL,
                message="The dataset contains no rows.",
                recommended_action=RecommendedAction.REVIEW,
            )
        )
        return found

    if summary.row_count < config.small_dataset_rows:
        found.append(
            AnalysisWarning(
                rule="small_dataset",
                category=WarningCategory.DATA_QUALITY,
                severity=Severity.MEDIUM,
                message=(
                    f"Only {summary.row_count} rows. Validation metrics will be unstable and "
                    "model comparison may not be meaningful."
                ),
                details={"row_count": summary.row_count},
            )
        )

    duplicate_fraction = summary.duplicate_row_count / summary.row_count
    if duplicate_fraction > config.duplicate_rows_warn_fraction:
        found.append(
            AnalysisWarning(
                rule="duplicate_rows",
                category=WarningCategory.DATA_QUALITY,
                severity=Severity.HIGH if duplicate_fraction > 0.1 else Severity.MEDIUM,
                message=(
                    f"{summary.duplicate_row_count} duplicate rows "
                    f"({summary.duplicate_row_percentage:.2f}%). Duplicates split across train "
                    "and validation inflate validation scores."
                ),
                details={
                    "duplicate_row_count": summary.duplicate_row_count,
                    "duplicate_row_percentage": summary.duplicate_row_percentage,
                },
            )
        )

    if summary.column_count > summary.row_count:
        found.append(
            AnalysisWarning(
                rule="wide_dataset",
                category=WarningCategory.SCHEMA,
                severity=Severity.MEDIUM,
                message=(
                    f"{summary.column_count} columns for {summary.row_count} rows. Expect "
                    "overfitting; reduce the feature set."
                ),
                details={"row_count": summary.row_count, "column_count": summary.column_count},
            )
        )

    if summary.sampled:
        found.append(
            AnalysisWarning(
                rule="dataset_sampled",
                category=WarningCategory.DATA_QUALITY,
                severity=Severity.INFO,
                message=(
                    f"Profiling used a sample of {summary.sample_row_count} rows. Statistics "
                    "describe the sample, not the full dataset."
                ),
                details={"sample_row_count": summary.sample_row_count},
            )
        )
    return found


def column_warnings(profile: ColumnProfile, config: ProfilingConfig) -> list[AnalysisWarning]:
    found: list[AnalysisWarning] = []
    fraction_missing = profile.missing_percentage / 100.0

    if profile.semantic_type is SemanticType.EMPTY:
        found.append(
            AnalysisWarning(
                rule="empty_column",
                category=WarningCategory.DATA_QUALITY,
                severity=Severity.HIGH,
                message="Column is entirely missing and carries no information.",
                column=profile.name,
                recommended_action=RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
            )
        )
    elif fraction_missing >= config.missing_high_fraction:
        found.append(
            AnalysisWarning(
                rule="high_missingness",
                category=WarningCategory.DATA_QUALITY,
                severity=Severity.HIGH,
                message=(
                    f"{profile.missing_percentage:.2f}% of values are missing. Imputation will "
                    "dominate this column."
                ),
                column=profile.name,
                recommended_action=RecommendedAction.CONSIDER_EXCLUDING,
                details={"missing_percentage": profile.missing_percentage},
            )
        )
    elif fraction_missing >= config.missing_warn_fraction:
        found.append(
            AnalysisWarning(
                rule="high_missingness",
                category=WarningCategory.DATA_QUALITY,
                severity=Severity.MEDIUM,
                message=f"{profile.missing_percentage:.2f}% of values are missing.",
                column=profile.name,
                details={"missing_percentage": profile.missing_percentage},
            )
        )

    if profile.is_constant and profile.semantic_type is not SemanticType.EMPTY:
        found.append(
            AnalysisWarning(
                rule="constant_column",
                category=WarningCategory.FEATURE,
                severity=Severity.MEDIUM,
                message="Column has a single distinct value and cannot inform a model.",
                column=profile.name,
                recommended_action=RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
            )
        )

    if profile.is_likely_id:
        found.append(
            AnalysisWarning(
                rule="likely_identifier",
                category=WarningCategory.FEATURE,
                severity=Severity.HIGH,
                message=(
                    f"{profile.unique_percentage:.2f}% of values are unique, which looks like an "
                    "identifier. Identifiers memorize rows instead of generalizing."
                ),
                column=profile.name,
                recommended_action=RecommendedAction.CONSIDER_EXCLUDING,
                details={"unique_percentage": profile.unique_percentage},
            )
        )
    elif profile.is_high_cardinality:
        found.append(
            AnalysisWarning(
                rule="high_cardinality",
                category=WarningCategory.FEATURE,
                severity=Severity.MEDIUM,
                message=(
                    f"{profile.unique_count} distinct values. One-hot encoding will expand the "
                    "feature space; consider grouping rare categories."
                ),
                column=profile.name,
                details={"unique_count": profile.unique_count},
            )
        )

    if profile.is_text_like:
        found.append(
            AnalysisWarning(
                rule="text_column",
                category=WarningCategory.FEATURE,
                severity=Severity.MEDIUM,
                message=(
                    "Column looks like free text. Text features are excluded by default in this "
                    "version; no vectorization is performed."
                ),
                column=profile.name,
                recommended_action=RecommendedAction.CONSIDER_EXCLUDING,
            )
        )

    if profile.semantic_type is SemanticType.DATETIME:
        found.append(
            AnalysisWarning(
                rule="datetime_column",
                category=WarningCategory.FEATURE,
                severity=Severity.INFO,
                message=(
                    "Datetime column detected. Only year, month and day-of-week are derived; "
                    "no temporal validation split is applied in this version."
                ),
                column=profile.name,
            )
        )
    return found


def target_warnings(
    target: TargetAnalysis, row_count: int, config: ProfilingConfig
) -> list[AnalysisWarning]:
    found: list[AnalysisWarning] = []
    if not target.exists:
        found.append(
            AnalysisWarning(
                rule="target_missing_from_dataset",
                category=WarningCategory.TARGET,
                severity=Severity.CRITICAL,
                message=f"Target column '{target.column}' is not present in the dataset.",
                column=target.column,
            )
        )
        return found

    if target.missing_count:
        severity = Severity.HIGH if target.missing_percentage > 10 else Severity.MEDIUM
        found.append(
            AnalysisWarning(
                rule="target_missing_values",
                category=WarningCategory.TARGET,
                severity=severity,
                message=(
                    f"{target.missing_count} rows ({target.missing_percentage:.2f}%) have no "
                    "target value. They are dropped before training."
                ),
                column=target.column,
                details={"missing_count": target.missing_count},
            )
        )

    if target.inferred_problem_type is None:
        found.append(
            AnalysisWarning(
                rule="target_not_learnable",
                category=WarningCategory.TARGET,
                severity=Severity.CRITICAL,
                message=(
                    f"Target column '{target.column}' cannot support supervised learning as it "
                    "stands. Check the column choice or the maximum supported class count."
                ),
                column=target.column,
            )
        )
        return found

    if target.class_distribution:
        smallest = min(entry.count for entry in target.class_distribution)
        if smallest < 10:
            found.append(
                AnalysisWarning(
                    rule="rare_target_class",
                    category=WarningCategory.TARGET,
                    severity=Severity.HIGH,
                    message=(
                        f"The smallest target class has only {smallest} rows. Stratified "
                        "splitting and validation metrics will be unreliable."
                    ),
                    column=target.column,
                    details={"smallest_class_count": smallest},
                )
            )
        if target.imbalance_ratio and target.imbalance_ratio >= config.imbalance_warn_ratio:
            severe = target.imbalance_ratio >= config.imbalance_high_ratio
            found.append(
                AnalysisWarning(
                    rule="target_imbalance",
                    category=WarningCategory.TARGET,
                    severity=Severity.HIGH if severe else Severity.MEDIUM,
                    message=(
                        f"Class imbalance ratio {target.imbalance_ratio:.2f}:1. Accuracy will be "
                        "misleading; prefer ROC AUC / PR AUC and enable class weighting."
                    ),
                    column=target.column,
                    details={"imbalance_ratio": round(target.imbalance_ratio, 4)},
                )
            )
    return found
