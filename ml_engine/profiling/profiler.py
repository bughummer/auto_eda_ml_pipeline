"""EDA orchestration: dataframe in, :class:`EdaReport` out.

Pure and deterministic: given the same frame and config it produces the same report. No IO,
no AWS, no randomness, no LLM.
"""

import time
from datetime import UTC, datetime

import pandas as pd
from pandas.api import types as pdt

from ml_engine.contracts.common import ProblemType, SemanticType
from ml_engine.contracts.eda import (
    ClassDistributionEntry,
    ColumnProfile,
    DatasetSummary,
    EdaReport,
    TargetAnalysis,
)
from ml_engine.contracts.warnings import sort_warnings
from ml_engine.profiling import stats as stats_module
from ml_engine.profiling import warnings as warning_rules
from ml_engine.profiling.config import ProfilingConfig
from ml_engine.profiling.inference import (
    infer_problem_type,
    infer_semantic_type,
    is_high_cardinality,
)

_NUMERIC_TYPES = {SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE}
_HIGH_CARDINALITY_CANDIDATES = {
    SemanticType.CATEGORICAL,
    SemanticType.HIGH_CARDINALITY_CATEGORICAL,
    SemanticType.TEXT,
    SemanticType.IDENTIFIER,
}
_CATEGORICAL_TYPES = {
    SemanticType.CATEGORICAL,
    SemanticType.HIGH_CARDINALITY_CATEGORICAL,
    SemanticType.BOOLEAN,
    SemanticType.IDENTIFIER,
    SemanticType.TEXT,
    SemanticType.CONSTANT,
}


def profile_dataset(
    frame: pd.DataFrame,
    *,
    experiment_id: str,
    target_column: str,
    source_uri: str,
    file_format: str,
    config: ProfilingConfig | None = None,
    source_size_bytes: int | None = None,
    sampled: bool = False,
) -> EdaReport:
    """Compute the complete deterministic EDA report for a dataset."""
    config = config or ProfilingConfig()
    started = time.perf_counter()

    summary = _dataset_summary(
        frame,
        source_uri=source_uri,
        file_format=file_format,
        source_size_bytes=source_size_bytes,
        sampled=sampled,
    )
    columns = [_profile_column(frame[name], config) for name in frame.columns]
    target = _analyze_target(frame, target_column, config)

    collected = warning_rules.dataset_warnings(summary, config)
    for profile in columns:
        if profile.name == target_column:
            continue
        collected.extend(warning_rules.column_warnings(profile, config))
    collected.extend(warning_rules.target_warnings(target, summary.row_count, config))

    return EdaReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        duration_seconds=round(time.perf_counter() - started, 4),
        dataset=summary,
        target=target,
        columns=columns,
        warnings=sort_warnings(collected),
        profiling_config=config.as_metadata(),
    )


def _dataset_summary(
    frame: pd.DataFrame,
    *,
    source_uri: str,
    file_format: str,
    source_size_bytes: int | None,
    sampled: bool,
) -> DatasetSummary:
    row_count = int(len(frame))
    try:
        duplicate_count = int(frame.duplicated().sum())
        duplicates_exact = True
    except TypeError:
        # Unhashable cell values (lists, dicts) make exact duplicate detection impossible.
        duplicate_count, duplicates_exact = 0, False
    return DatasetSummary(
        row_count=row_count,
        column_count=int(frame.shape[1]),
        duplicate_row_count=duplicate_count,
        duplicate_row_percentage=round(100.0 * duplicate_count / row_count, 4) if row_count else 0.0,
        memory_usage_bytes=int(frame.memory_usage(deep=True).sum()),
        memory_usage_is_exact=duplicates_exact,
        file_format=file_format,
        source_uri=source_uri,
        source_size_bytes=source_size_bytes,
        sampled=sampled,
        sample_row_count=row_count if sampled else None,
    )


def _profile_column(series: pd.Series, config: ProfilingConfig) -> ColumnProfile:
    row_count = int(len(series))
    missing_count = int(series.isna().sum())
    non_null_count = row_count - missing_count
    unique_count = int(series.dropna().nunique())

    semantic_type = infer_semantic_type(
        series,
        config,
        row_count=row_count,
        unique_count=unique_count,
        missing_count=missing_count,
    )

    profile = ColumnProfile(
        name=str(series.name),
        semantic_type=semantic_type,
        dtype=str(series.dtype),
        missing_count=missing_count,
        missing_percentage=round(100.0 * missing_count / row_count, 4) if row_count else 0.0,
        unique_count=unique_count,
        unique_percentage=round(100.0 * unique_count / non_null_count, 4) if non_null_count else 0.0,
        is_constant=semantic_type is SemanticType.CONSTANT,
        is_high_cardinality=(
            semantic_type in _HIGH_CARDINALITY_CANDIDATES
            and is_high_cardinality(unique_count, non_null_count, config)
        ),
        is_likely_id=semantic_type is SemanticType.IDENTIFIER,
        is_text_like=semantic_type is SemanticType.TEXT,
        sample_values=stats_module.sample_values(series, config.sample_values),
    )

    if semantic_type in _NUMERIC_TYPES or (
        pdt.is_numeric_dtype(series) and semantic_type is not SemanticType.DATETIME
    ):
        profile.numeric = stats_module.numeric_stats(series)
    if semantic_type is SemanticType.DATETIME:
        profile.datetime_stats = stats_module.datetime_stats(series)
    elif semantic_type in _CATEGORICAL_TYPES:
        profile.categorical = stats_module.categorical_stats(series, config)
    return profile


def _analyze_target(
    frame: pd.DataFrame, target_column: str, config: ProfilingConfig
) -> TargetAnalysis:
    if target_column not in frame.columns:
        return TargetAnalysis(column=target_column, exists=False, inferred_problem_type=None)

    series = frame[target_column]
    row_count = int(len(series))
    missing_count = int(series.isna().sum())
    problem_type, _reason = infer_problem_type(series, config)

    analysis = TargetAnalysis(
        column=target_column,
        exists=True,
        inferred_problem_type=problem_type,
        dtype=str(series.dtype),
        missing_count=missing_count,
        missing_percentage=round(100.0 * missing_count / row_count, 4) if row_count else 0.0,
        unique_count=int(series.dropna().nunique()),
    )

    if problem_type is ProblemType.REGRESSION:
        analysis.numeric = stats_module.numeric_stats(series)
        return analysis

    if problem_type is None:
        if pdt.is_numeric_dtype(series):
            analysis.numeric = stats_module.numeric_stats(series)
        return analysis

    non_null = series.dropna()
    counts = non_null.astype("string").value_counts()
    total = int(counts.sum())
    analysis.class_distribution = [
        ClassDistributionEntry(
            label=str(label),
            count=int(count),
            percentage=round(100.0 * int(count) / total, 4) if total else 0.0,
        )
        for label, count in counts.items()
    ]
    smallest = int(counts.min())
    analysis.imbalance_ratio = round(float(counts.max()) / smallest, 6) if smallest else None
    if problem_type is ProblemType.BINARY_CLASSIFICATION:
        analysis.positive_class = _positive_class(counts)
    return analysis


def _positive_class(counts: pd.Series) -> str:
    """Positive class = the conventional '1'/'true'/'yes' label, else the minority class."""
    labels = {str(label).strip().lower(): str(label) for label in counts.index}
    for token in ("1", "true", "yes", "y", "t"):
        if token in labels:
            return labels[token]
    return str(counts.idxmin())
