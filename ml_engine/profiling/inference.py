"""Semantic type and problem type inference.

Deterministic rules over dtypes and value distributions. No heuristics that depend on
column names — naming heuristics belong to the leakage package, where they are surfaced as
reviewable findings rather than silently changing behaviour.
"""

import pandas as pd
from pandas.api import types as pdt

from ml_engine.contracts.common import ProblemType, SemanticType
from ml_engine.profiling.config import ProfilingConfig

_BOOLEAN_TOKENS = ({"true", "false"}, {"yes", "no"}, {"y", "n"}, {"t", "f"}, {"0", "1"})


def is_datetime_series(series: pd.Series) -> bool:
    return bool(pdt.is_datetime64_any_dtype(series))


def looks_like_datetime(series: pd.Series, config: ProfilingConfig) -> bool:
    """True when a non-datetime column parses as dates for nearly every non-null value."""
    non_null = series.dropna()
    if non_null.empty or pdt.is_numeric_dtype(non_null) or pdt.is_bool_dtype(non_null):
        return False
    sample = non_null.head(500).astype("string")
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return float(parsed.notna().mean()) >= config.datetime_parse_fraction


def is_boolean_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if pdt.is_bool_dtype(non_null):
        return True
    values = set(non_null.unique().tolist())
    if len(values) > 2:
        return False
    if pdt.is_numeric_dtype(non_null):
        return values.issubset({0, 1}) and len(values) == 2
    lowered = {str(v).strip().lower() for v in values}
    return any(lowered.issubset(tokens) and len(lowered) == 2 for tokens in _BOOLEAN_TOKENS)


def text_statistics(series: pd.Series) -> tuple[float, int, float]:
    """Return (mean length, max length, mean word count) for a string-like column."""
    non_null = series.dropna().astype("string")
    if non_null.empty:
        return 0.0, 0, 0.0
    sample = non_null.head(1000)
    lengths = sample.str.len()
    words = sample.str.split().str.len()
    return (
        float(lengths.mean()),
        int(lengths.max()),
        float(words.mean()) if len(words) else 0.0,
    )


def infer_semantic_type(
    series: pd.Series,
    config: ProfilingConfig,
    *,
    row_count: int,
    unique_count: int,
    missing_count: int,
) -> SemanticType:
    """Classify what a column *means*, independent of how pandas stored it."""
    if row_count == 0 or missing_count == row_count:
        return SemanticType.EMPTY
    if unique_count <= 1:
        return SemanticType.CONSTANT
    if is_datetime_series(series) or looks_like_datetime(series, config):
        return SemanticType.DATETIME
    if is_boolean_like(series):
        return SemanticType.BOOLEAN

    non_null_count = row_count - missing_count
    unique_ratio = unique_count / non_null_count if non_null_count else 0.0

    if pdt.is_numeric_dtype(series):
        is_integral = pdt.is_integer_dtype(series) or _is_integral_float(series)
        if (
            is_integral
            and row_count >= config.identifier_min_rows
            and unique_ratio >= config.identifier_unique_ratio
        ):
            return SemanticType.IDENTIFIER
        if is_integral and unique_count <= config.discrete_max_unique:
            return SemanticType.NUMERIC_DISCRETE
        return SemanticType.NUMERIC_CONTINUOUS

    mean_length, _max_length, mean_words = text_statistics(series)
    if mean_length >= config.text_mean_length or mean_words >= config.text_min_word_count:
        return SemanticType.TEXT
    if row_count >= config.identifier_min_rows and unique_ratio >= config.identifier_unique_ratio:
        return SemanticType.IDENTIFIER
    if is_high_cardinality(unique_count, non_null_count, config):
        return SemanticType.HIGH_CARDINALITY_CATEGORICAL
    return SemanticType.CATEGORICAL


def _is_integral_float(series: pd.Series) -> bool:
    if pdt.is_integer_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty or not pdt.is_float_dtype(non_null):
        return False
    sample = non_null.head(2000)
    try:
        return bool((sample % 1 == 0).all())
    except TypeError:
        return False


def is_high_cardinality(unique_count: int, non_null_count: int, config: ProfilingConfig) -> bool:
    if non_null_count <= 0:
        return False
    ratio = unique_count / non_null_count
    return unique_count > config.high_cardinality_absolute and ratio > config.high_cardinality_ratio


def infer_problem_type(
    target: pd.Series, config: ProfilingConfig
) -> tuple[ProblemType | None, str | None]:
    """Infer the problem type from the target column.

    Returns ``(problem_type, reason)``; ``problem_type`` is ``None`` when the target cannot
    support supervised learning, with ``reason`` explaining why.
    """
    non_null = target.dropna()
    if non_null.empty:
        return None, "target column contains no non-missing values"

    unique_count = int(non_null.nunique())
    if unique_count <= 1:
        return None, "target column has a single distinct value"

    if pdt.is_bool_dtype(non_null) or is_boolean_like(non_null):
        return ProblemType.BINARY_CLASSIFICATION, "target has two boolean-like values"

    if pdt.is_numeric_dtype(non_null):
        is_integral = pdt.is_integer_dtype(non_null) or _is_integral_float(non_null)
        if unique_count == 2:
            return ProblemType.BINARY_CLASSIFICATION, "numeric target has exactly two values"
        if is_integral and unique_count <= config.max_classes_for_classification:
            return (
                ProblemType.MULTICLASS_CLASSIFICATION,
                f"integer target with {unique_count} distinct values",
            )
        return ProblemType.REGRESSION, "continuous numeric target"

    if unique_count == 2:
        return ProblemType.BINARY_CLASSIFICATION, "categorical target has exactly two values"
    if unique_count <= config.max_classes_for_classification:
        return (
            ProblemType.MULTICLASS_CLASSIFICATION,
            f"categorical target with {unique_count} distinct values",
        )
    return None, (
        f"categorical target has {unique_count} distinct values, above the supported maximum of "
        f"{config.max_classes_for_classification}"
    )
