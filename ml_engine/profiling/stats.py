"""Deterministic per-column statistics."""

import math

import pandas as pd
from pandas.api import types as pdt

from ml_engine.contracts.eda import (
    CategoricalStats,
    CategoryFrequency,
    DatetimeStats,
    NumericStats,
)
from ml_engine.profiling.config import ProfilingConfig


def _clean(value: float | None) -> float | None:
    """NaN/inf are not valid JSON; they become nulls with no silent substitution."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def numeric_stats(series: pd.Series) -> NumericStats:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return NumericStats(
            min=None, max=None, mean=None, median=None, std=None,
            q01=None, q05=None, q25=None, q75=None, q95=None, q99=None,
            zero_count=0, zero_percentage=0.0, negative_count=0, skewness=None,
        )
    total = int(len(numeric))
    quantiles = non_null.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    zero_count = int((non_null == 0).sum())
    return NumericStats(
        min=_clean(non_null.min()),
        max=_clean(non_null.max()),
        mean=_clean(non_null.mean()),
        median=_clean(quantiles.loc[0.5]),
        std=_clean(non_null.std()) if len(non_null) > 1 else 0.0,
        q01=_clean(quantiles.loc[0.01]),
        q05=_clean(quantiles.loc[0.05]),
        q25=_clean(quantiles.loc[0.25]),
        q75=_clean(quantiles.loc[0.75]),
        q95=_clean(quantiles.loc[0.95]),
        q99=_clean(quantiles.loc[0.99]),
        zero_count=zero_count,
        zero_percentage=round(100.0 * zero_count / total, 4) if total else 0.0,
        negative_count=int((non_null < 0).sum()),
        skewness=_clean(non_null.skew()) if len(non_null) > 2 else None,
    )


def categorical_stats(series: pd.Series, config: ProfilingConfig) -> CategoricalStats:
    non_null = series.dropna()
    total = int(len(non_null))
    if total == 0:
        return CategoricalStats(top_values=[], truncated=False)
    as_text = non_null.astype("string")
    counts = as_text.value_counts()
    top = counts.head(config.top_values)
    lengths = as_text.head(1000).str.len()
    return CategoricalStats(
        top_values=[
            CategoryFrequency(
                value=str(value),
                count=int(count),
                percentage=round(100.0 * int(count) / total, 4),
            )
            for value, count in top.items()
        ],
        truncated=bool(len(counts) > len(top)),
        mean_length=_clean(lengths.mean()),
        max_length=int(lengths.max()) if len(lengths) else None,
    )


def datetime_stats(series: pd.Series) -> DatetimeStats:
    if pdt.is_datetime64_any_dtype(series):
        parsed = series
    else:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null = parsed.dropna()
    if non_null.empty:
        return DatetimeStats(min=None, max=None, range_days=None)
    minimum, maximum = non_null.min(), non_null.max()
    span = maximum - minimum
    return DatetimeStats(
        min=minimum.to_pydatetime(),
        max=maximum.to_pydatetime(),
        range_days=_clean(span.total_seconds() / 86400.0),
    )


def sample_values(series: pd.Series, count: int) -> list[str]:
    if count <= 0:
        return []
    non_null = series.dropna()
    if non_null.empty:
        return []
    return [str(v) for v in non_null.head(count).tolist()]
