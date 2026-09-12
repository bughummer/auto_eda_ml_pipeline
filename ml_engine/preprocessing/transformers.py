"""Custom, picklable transformers used by the preprocessing pipelines.

All of them are stateless with respect to the data (they derive, they do not learn), so
they carry no risk of leaking validation information. Anything that *learns* — imputers,
scalers, encoders — is a stock scikit-learn transformer fitted on the training fold only.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SUPPORTED_DATETIME_FEATURES = ("year", "month", "day", "day_of_week", "quarter", "hour")


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Derive a small, controlled set of numeric features from datetime columns.

    Deliberately minimal: no cyclical encodings, no elapsed-time-to-event features, and no
    features that could encode information from after the prediction moment.
    """

    def __init__(self, features: tuple[str, ...] = ("year", "month", "day_of_week")) -> None:
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DatetimeFeatureExtractor":  # noqa: N803
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        unsupported = [f for f in self.features if f not in SUPPORTED_DATETIME_FEATURES]
        if unsupported:
            raise ValueError(
                f"Unsupported datetime features {unsupported}. "
                f"Supported: {', '.join(SUPPORTED_DATETIME_FEATURES)}."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        frame = pd.DataFrame(index=X.index)
        for column in self.feature_names_in_:
            parsed = pd.to_datetime(X[column], errors="coerce", format="mixed")
            for feature in self.features:
                frame[f"{column}__{feature}"] = _datetime_part(parsed, feature)
        return frame

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        return np.array(
            [f"{column}__{feature}" for column in self.feature_names_in_ for feature in self.features]
        )


def _datetime_part(parsed: pd.Series, feature: str) -> pd.Series:
    accessor = parsed.dt
    match feature:
        case "year":
            values = accessor.year
        case "month":
            values = accessor.month
        case "day":
            values = accessor.day
        case "day_of_week":
            values = accessor.dayofweek
        case "quarter":
            values = accessor.quarter
        case "hour":
            values = accessor.hour
        case _:  # pragma: no cover - guarded in fit
            raise ValueError(f"Unsupported datetime feature {feature!r}")
    return values.astype("float64")


class BooleanEncoder(BaseEstimator, TransformerMixin):
    """Normalize boolean-ish columns (True/False, yes/no, 0/1, t/f) to 0.0/1.0 floats."""

    _TRUE = frozenset({"true", "yes", "y", "t", "1", "1.0"})
    _FALSE = frozenset({"false", "no", "n", "f", "0", "0.0"})

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BooleanEncoder":  # noqa: N803
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return pd.DataFrame(
            {column: _to_binary(X[column]) for column in self.feature_names_in_}, index=X.index
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        return np.array(self.feature_names_in_)


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    text = series.astype("string").str.strip().str.lower()
    mapped = pd.Series(np.nan, index=series.index, dtype="float64")
    mapped[text.isin(BooleanEncoder._TRUE)] = 1.0
    mapped[text.isin(BooleanEncoder._FALSE)] = 0.0
    return mapped


class CategoricalAsString(BaseEstimator, TransformerMixin):
    """Cast categorical columns to filled strings for libraries with native category support."""

    def __init__(self, fill_value: str = "__missing__") -> None:
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalAsString":  # noqa: N803
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return pd.DataFrame(
            {
                column: X[column].astype("string").fillna(self.fill_value).astype("object")
                for column in self.feature_names_in_
            },
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        return np.array(self.feature_names_in_)
