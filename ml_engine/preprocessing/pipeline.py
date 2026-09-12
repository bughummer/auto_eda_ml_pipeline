"""The fitted preprocessing artifact.

Two strategies, one interface:

``dense_numeric``      imputation + scaling + one-hot encoding, for models that need a dense
                       numeric matrix (linear models, scikit-learn forests, XGBoost).
``native_categorical`` imputation only, categories kept as strings, for libraries with
                       native categorical support (CatBoost).

Learned state (imputation values, scaler statistics, encoder categories) is fitted on the
training fold exclusively. ``transform`` never re-fits.
"""

import hashlib
import json
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_engine.contracts.config import PreprocessingConfig
from ml_engine.contracts.model import PreprocessingMetadata
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.preprocessing.planning import ColumnPlan
from ml_engine.preprocessing.transformers import (
    BooleanEncoder,
    CategoricalAsString,
    DatetimeFeatureExtractor,
)

Strategy = Literal["dense_numeric", "native_categorical"]


class PreprocessingError(RuntimeError):
    """Raised when no usable feature survives preprocessing."""


class FeaturePipeline:
    """Wraps a fitted scikit-learn transformer plus the metadata describing what it did."""

    def __init__(
        self,
        plan: ColumnPlan,
        config: PreprocessingConfig,
        strategy: Strategy = "dense_numeric",
    ) -> None:
        if plan.is_empty:
            raise PreprocessingError(
                "No usable features remain after preprocessing. Review the feature selection: "
                f"dropped columns were {sorted(plan.dropped)}."
            )
        self.plan = plan
        self.config = config
        self.strategy: Strategy = strategy
        self._transformer = _build_transformer(plan, config, strategy)
        self._fitted = False
        self._feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_names(self) -> list[str]:
        self._require_fitted()
        return list(self._feature_names)

    @property
    def categorical_feature_names(self) -> list[str]:
        """Columns a native-categorical model should treat as categories."""
        return list(self.plan.categorical) if self.strategy == "native_categorical" else []

    def fit(self, frame: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePipeline":
        """Fit on the training fold only. Calling this with validation data is a bug."""
        self._transformer.fit(frame[self.plan.usable], y)
        self._feature_names = self._resolve_feature_names()
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        transformed = self._transformer.transform(frame[self.plan.usable])
        if isinstance(transformed, pd.DataFrame):
            return transformed
        dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        return pd.DataFrame(dense, columns=self._feature_names, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(frame, y).transform(frame)

    def metadata(self) -> PreprocessingMetadata:
        self._require_fitted()
        return PreprocessingMetadata(
            numeric_columns=list(self.plan.numeric),
            categorical_columns=list(self.plan.categorical),
            boolean_columns=list(self.plan.boolean),
            datetime_columns=list(self.plan.datetime),
            derived_datetime_features=list(self.config.datetime_features)
            if self.plan.datetime
            else [],
            dropped_columns=dict(self.plan.dropped),
            output_feature_names=list(self._feature_names),
            output_feature_count=len(self._feature_names),
            strategy=self.strategy,
            fitted_on="train",
            config_digest=self.config_digest(),
        )

    def warnings(self) -> list[AnalysisWarning]:
        return list(self.plan.warnings)

    def config_digest(self) -> str:
        """Stable digest of the preprocessing definition, for reproducibility checks."""
        payload = {
            "strategy": self.strategy,
            "config": self.config.model_dump(),
            "numeric": self.plan.numeric,
            "categorical": self.plan.categorical,
            "boolean": self.plan.boolean,
            "datetime": self.plan.datetime,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise PreprocessingError("FeaturePipeline must be fitted before it can be used.")

    def _resolve_feature_names(self) -> list[str]:
        try:
            return [str(name) for name in self._transformer.get_feature_names_out()]
        except (AttributeError, ValueError):  # pragma: no cover - defensive
            return list(self.plan.usable)


def _build_transformer(
    plan: ColumnPlan, config: PreprocessingConfig, strategy: Strategy
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", _numeric_imputer(config))]
    if strategy == "dense_numeric" and config.scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    blocks: list[tuple[str, Pipeline, list[str]]] = []
    if plan.numeric:
        blocks.append(("numeric", Pipeline(numeric_steps), plan.numeric))
    if plan.boolean:
        blocks.append(
            (
                "boolean",
                Pipeline(
                    [
                        ("encode", BooleanEncoder()),
                        ("impute", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                plan.boolean,
            )
        )
    if plan.datetime:
        blocks.append(
            (
                "datetime",
                Pipeline(
                    [
                        ("derive", DatetimeFeatureExtractor(tuple(config.datetime_features))),
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                plan.datetime,
            )
        )
    if plan.categorical:
        blocks.append(("categorical", _categorical_pipeline(config, strategy), plan.categorical))

    transformer = ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)
    if strategy == "native_categorical":
        # Keep a DataFrame so category columns survive as strings for the model library.
        transformer.set_output(transform="pandas")
    return transformer


def _numeric_imputer(config: PreprocessingConfig) -> SimpleImputer:
    if config.numeric_imputation == "constant":
        return SimpleImputer(strategy="constant", fill_value=config.numeric_fill_value)
    return SimpleImputer(strategy=config.numeric_imputation)


def _categorical_pipeline(config: PreprocessingConfig, strategy: Strategy) -> Pipeline:
    imputer = (
        SimpleImputer(strategy="most_frequent")
        if config.categorical_imputation == "most_frequent"
        else SimpleImputer(strategy="constant", fill_value=config.categorical_fill_value)
    )
    if strategy == "native_categorical":
        return Pipeline([("as_string", CategoricalAsString(config.categorical_fill_value))])
    return Pipeline(
        [
            ("as_string", CategoricalAsString(config.categorical_fill_value)),
            ("impute", imputer),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=config.one_hot_max_categories,
                    min_frequency=config.one_hot_min_frequency,
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )
