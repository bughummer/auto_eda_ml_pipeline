"""CatBoost plugins.

The one plugin that consumes categorical columns natively, so it runs on the
``native_categorical`` preprocessing strategy instead of one-hot encoded input.
"""

from typing import Any

import numpy as np
import pandas as pd

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.model import FeatureImportance
from ml_engine.models.base import (
    ModelNotAvailableError,
    ModelPlugin,
    TrainingContext,
    normalize_importance,
)

try:  # pragma: no cover - environment dependent
    import catboost

    _CATBOOST_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - environment dependent
    catboost = None  # type: ignore[assignment]
    _CATBOOST_ERROR = f"catboost is not installed ({exc})"


class _CatBoostBase(ModelPlugin):
    library = "catboost"
    requires_dense_numeric = False
    supports_native_categorical = True

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        return (catboost is not None), _CATBOOST_ERROR

    def library_version(self) -> str:
        if catboost is None:
            raise ModelNotAvailableError(_CATBOOST_ERROR or "catboost is not installed")
        return catboost.__version__

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        return {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": context.random_seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }

    def fit(
        self, estimator: Any, x: pd.DataFrame, y: pd.Series, context: TrainingContext
    ) -> Any:
        categorical = [c for c in context.categorical_features if c in x.columns]
        estimator.fit(x, y, cat_features=categorical or None)
        return estimator

    def feature_importance(
        self, estimator: Any, feature_names: list[str]
    ) -> FeatureImportance | None:
        try:
            values = np.asarray(estimator.get_feature_importance(), dtype="float64")
        except Exception:  # noqa: BLE001 - importance is best effort, never fatal
            return None
        return normalize_importance(
            values,
            feature_names,
            method="catboost_prediction_values_change",
            note="CatBoost PredictionValuesChange importance over the native feature space.",
        )

    def _guard(self) -> None:
        if catboost is None:
            raise ModelNotAvailableError(_CATBOOST_ERROR or "catboost is not installed")


class CatBoostClassifierPlugin(_CatBoostBase):
    name = "catboost"
    display_name = "CatBoost"
    description = (
        "Gradient boosting with native categorical handling — no one-hot expansion, so "
        "high-cardinality categories stay usable."
    )
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    supports_class_weighting = True

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        if context.wants_class_weighting:
            params["auto_class_weights"] = "Balanced"
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        self._guard()
        return catboost.CatBoostClassifier(**params)


class CatBoostRegressorPlugin(_CatBoostBase):
    name = "catboost_regressor"
    display_name = "CatBoost Regressor"
    description = "CatBoost regression with native categorical handling."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def predict_proba(self, estimator: Any, x: pd.DataFrame) -> np.ndarray | None:  # noqa: ARG002
        return None

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        self._guard()
        return catboost.CatBoostRegressor(**params)
