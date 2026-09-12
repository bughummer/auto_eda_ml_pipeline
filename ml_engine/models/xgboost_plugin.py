"""XGBoost plugins. Optional dependency: absent XGBoost degrades the catalogue, nothing else."""

from typing import Any

import numpy as np
import pandas as pd

from ml_engine.contracts.common import ProblemType
from ml_engine.models.base import ModelNotAvailableError, ModelPlugin, TrainingContext
from ml_engine.models.label_encoding import LabelEncodedClassifier

try:  # pragma: no cover - environment dependent
    import xgboost as xgb

    _XGBOOST_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - environment dependent
    xgb = None  # type: ignore[assignment]
    _XGBOOST_ERROR = f"xgboost is not installed ({exc})"


class _XGBoostBase(ModelPlugin):
    library = "xgboost"
    requires_dense_numeric = True

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        return (xgb is not None), _XGBOOST_ERROR

    def library_version(self) -> str:
        if xgb is None:
            raise ModelNotAvailableError(_XGBOOST_ERROR or "xgboost is not installed")
        return xgb.__version__

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        # Conservative gradient boosting: shallow trees, moderate learning rate, subsampling
        # for variance reduction. No search — this is a baseline, not a tuned model.
        return {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": context.random_seed,
            "verbosity": 0,
        }

    def _guard(self) -> None:
        if xgb is None:
            raise ModelNotAvailableError(_XGBOOST_ERROR or "xgboost is not installed")


class XGBoostClassifierPlugin(_XGBoostBase):
    name = "xgboost"
    display_name = "XGBoost"
    description = (
        "Gradient-boosted trees. Usually the strongest classical model on tabular data; "
        "handles missing values natively."
    )
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    supports_class_weighting = True

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        params["eval_metric"] = (
            "logloss" if context.problem_type is ProblemType.BINARY_CLASSIFICATION else "mlogloss"
        )
        if context.wants_class_weighting:
            counts = context.positive_negative_counts()
            if counts:
                positive, negative = counts
                if positive > 0:
                    params["scale_pos_weight"] = round(negative / positive, 6)
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        self._guard()
        # XGBoost's sklearn API only accepts integer classes; labels stay strings everywhere else.
        return LabelEncodedClassifier(xgb.XGBClassifier(**params))

    def fit(
        self, estimator: Any, x: pd.DataFrame, y: pd.Series, context: TrainingContext
    ) -> Any:
        """Multiclass imbalance is handled with sample weights; binary uses scale_pos_weight."""
        if (
            context.wants_class_weighting
            and context.problem_type is ProblemType.MULTICLASS_CLASSIFICATION
        ):
            from sklearn.utils.class_weight import compute_sample_weight

            estimator.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
            return estimator
        estimator.fit(x, y)
        return estimator


class XGBoostRegressorPlugin(_XGBoostBase):
    name = "xgboost_regressor"
    display_name = "XGBoost Regressor"
    description = "Gradient-boosted regression trees; the strongest default regression baseline."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def predict_proba(self, estimator: Any, x: pd.DataFrame) -> np.ndarray | None:  # noqa: ARG002
        return None

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        params["eval_metric"] = "rmse"
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        self._guard()
        return xgb.XGBRegressor(**params)
