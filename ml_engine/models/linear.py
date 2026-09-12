"""Linear baselines: logistic regression (classification) and elastic net (regression).

Every experiment should contain a linear baseline. If a gradient-boosted model cannot beat
a regularized linear model, the extra complexity is not paying for itself.
"""

from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet, LogisticRegression

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.model import FeatureImportance
from ml_engine.models.base import ModelPlugin, TrainingContext, normalize_importance


class LogisticRegressionPlugin(ModelPlugin):
    name = "logistic_regression"
    display_name = "Logistic Regression"
    library = "scikit-learn"
    description = (
        "Regularized linear classifier. Fast, stable and the reference baseline every other "
        "model must beat to justify its complexity."
    )
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    requires_dense_numeric = True
    supports_class_weighting = True

    def library_version(self) -> str:
        return sklearn.__version__

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params: dict[str, Any] = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": context.random_seed,
            "n_jobs": None,
        }
        if context.wants_class_weighting:
            params["class_weight"] = "balanced"
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return LogisticRegression(**params)

    def feature_importance(
        self, estimator: Any, feature_names: list[str]
    ) -> FeatureImportance | None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return None
        values = np.asarray(coefficients, dtype="float64")
        is_binary = values.ndim == 2 and values.shape[0] == 1
        return normalize_importance(
            values[0] if is_binary else values,
            feature_names,
            method="coefficient_magnitude",
            is_signed=is_binary,
            note=(
                "Coefficients are comparable because numeric features are standardized. "
                "Sign indicates direction for binary problems; multiclass values are mean "
                "absolute coefficients across classes."
            ),
        )


class ElasticNetPlugin(ModelPlugin):
    name = "elastic_net"
    display_name = "Elastic Net Regression"
    library = "scikit-learn"
    description = (
        "L1/L2-regularized linear regression. The interpretable regression baseline; "
        "sparse coefficients make redundant features visible."
    )
    supported_problem_types = (ProblemType.REGRESSION,)
    requires_dense_numeric = True
    supports_class_weighting = False

    def library_version(self) -> str:
        return sklearn.__version__

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        return {
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "max_iter": 5000,
            "random_state": context.random_seed,
            "selection": "cyclic",
        }

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return ElasticNet(**params)

    def predict_proba(self, estimator: Any, x: pd.DataFrame) -> np.ndarray | None:  # noqa: ARG002
        return None

    def feature_importance(
        self, estimator: Any, feature_names: list[str]
    ) -> FeatureImportance | None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return None
        return normalize_importance(
            np.asarray(coefficients, dtype="float64"),
            feature_names,
            method="coefficient_magnitude",
            is_signed=True,
            note="Standardized-feature coefficients; zero means the penalty removed the feature.",
        )
