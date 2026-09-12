"""Random forest plugins. Strong, low-maintenance non-linear baselines."""

from typing import Any

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_engine.contracts.common import ProblemType
from ml_engine.models.base import ModelPlugin, TrainingContext


class _RandomForestBase(ModelPlugin):
    library = "scikit-learn"
    requires_dense_numeric = True

    def library_version(self) -> str:
        return sklearn.__version__

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        # Conservative: enough trees to be stable, leaf size scaled to dataset size to keep
        # small datasets from memorizing.
        min_samples_leaf = 1 if context.row_count > 10_000 else 2
        return {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": min_samples_leaf,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": context.random_seed,
        }


class RandomForestClassifierPlugin(_RandomForestBase):
    name = "random_forest"
    display_name = "Random Forest"
    description = (
        "Bagged decision trees. Handles non-linearities and interactions without tuning and "
        "is resistant to irrelevant features."
    )
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    supports_class_weighting = True

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        if context.wants_class_weighting:
            params["class_weight"] = "balanced_subsample"
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return RandomForestClassifier(**params)


class RandomForestRegressorPlugin(_RandomForestBase):
    name = "random_forest_regressor"
    display_name = "Random Forest Regressor"
    description = "Bagged regression trees; a robust non-linear regression baseline."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        params["max_features"] = 1.0
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return RandomForestRegressor(**params)
