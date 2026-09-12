"""The model plugin contract.

Everything model-specific lives behind this interface. Orchestration code — the jobs, the
state machine, the API, the UI — never branches on a model name.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml_engine.contracts.common import ClassWeighting, ProblemType
from ml_engine.contracts.model import (
    FeatureImportance,
    FeatureImportanceEntry,
    ModelDescriptor,
)


class ModelNotAvailableError(RuntimeError):
    """Raised when a plugin's optional library is not installed in this environment."""


@dataclass(slots=True)
class TrainingContext:
    """Everything a plugin may need to choose sane defaults, without seeing the raw data."""

    problem_type: ProblemType
    random_seed: int = 42
    row_count: int = 0
    feature_count: int = 0
    class_labels: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    class_weighting: ClassWeighting = ClassWeighting.AUTO
    categorical_features: list[str] = field(default_factory=list)

    @property
    def class_count(self) -> int:
        return len(self.class_labels)

    @property
    def wants_class_weighting(self) -> bool:
        return (
            self.class_weighting is ClassWeighting.AUTO
            and self.problem_type.is_classification
            and bool(self.class_counts)
        )

    def positive_negative_counts(self) -> tuple[int, int] | None:
        """(positive, negative) counts for a binary problem, ordered by label."""
        if self.problem_type is not ProblemType.BINARY_CLASSIFICATION or len(self.class_counts) != 2:
            return None
        ordered = sorted(self.class_counts.items(), key=lambda item: item[0])
        negative, positive = ordered[0][1], ordered[1][1]
        return positive, negative


class ModelPlugin(ABC):
    """A trainable classical-ML model.

    Implementations declare what they support and how they behave; the training job does the
    same thing for every plugin.
    """

    name: str = ""
    display_name: str = ""
    library: str = ""
    description: str = ""
    supported_problem_types: tuple[ProblemType, ...] = ()
    requires_dense_numeric: bool = True
    supports_native_categorical: bool = False
    supports_class_weighting: bool = True

    # --- availability -----------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """Optional dependencies report their absence instead of breaking the platform."""
        return True, None

    @abstractmethod
    def library_version(self) -> str: ...

    # --- capabilities -----------------------------------------------------
    def supports(self, problem_type: ProblemType) -> bool:
        return problem_type in self.supported_problem_types

    def supports_probabilities(self, problem_type: ProblemType) -> bool:
        return problem_type.is_classification

    @property
    def preprocessing_strategy(self) -> str:
        """Which fitted preprocessing pipeline this plugin consumes."""
        return "native_categorical" if self.supports_native_categorical else "dense_numeric"

    # --- lifecycle --------------------------------------------------------
    @abstractmethod
    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        """Conservative defaults. No hyperparameter search in this version."""

    @abstractmethod
    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:
        """Return an unfitted estimator configured with ``params``."""

    def fit(
        self,
        estimator: Any,
        x: pd.DataFrame,
        y: pd.Series,
        context: TrainingContext,  # noqa: ARG002
    ) -> Any:
        estimator.fit(x, y)
        return estimator

    def predict(self, estimator: Any, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(estimator.predict(x))

    def predict_proba(self, estimator: Any, x: pd.DataFrame) -> np.ndarray | None:
        if not hasattr(estimator, "predict_proba"):
            return None
        return np.asarray(estimator.predict_proba(x))

    def class_labels(self, estimator: Any) -> list[str] | None:
        classes = getattr(estimator, "classes_", None)
        return None if classes is None else [str(c) for c in classes]

    # --- interpretability -------------------------------------------------
    def feature_importance(
        self, estimator: Any, feature_names: list[str]
    ) -> FeatureImportance | None:
        """Default: native ``feature_importances_`` when the library exposes it."""
        raw = getattr(estimator, "feature_importances_", None)
        if raw is None:
            return None
        return normalize_importance(
            np.asarray(raw, dtype="float64"), feature_names, method="native_importance"
        )

    def resolve_params(
        self, overrides: dict[str, Any] | None, context: TrainingContext
    ) -> dict[str, Any]:
        """Plugin defaults, then user overrides. Unknown keys are the plugin's problem."""
        params = self.default_params(context)
        params.update(overrides or {})
        return params

    def descriptor(self, problem_type: ProblemType | None = None) -> ModelDescriptor:
        available, reason = self.is_available()
        return ModelDescriptor(
            name=self.name,
            display_name=self.display_name,
            library=self.library,
            library_version=self.library_version() if available else None,
            supported_problem_types=list(self.supported_problem_types),
            requires_dense_numeric=self.requires_dense_numeric,
            supports_native_categorical=self.supports_native_categorical,
            supports_class_weighting=self.supports_class_weighting,
            supports_probabilities=self.supports_probabilities(
                problem_type or ProblemType.BINARY_CLASSIFICATION
            ),
            available=available,
            unavailable_reason=reason,
            default_parameters=_json_safe(
                self.default_params(TrainingContext(problem_type=problem_type or ProblemType.BINARY_CLASSIFICATION))
            )
            if available
            else {},
            description=self.description,
        )


def _json_safe(params: dict[str, Any]) -> dict[str, float | int | str | bool | None]:
    safe: dict[str, float | int | str | bool | None] = {}
    for key, value in params.items():
        safe[key] = value if isinstance(value, (int, float, str, bool)) or value is None else str(value)
    return safe


def normalize_importance(
    values: np.ndarray,
    feature_names: list[str],
    *,
    method: str,
    is_signed: bool = False,
    note: str | None = None,
) -> FeatureImportance:
    """Normalize any importance vector to a comparable, ranked representation.

    ``importance`` is the share of total absolute magnitude; ``raw_value`` keeps the
    library's own number (including sign, for linear coefficients).
    """
    if values.ndim > 1:
        # Multiclass linear coefficients: aggregate magnitude across classes.
        values = np.abs(values).mean(axis=0)
    magnitudes = np.abs(values)
    total = float(magnitudes.sum())
    entries = [
        FeatureImportanceEntry(
            feature=name,
            importance=float(magnitude / total) if total > 0 else 0.0,
            raw_value=float(raw),
            rank=0,
        )
        for name, magnitude, raw in zip(feature_names, magnitudes, values, strict=False)
    ]
    entries.sort(key=lambda entry: abs(entry.raw_value), reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry.rank = rank
    return FeatureImportance(method=method, is_signed=is_signed, entries=entries, note=note)
