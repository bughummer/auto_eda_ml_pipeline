"""Train one model and produce its :class:`ModelMetadata` artifact.

This is the deterministic core of the training job: no AWS, no IO, no orchestration. The
job entrypoint reads artifacts, calls this, and writes the result back.
"""

import contextlib
import platform
import sys
import time
import warnings as python_warnings
from datetime import UTC, datetime

import pandas as pd

from ml_engine.contracts.common import (
    ClassWeighting,
    ModelRunStatus,
    ProblemType,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.model import ModelMetadata, PreprocessingMetadata
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.evaluation import evaluate, resolve_primary_metric
from ml_engine.models import ModelPlugin, TrainingContext
from ml_engine.preprocessing import FeaturePipeline

OVERFITTING_GAP = 0.15

# Library warnings worth showing a data scientist, mapped to stable rule ids.
_REPORTED_FIT_WARNINGS = {
    "ConvergenceWarning": "model_did_not_converge",
    "DataConversionWarning": "input_data_converted",
    "UndefinedMetricWarning": "undefined_metric_during_fit",
}


class TrainingError(RuntimeError):
    """Raised when a model cannot be trained at all."""


def build_training_context(
    *,
    problem_type: ProblemType,
    y_train: pd.Series,
    feature_count: int,
    random_seed: int,
    class_weighting: ClassWeighting,
    categorical_features: list[str],
) -> TrainingContext:
    class_labels: list[str] = []
    class_counts: dict[str, int] = {}
    if problem_type.is_classification:
        counts = y_train.astype("string").value_counts()
        class_labels = sorted(str(label) for label in counts.index)
        class_counts = {str(label): int(count) for label, count in counts.items()}
    return TrainingContext(
        problem_type=problem_type,
        random_seed=random_seed,
        row_count=len(y_train),
        feature_count=feature_count,
        class_labels=class_labels,
        class_counts=class_counts,
        class_weighting=class_weighting,
        categorical_features=categorical_features,
    )


def train_model(
    plugin: ModelPlugin,
    *,
    experiment_id: str,
    problem_type: ProblemType,
    pipeline: FeaturePipeline,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_column: str,
    param_overrides: dict | None = None,
    primary_metric: str | None = None,
    random_seed: int = 42,
    class_weighting: ClassWeighting = ClassWeighting.AUTO,
    compute_train_metrics: bool = True,
) -> tuple[ModelMetadata, object]:
    """Fit, evaluate and describe one model. Returns the metadata and the fitted estimator.

    ``pipeline`` must already be fitted on the training fold. It is applied, never re-fitted,
    so validation data can never influence a learned transform.
    """
    if not plugin.supports(problem_type):
        raise TrainingError(
            f"Model '{plugin.name}' does not support {problem_type.value}. "
            f"Supported: {', '.join(p.value for p in plugin.supported_problem_types)}."
        )
    available, reason = plugin.is_available()
    if not available:
        raise TrainingError(f"Model '{plugin.name}' is unavailable in this environment: {reason}")
    if not pipeline.is_fitted:
        raise TrainingError("The preprocessing pipeline must be fitted before training.")

    warnings: list[AnalysisWarning] = []
    y_train = train_frame[target_column]
    y_validation = validation_frame[target_column]
    if problem_type.is_classification:
        y_train = y_train.astype("string")
        y_validation = y_validation.astype("string")

    x_train = pipeline.transform(train_frame)
    x_validation = pipeline.transform(validation_frame)

    context = build_training_context(
        problem_type=problem_type,
        y_train=y_train,
        feature_count=x_train.shape[1],
        random_seed=random_seed,
        class_weighting=class_weighting,
        categorical_features=pipeline.categorical_feature_names,
    )
    if (
        class_weighting is ClassWeighting.AUTO
        and problem_type.is_classification
        and not plugin.supports_class_weighting
    ):
        warnings.append(
            AnalysisWarning(
                rule="class_weighting_unsupported",
                category=WarningCategory.MODEL,
                severity=Severity.INFO,
                message=(
                    f"Class weighting was requested but '{plugin.display_name}' does not support "
                    "it; the model was trained unweighted."
                ),
            )
        )

    params = plugin.resolve_params(param_overrides, context)
    estimator = plugin.build(params, context)

    started = datetime.now(UTC)
    clock = time.perf_counter()
    try:
        with python_warnings.catch_warnings(record=True) as captured:
            python_warnings.simplefilter("always")
            estimator = plugin.fit(estimator, x_train, y_train, context)
    except Exception as exc:
        raise TrainingError(f"{plugin.display_name} failed during fit: {exc}") from exc
    duration = time.perf_counter() - clock
    finished = datetime.now(UTC)
    warnings.extend(_fit_warnings(captured, plugin.display_name))

    class_labels = plugin.class_labels(estimator) if problem_type.is_classification else None
    positive_class = (
        _positive_class(class_labels, context)
        if problem_type is ProblemType.BINARY_CLASSIFICATION
        else None
    )

    validation_metrics = evaluate(
        problem_type,
        y_validation,
        plugin.predict(estimator, x_validation),
        y_proba=plugin.predict_proba(estimator, x_validation),
        class_labels=class_labels,
        positive_class=positive_class,
    )
    train_metrics = None
    if compute_train_metrics:
        train_metrics = evaluate(
            problem_type,
            y_train,
            plugin.predict(estimator, x_train),
            y_proba=plugin.predict_proba(estimator, x_train),
            class_labels=class_labels,
            positive_class=positive_class,
        )

    metric_name = resolve_primary_metric(problem_type, primary_metric)
    primary_score = validation_metrics.get(metric_name)
    if primary_score is None:
        warnings.append(
            AnalysisWarning(
                rule="primary_metric_unavailable",
                category=WarningCategory.METRIC,
                severity=Severity.HIGH,
                message=(
                    f"The primary metric '{metric_name}' could not be computed for "
                    f"'{plugin.display_name}', so this model cannot be ranked."
                ),
                details={"metric": metric_name},
            )
        )
    warnings.extend(validation_metrics.warnings)
    if train_metrics is not None:
        warnings.extend(_overfitting_warnings(metric_name, train_metrics, validation_metrics))

    importance = plugin.feature_importance(estimator, pipeline.feature_names)

    metadata = ModelMetadata(
        experiment_id=experiment_id,
        model_name=plugin.name,
        display_name=plugin.display_name,
        status=ModelRunStatus.COMPLETED,
        problem_type=problem_type,
        library=plugin.library,
        library_version=plugin.library_version(),
        hyperparameters=_json_safe(params),
        features_used=pipeline.feature_names,
        feature_count=len(pipeline.feature_names),
        training_row_count=len(train_frame),
        validation_row_count=len(validation_frame),
        training_started_at=started,
        training_completed_at=finished,
        training_duration_seconds=round(duration, 4),
        primary_metric=metric_name,
        primary_score=primary_score,
        metrics=validation_metrics,
        train_metrics=train_metrics,
        feature_importance=importance,
        preprocessing=pipeline.metadata(),
        package_versions=package_versions(plugin),
        warnings=warnings,
    )
    return metadata, estimator


def _positive_class(class_labels: list[str] | None, context: TrainingContext) -> str | None:
    labels = class_labels or context.class_labels
    if not labels:
        return None
    lowered = {label.strip().lower(): label for label in labels}
    for token in ("1", "true", "yes", "y", "t"):
        if token in lowered:
            return lowered[token]
    if context.class_counts:
        present = {k: v for k, v in context.class_counts.items() if k in labels}
        if present:
            return min(present, key=lambda label: present[label])
    return labels[-1]


def _fit_warnings(captured, display_name: str) -> list[AnalysisWarning]:
    """Surface library warnings that change how a result should be read (e.g. non-convergence)."""
    collected: list[AnalysisWarning] = []
    seen: set[str] = set()
    for entry in captured:
        category = entry.category.__name__
        if category not in _REPORTED_FIT_WARNINGS or category in seen:
            continue
        seen.add(category)
        collected.append(
            AnalysisWarning(
                rule=_REPORTED_FIT_WARNINGS[category],
                category=WarningCategory.MODEL,
                severity=Severity.MEDIUM,
                message=f"{display_name}: {str(entry.message).splitlines()[0]}",
                details={"warning_class": category},
            )
        )
    return collected


def _overfitting_warnings(metric: str, train_metrics, validation_metrics) -> list[AnalysisWarning]:
    train_score = train_metrics.get(metric)
    validation_score = validation_metrics.get(metric)
    if train_score is None or validation_score is None:
        return []
    gap = abs(train_score - validation_score)
    scale = max(abs(train_score), 1e-9)
    if gap / scale < OVERFITTING_GAP:
        return []
    return [
        AnalysisWarning(
            rule="large_train_validation_gap",
            category=WarningCategory.MODEL,
            severity=Severity.MEDIUM,
            message=(
                f"Training {metric} ({train_score:.4f}) differs from validation {metric} "
                f"({validation_score:.4f}) by more than {OVERFITTING_GAP:.0%}. The model is "
                "likely overfitting."
            ),
            details={"metric": metric, "train": train_score, "validation": validation_score},
        )
    ]


def _json_safe(params: dict) -> dict:
    return {
        key: value if isinstance(value, (int, float, str, bool)) or value is None else str(value)
        for key, value in params.items()
    }


def package_versions(plugin: ModelPlugin | None = None) -> dict[str, str]:
    """Library versions that materially affect results. Stored with every model."""
    import numpy
    import pandas
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }
    if plugin is not None:
        with contextlib.suppress(Exception):  # version reporting is never fatal
            versions[plugin.library] = plugin.library_version()
    return versions


def environment_metadata() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def empty_preprocessing_metadata() -> PreprocessingMetadata:
    return PreprocessingMetadata()
