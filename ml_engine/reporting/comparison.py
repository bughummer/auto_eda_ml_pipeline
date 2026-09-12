"""Cross-model comparison and the experiment summary.

The only place that decides which model is "best", using the declared direction of the
primary metric. Failed models are represented, never dropped.
"""

from datetime import UTC, datetime

from ml_engine.contracts.common import (
    ExperimentStatus,
    ModelRunStatus,
    ProblemType,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.comparison import (
    ComparisonReport,
    ExperimentSummary,
    ModelComparisonEntry,
)
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.contracts.warnings import AnalysisWarning, sort_warnings
from ml_engine.evaluation import metric_direction, resolve_primary_metric
from ml_engine.evaluation.metrics import MetricDirection

# A validation score this close to perfect is almost always leakage, not a good model.
SUSPICIOUSLY_PERFECT = 0.999
BOUNDED_METRICS = frozenset(
    {"roc_auc", "pr_auc", "accuracy", "f1", "macro_f1", "weighted_f1", "r2"}
)


def build_comparison(
    *,
    experiment_id: str,
    problem_type: ProblemType,
    primary_metric: str | None,
    successes: list[ModelMetadata],
    failures: list[ModelFailure],
) -> ComparisonReport:
    """Rank successful models and record failures alongside them."""
    metric = resolve_primary_metric(problem_type, primary_metric)
    direction = metric_direction(problem_type, metric)

    entries = [
        ModelComparisonEntry(
            model_name=metadata.model_name,
            display_name=metadata.display_name,
            status=metadata.status,
            primary_score=metadata.metrics.get(metric),
            metrics=dict(metadata.metrics.values),
            training_duration_seconds=metadata.training_duration_seconds,
            feature_count=metadata.feature_count,
            warning_count=len(metadata.warnings),
        )
        for metadata in successes
    ]

    ranked = [entry for entry in entries if entry.primary_score is not None]
    ranked.sort(
        key=lambda entry: entry.primary_score,
        reverse=direction is MetricDirection.MAXIMIZE,
    )
    for rank, entry in enumerate(ranked, start=1):
        entry.rank = rank

    entries.extend(
        ModelComparisonEntry(
            model_name=failure.model_name,
            display_name=failure.model_name,
            status=ModelRunStatus.FAILED,
            failure_message=failure.message,
        )
        for failure in failures
    )
    entries.sort(key=lambda entry: (entry.rank is None, entry.rank or 0, entry.model_name))

    warnings: list[AnalysisWarning] = []
    if failures:
        warnings.append(
            AnalysisWarning(
                rule="model_training_failed",
                category=WarningCategory.MODEL,
                severity=Severity.MEDIUM,
                message=(
                    f"{len(failures)} of {len(successes) + len(failures)} models failed: "
                    + ", ".join(f.model_name for f in failures)
                    + ". The experiment completed with the remaining models."
                ),
                details={"failed_models": ", ".join(f.model_name for f in failures)},
            )
        )
    if not ranked:
        warnings.append(
            AnalysisWarning(
                rule="no_rankable_model",
                category=WarningCategory.MODEL,
                severity=Severity.HIGH,
                message=(
                    f"No model produced the primary metric '{metric}', so no best model could "
                    "be selected."
                ),
            )
        )
    if (
        ranked
        and metric in BOUNDED_METRICS
        and ranked[0].primary_score is not None
        and ranked[0].primary_score >= SUSPICIOUSLY_PERFECT
    ):
        warnings.append(
            AnalysisWarning(
                rule="suspiciously_perfect_score",
                category=WarningCategory.LEAKAGE,
                severity=Severity.HIGH,
                message=(
                    f"The best model reaches {metric}={ranked[0].primary_score:.4f} on validation "
                    "data. Scores this close to perfect nearly always mean a feature encodes the "
                    "outcome. Re-check the leakage findings before trusting this result."
                ),
                details={"metric": metric, "score": ranked[0].primary_score},
            )
        )

    if len(ranked) > 1:
        best, runner_up = ranked[0].primary_score, ranked[1].primary_score
        if best is not None and runner_up is not None and abs(best - runner_up) < 1e-6:
            warnings.append(
                AnalysisWarning(
                    rule="models_statistically_indistinguishable",
                    category=WarningCategory.MODEL,
                    severity=Severity.INFO,
                    message=(
                        f"The top two models differ by less than 1e-6 on '{metric}'. Prefer the "
                        "simpler or faster model."
                    ),
                )
            )

    return ComparisonReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        problem_type=problem_type,
        primary_metric=metric,
        direction=direction,
        best_model=ranked[0].model_name if ranked else None,
        best_score=ranked[0].primary_score if ranked else None,
        models=entries,
        succeeded_count=len(successes),
        failed_count=len(failures),
        warnings=warnings,
    )


def final_status(comparison: ComparisonReport) -> ExperimentStatus:
    """An experiment with a usable result is not a failure, even if a model failed."""
    if comparison.succeeded_count == 0:
        return ExperimentStatus.FAILED
    if comparison.failed_count or comparison.warnings:
        return ExperimentStatus.COMPLETED_WITH_WARNINGS
    return ExperimentStatus.COMPLETED


def build_summary(
    *,
    config: ExperimentConfig,
    status: ExperimentStatus,
    comparison: ComparisonReport | None,
    eda: EdaReport | None,
    leakage: LeakageReport | None,
    models: list[ModelMetadata],
) -> ExperimentSummary:
    """Assemble the single artifact that reporting and the reasoning layer consume."""
    best_importance = None
    if comparison and comparison.best_model:
        best = next((m for m in models if m.model_name == comparison.best_model), None)
        best_importance = best.feature_importance if best else None

    collected: list[AnalysisWarning] = []
    if comparison:
        collected.extend(comparison.warnings)
    for metadata in models:
        collected.extend(metadata.warnings)

    return ExperimentSummary(
        experiment_id=config.experiment_id,
        name=config.name,
        generated_at=datetime.now(UTC),
        status=status,
        config=config,
        dataset_row_count=eda.dataset.row_count if eda else None,
        dataset_column_count=eda.dataset.column_count if eda else None,
        selected_feature_count=len(config.feature_selection.selected_features),
        excluded_feature_count=len(config.feature_selection.excluded_features),
        comparison=comparison,
        best_model_importance=best_importance,
        eda_warnings=eda.warnings if eda else [],
        leakage_warning_count=len(leakage.findings) if leakage else 0,
        warnings=sort_warnings(collected),
    )
