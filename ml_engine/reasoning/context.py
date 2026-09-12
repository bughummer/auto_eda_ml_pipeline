"""Assemble the deterministic context the reasoning model is allowed to see.

The model receives *results*, never raw data: no rows, no values beyond capped category
samples, and every number already computed by pandas or scikit-learn. Context is bounded so
one enormous dataset cannot blow the token budget or the cost of an experiment.
"""

from typing import Any

from ml_engine.contracts.comparison import ExperimentSummary
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport

MAX_COLUMNS = 80
MAX_WARNINGS = 40
MAX_FINDINGS = 40
MAX_IMPORTANCE = 25


def build_context(
    *,
    eda: EdaReport,
    leakage: LeakageReport | None,
    dictionary: DataDictionary | None,
    summary: ExperimentSummary | None,
    target_definition: str | None = None,
    prediction_timing: str | None = None,
) -> dict[str, Any]:
    """Build the JSON document handed to the model as evidence."""
    context: dict[str, Any] = {
        "experiment_id": eda.experiment_id,
        "target": {
            "column": eda.target.column,
            "problem_type": eda.target.inferred_problem_type.value
            if eda.target.inferred_problem_type
            else None,
            "business_definition": target_definition,
            "prediction_timing": prediction_timing,
            "class_distribution": [
                entry.model_dump() for entry in (eda.target.class_distribution or [])
            ],
            "imbalance_ratio": eda.target.imbalance_ratio,
        },
        "dataset": {
            "row_count": eda.dataset.row_count,
            "column_count": eda.dataset.column_count,
            "duplicate_row_percentage": eda.dataset.duplicate_row_percentage,
            "sampled": eda.dataset.sampled,
        },
        "columns": [_column(profile) for profile in eda.columns[:MAX_COLUMNS]],
        "eda_warnings": [
            {
                "rule": w.rule,
                "category": w.category.value,
                "severity": w.severity.value,
                "column": w.column,
                "message": w.message,
            }
            for w in eda.warnings[:MAX_WARNINGS]
        ],
    }
    if len(eda.columns) > MAX_COLUMNS:
        context["columns_truncated"] = len(eda.columns) - MAX_COLUMNS

    if leakage is not None:
        context["deterministic_leakage_findings"] = [
            {
                "feature": f.feature,
                "rule": f.rule,
                "risk_level": f.risk_level.value,
                "severity": f.severity.value,
                "explanation": f.explanation,
            }
            for f in leakage.findings[:MAX_FINDINGS]
        ]

    if dictionary is not None:
        context["column_documentation"] = [
            {
                "column": doc.column,
                "business_definition": doc.business_definition,
                "source_system": doc.source_system,
                "collection_timing": doc.collection_timing,
                "update_frequency": doc.update_frequency,
                "available_at_prediction_time": doc.available_at_prediction_time,
            }
            for doc in dictionary.columns[:MAX_COLUMNS]
        ]
        context["undocumented_columns"] = dictionary.undocumented_columns[:MAX_COLUMNS]

    if summary is not None and summary.comparison is not None:
        context["model_results"] = {
            "primary_metric": summary.comparison.primary_metric,
            "direction": summary.comparison.direction.value,
            "best_model": summary.comparison.best_model,
            "models": [
                {
                    "model": entry.model_name,
                    "status": entry.status.value,
                    "primary_score": entry.primary_score,
                    "metrics": entry.metrics,
                }
                for entry in summary.comparison.models
            ],
        }
        context["selected_features"] = summary.config.feature_selection.selected_features[
            :MAX_COLUMNS
        ]
        context["excluded_features"] = summary.config.feature_selection.excluded_features[
            :MAX_COLUMNS
        ]
        if summary.best_model_importance is not None:
            context["best_model_feature_importance"] = [
                {"feature": entry.feature, "importance": round(entry.importance, 6)}
                for entry in summary.best_model_importance.entries[:MAX_IMPORTANCE]
            ]
    return context


def _column(profile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "semantic_type": profile.semantic_type.value,
        "missing_percentage": profile.missing_percentage,
        "unique_count": profile.unique_count,
        "is_constant": profile.is_constant,
        "is_likely_id": profile.is_likely_id,
        "is_high_cardinality": profile.is_high_cardinality,
    }


def referenced_columns(context: dict[str, Any]) -> set[str]:
    """Column names the model is permitted to talk about."""
    return {column["name"] for column in context.get("columns", [])}
