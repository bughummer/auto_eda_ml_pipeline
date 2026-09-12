"""EDA must describe a dataset truthfully, and flag what a reviewer needs to see."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.contracts.common import ProblemType, SemanticType
from ml_engine.profiling import ProfilingConfig, infer_problem_type, profile_dataset


def _profile(frame: pd.DataFrame, target: str = "churned"):
    return profile_dataset(
        frame,
        experiment_id="exp-1",
        target_column=target,
        source_uri="memory://test",
        file_format="csv",
    )


def test_dataset_summary_counts_rows_columns_and_duplicates(classification_frame):
    frame = pd.concat([classification_frame, classification_frame.head(5)], ignore_index=True)
    report = _profile(frame)
    assert report.dataset.row_count == len(frame)
    assert report.dataset.column_count == frame.shape[1]
    assert report.dataset.duplicate_row_count == 5
    assert report.dataset.memory_usage_bytes > 0


def test_semantic_types_are_inferred_per_column(classification_eda):
    types = {c.name: c.semantic_type for c in classification_eda.columns}
    assert types["customer_id"] is SemanticType.IDENTIFIER
    assert types["tenure_months"] in {
        SemanticType.NUMERIC_CONTINUOUS,
        SemanticType.NUMERIC_DISCRETE,
    }
    assert types["contract"] is SemanticType.CATEGORICAL
    assert types["is_business"] is SemanticType.BOOLEAN
    assert types["signup_date"] is SemanticType.DATETIME
    assert types["region_code"] is SemanticType.CONSTANT


def test_numeric_statistics_are_complete_and_finite(classification_eda):
    charges = classification_eda.column("monthly_charges")
    assert charges.numeric is not None
    for field in ("min", "max", "mean", "median", "std", "q01", "q25", "q75", "q99"):
        assert getattr(charges.numeric, field) is not None
    assert charges.missing_count > 0
    assert 0 < charges.missing_percentage < 100


def test_categorical_statistics_are_capped(classification_frame):
    frame = classification_frame.copy()
    frame["many"] = [f"v{i}" for i in range(len(frame))]
    report = _profile(frame)
    stats = report.column("many").categorical
    assert stats is not None
    assert len(stats.top_values) <= ProfilingConfig().top_values
    assert stats.truncated is True


def test_target_analysis_reports_class_balance(classification_eda):
    target = classification_eda.target
    assert target.exists
    assert target.inferred_problem_type is ProblemType.BINARY_CLASSIFICATION
    assert target.class_distribution is not None
    assert target.imbalance_ratio is not None and target.imbalance_ratio >= 1.0
    assert abs(sum(entry.percentage for entry in target.class_distribution) - 100) < 0.01


def test_missing_target_column_is_critical():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    report = _profile(frame, target="nope")
    assert report.target.exists is False
    assert any(w.rule == "target_missing_from_dataset" for w in report.warnings)


def test_expected_warnings_are_raised(classification_eda):
    rules = {(w.rule, w.column) for w in classification_eda.warnings}
    assert ("likely_identifier", "customer_id") in rules
    assert ("constant_column", "region_code") in rules
    assert ("high_missingness", "monthly_charges") in rules
    assert ("text_column", "notes") in rules


def test_warnings_are_ordered_by_severity(classification_eda):
    from ml_engine.contracts.common import SEVERITY_ORDER

    severities = [SEVERITY_ORDER[w.severity] for w in classification_eda.warnings]
    assert severities == sorted(severities, reverse=True)


def test_regression_target_produces_numeric_summary(regression_frame):
    report = profile_dataset(
        regression_frame,
        experiment_id="exp-2",
        target_column="price",
        source_uri="memory://test",
        file_format="parquet",
    )
    assert report.target.inferred_problem_type is ProblemType.REGRESSION
    assert report.target.numeric is not None
    assert report.target.class_distribution is None


def test_profiling_is_deterministic(classification_frame):
    first = _profile(classification_frame)
    second = _profile(classification_frame)
    assert first.dataset == second.dataset
    assert [c.model_dump() for c in first.columns] == [c.model_dump() for c in second.columns]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1] * 50, ProblemType.BINARY_CLASSIFICATION),
        (["a", "b", "c"] * 40, ProblemType.MULTICLASS_CLASSIFICATION),
        (list(np.linspace(0, 100, 120)), ProblemType.REGRESSION),
    ],
)
def test_problem_type_inference(values, expected):
    inferred, _reason = infer_problem_type(pd.Series(values), ProfilingConfig())
    assert inferred is expected


def test_single_class_target_is_not_learnable():
    inferred, reason = infer_problem_type(pd.Series([1] * 50), ProfilingConfig())
    assert inferred is None
    assert "single distinct value" in reason


def test_empty_dataset_is_reported_not_crashed():
    report = _profile(pd.DataFrame({"churned": []}))
    assert report.dataset.row_count == 0
    assert any(w.rule == "empty_dataset" for w in report.warnings)
