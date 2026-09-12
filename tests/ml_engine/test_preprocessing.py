"""Preprocessing must be complete, reproducible, and free of validation leakage."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.contracts.config import PreprocessingConfig
from ml_engine.preprocessing import FeaturePipeline, PreprocessingError, plan_columns


@pytest.fixture
def plan(classification_frame, classification_eda):
    features = [c for c in classification_frame.columns if c != "churned"]
    return plan_columns(classification_frame, features, PreprocessingConfig(), classification_eda)


def test_columns_are_assigned_to_the_right_transformer(plan):
    assert "tenure_months" in plan.numeric
    assert "contract" in plan.categorical
    assert "is_business" in plan.boolean
    assert "signup_date" in plan.datetime


def test_text_and_constant_columns_are_dropped_with_reasons(plan):
    assert "notes" in plan.dropped
    assert "region_code" in plan.dropped
    assert all(reason for reason in plan.dropped.values())


def test_missing_selected_feature_is_reported(classification_frame):
    plan = plan_columns(classification_frame, ["ghost"], PreprocessingConfig())
    assert plan.dropped["ghost"] == "column is not present in the dataset"
    assert any(w.rule == "selected_feature_missing" for w in plan.warnings)


def test_dense_pipeline_produces_a_finite_numeric_matrix(classification_frame, plan):
    pipeline = FeaturePipeline(plan, PreprocessingConfig())
    transformed = pipeline.fit_transform(classification_frame)
    assert transformed.shape[0] == len(classification_frame)
    assert np.isfinite(transformed.to_numpy(dtype="float64")).all()
    assert len(pipeline.feature_names) == transformed.shape[1]


def test_datetime_columns_become_controlled_derived_features(classification_frame, plan):
    pipeline = FeaturePipeline(plan, PreprocessingConfig()).fit(classification_frame)
    derived = [name for name in pipeline.feature_names if name.startswith("signup_date__")]
    assert derived == ["signup_date__year", "signup_date__month", "signup_date__day_of_week"]


def test_transform_never_refits_on_validation_data(classification_frame, plan):
    """The imputed value must come from the training fold, not from the data being transformed."""
    train = classification_frame.iloc[:200]
    validation = classification_frame.iloc[200:].copy()
    pipeline = FeaturePipeline(plan, PreprocessingConfig()).fit(train)

    train_median = train["monthly_charges"].median()
    validation["monthly_charges"] = np.nan
    transformed = pipeline.transform(validation)

    imputer = pipeline._transformer.named_transformers_["numeric"].named_steps["impute"]
    learned = imputer.statistics_[pipeline.plan.numeric.index("monthly_charges")]
    assert learned == pytest.approx(train_median)
    assert np.isfinite(transformed.to_numpy(dtype="float64")).all()


def test_unseen_categories_do_not_break_transform(classification_frame, plan):
    train = classification_frame.iloc[:200]
    validation = classification_frame.iloc[200:].copy()
    validation.loc[validation.index[:5], "contract"] = "brand-new-category"
    pipeline = FeaturePipeline(plan, PreprocessingConfig()).fit(train)
    assert pipeline.transform(validation).shape[1] == len(pipeline.feature_names)


def test_native_categorical_strategy_keeps_categories_as_strings(classification_frame, plan):
    pipeline = FeaturePipeline(plan, PreprocessingConfig(), strategy="native_categorical")
    transformed = pipeline.fit_transform(classification_frame)
    assert "contract" in transformed.columns
    assert transformed["contract"].dtype == object
    assert pipeline.categorical_feature_names == plan.categorical


def test_metadata_describes_what_happened(classification_frame, plan):
    pipeline = FeaturePipeline(plan, PreprocessingConfig()).fit(classification_frame)
    metadata = pipeline.metadata()
    assert metadata.fitted_on == "train"
    assert metadata.output_feature_count == len(pipeline.feature_names)
    assert metadata.dropped_columns == plan.dropped
    assert metadata.config_digest


def test_identical_configuration_yields_an_identical_digest(classification_frame, plan):
    first = FeaturePipeline(plan, PreprocessingConfig()).fit(classification_frame)
    second = FeaturePipeline(plan, PreprocessingConfig()).fit(classification_frame)
    assert first.config_digest() == second.config_digest()


def test_pipeline_refuses_to_transform_before_fitting(plan):
    with pytest.raises(PreprocessingError):
        FeaturePipeline(plan, PreprocessingConfig()).transform(pd.DataFrame())


def test_empty_plan_is_rejected_with_a_useful_message(classification_frame):
    plan = plan_columns(classification_frame, ["region_code"], PreprocessingConfig())
    with pytest.raises(PreprocessingError, match="No usable feature"):
        FeaturePipeline(plan, PreprocessingConfig())
