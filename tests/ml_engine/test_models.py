"""Every registered plugin must honour the same contract."""

import pandas as pd
import pytest

from ml_engine.contracts.common import ClassWeighting, ProblemType
from ml_engine.contracts.config import PreprocessingConfig
from ml_engine.models import (
    TrainingContext,
    UnknownModelError,
    all_plugins,
    catalogue,
    default_model_names,
    get_plugin,
    plugins_for,
)
from ml_engine.preprocessing import FeaturePipeline, plan_columns
from ml_engine.training import train_model

CLASSIFICATION_FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business"]
REGRESSION_FEATURES = ["area_sqm", "rooms", "district"]


def _pipeline(frame, features, strategy):
    config = PreprocessingConfig()
    return FeaturePipeline(plan_columns(frame, features, config), config, strategy=strategy).fit(
        frame
    )


@pytest.mark.parametrize("model_name", default_model_names(ProblemType.BINARY_CLASSIFICATION))
def test_classification_plugins_train_and_report(model_name, classification_frame):
    plugin = get_plugin(model_name)
    train, validation = classification_frame.iloc[:320], classification_frame.iloc[320:]
    pipeline = _pipeline(train, CLASSIFICATION_FEATURES, plugin.preprocessing_strategy)
    metadata, estimator = train_model(
        plugin,
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        pipeline=pipeline,
        train_frame=train,
        validation_frame=validation,
        target_column="churned",
    )
    assert metadata.model_name == model_name
    assert metadata.primary_metric == "roc_auc"
    assert metadata.primary_score is not None
    assert metadata.metrics.confusion_matrix is not None
    assert metadata.feature_importance is not None
    assert metadata.library_version
    assert metadata.training_duration_seconds >= 0
    assert estimator is not None


@pytest.mark.parametrize("model_name", default_model_names(ProblemType.REGRESSION))
def test_regression_plugins_train_and_report(model_name, regression_frame):
    plugin = get_plugin(model_name)
    train, validation = regression_frame.iloc[:240], regression_frame.iloc[240:]
    pipeline = _pipeline(train, REGRESSION_FEATURES, plugin.preprocessing_strategy)
    metadata, _estimator = train_model(
        plugin,
        experiment_id="exp-1",
        problem_type=ProblemType.REGRESSION,
        pipeline=pipeline,
        train_frame=train,
        validation_frame=validation,
        target_column="price",
    )
    assert metadata.primary_metric == "rmse"
    assert metadata.primary_score is not None and metadata.primary_score > 0
    assert set(metadata.metrics.values) >= {"mae", "rmse", "r2"}


def test_feature_importance_is_normalized(classification_frame):
    plugin = get_plugin("random_forest")
    pipeline = _pipeline(classification_frame, CLASSIFICATION_FEATURES, "dense_numeric")
    metadata, _ = train_model(
        plugin,
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        pipeline=pipeline,
        train_frame=classification_frame.iloc[:320],
        validation_frame=classification_frame.iloc[320:],
        target_column="churned",
    )
    entries = metadata.feature_importance.entries
    assert sum(entry.importance for entry in entries) == pytest.approx(1.0, abs=1e-6)
    assert [entry.rank for entry in entries] == list(range(1, len(entries) + 1))


def test_class_weighting_is_applied_where_supported():
    context = TrainingContext(
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        class_labels=["0", "1"],
        class_counts={"0": 900, "1": 100},
        class_weighting=ClassWeighting.AUTO,
    )
    assert get_plugin("logistic_regression").default_params(context)["class_weight"] == "balanced"
    assert (
        get_plugin("random_forest").default_params(context)["class_weight"] == "balanced_subsample"
    )
    assert get_plugin("xgboost").default_params(context)["scale_pos_weight"] == pytest.approx(9.0)
    assert get_plugin("catboost").default_params(context)["auto_class_weights"] == "Balanced"


def test_class_weighting_can_be_turned_off():
    context = TrainingContext(
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        class_labels=["0", "1"],
        class_counts={"0": 900, "1": 100},
        class_weighting=ClassWeighting.NONE,
    )
    assert "class_weight" not in get_plugin("logistic_regression").default_params(context)


def test_plugins_declare_their_problem_types():
    for plugin in all_plugins():
        assert plugin.supported_problem_types
        for problem_type in plugin.supported_problem_types:
            assert plugin in plugins_for(problem_type, only_available=False)


def test_catalogue_reports_availability():
    descriptors = {d.name: d for d in catalogue()}
    assert descriptors["logistic_regression"].available is True
    for descriptor in descriptors.values():
        assert descriptor.display_name and descriptor.description
        if not descriptor.available:
            assert descriptor.unavailable_reason


def test_unknown_model_is_rejected():
    with pytest.raises(UnknownModelError):
        get_plugin("magic_model")


def test_plugin_rejects_unsupported_problem_type(classification_frame):
    from ml_engine.training import TrainingError

    pipeline = _pipeline(classification_frame, CLASSIFICATION_FEATURES, "dense_numeric")
    with pytest.raises(TrainingError, match="does not support"):
        train_model(
            get_plugin("elastic_net"),
            experiment_id="exp-1",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            pipeline=pipeline,
            train_frame=classification_frame,
            validation_frame=classification_frame,
            target_column="churned",
        )


def test_training_requires_a_fitted_pipeline(classification_frame):
    from ml_engine.training import TrainingError

    config = PreprocessingConfig()
    unfitted = FeaturePipeline(
        plan_columns(classification_frame, CLASSIFICATION_FEATURES, config), config
    )
    with pytest.raises(TrainingError, match="must be fitted"):
        train_model(
            get_plugin("logistic_regression"),
            experiment_id="exp-1",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            pipeline=unfitted,
            train_frame=classification_frame,
            validation_frame=classification_frame,
            target_column="churned",
        )


def test_string_labels_survive_every_plugin(classification_frame):
    """Labels stay human-readable end to end, whatever the library wants internally."""
    frame = classification_frame.copy()
    frame["churned"] = frame["churned"].map({0: "retained", 1: "churned"})
    for model_name in default_model_names(ProblemType.BINARY_CLASSIFICATION):
        plugin = get_plugin(model_name)
        pipeline = _pipeline(
            frame.iloc[:320], CLASSIFICATION_FEATURES, plugin.preprocessing_strategy
        )
        metadata, _ = train_model(
            plugin,
            experiment_id="exp-1",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            pipeline=pipeline,
            train_frame=frame.iloc[:320],
            validation_frame=frame.iloc[320:],
            target_column="churned",
        )
        assert set(metadata.metrics.confusion_matrix.labels) == {"retained", "churned"}


def test_all_registered_plugins_train(classification_frame, regression_frame):
    """The catalogue guard: adding a plugin without exercising it here is a gap."""
    trained = set()
    for plugin in all_plugins():
        if not plugin.is_available()[0]:
            continue
        if ProblemType.REGRESSION in plugin.supported_problem_types:
            frame, features, target, problem_type = (
                regression_frame,
                REGRESSION_FEATURES,
                "price",
                ProblemType.REGRESSION,
            )
        else:
            frame, features, target, problem_type = (
                classification_frame,
                CLASSIFICATION_FEATURES,
                "churned",
                ProblemType.BINARY_CLASSIFICATION,
            )
        split = int(len(frame) * 0.8)
        pipeline = _pipeline(frame.iloc[:split], features, plugin.preprocessing_strategy)
        metadata, _ = train_model(
            plugin,
            experiment_id="exp-1",
            problem_type=problem_type,
            pipeline=pipeline,
            train_frame=frame.iloc[:split],
            validation_frame=frame.iloc[split:],
            target_column=target,
        )
        assert isinstance(metadata.hyperparameters, dict)
        trained.add(plugin.name)
    assert trained == {p.name for p in all_plugins() if p.is_available()[0]}


def test_predictions_are_aligned_with_the_validation_frame(classification_frame):
    plugin = get_plugin("logistic_regression")
    train, validation = classification_frame.iloc[:320], classification_frame.iloc[320:]
    pipeline = _pipeline(train, CLASSIFICATION_FEATURES, "dense_numeric")
    metadata, estimator = train_model(
        plugin,
        experiment_id="exp-1",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        pipeline=pipeline,
        train_frame=train,
        validation_frame=validation,
        target_column="churned",
    )
    predictions = plugin.predict(estimator, pipeline.transform(validation))
    assert len(predictions) == len(validation)
    assert metadata.validation_row_count == len(validation)
    assert isinstance(pipeline.transform(validation), pd.DataFrame)
