"""Splitting must be reproducible, proportional where it can be, and honest where it cannot."""

import pandas as pd
import pytest

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.config import SplitConfig
from ml_engine.splitting import SplittingError, resolve_strategy, split_dataset


def test_classification_split_is_stratified(classification_frame):
    target = classification_frame["churned"]
    result = split_dataset(
        classification_frame, target, ProblemType.BINARY_CLASSIFICATION, SplitConfig()
    )
    train_rate = target.loc[result.train_index].mean()
    validation_rate = target.loc[result.validation_index].mean()
    assert result.summary.stratified is True
    assert abs(train_rate - validation_rate) < 0.05


def test_folds_are_disjoint_and_complete(classification_frame):
    result = split_dataset(
        classification_frame,
        classification_frame["churned"],
        ProblemType.BINARY_CLASSIFICATION,
        SplitConfig(),
    )
    assert not set(result.train_index) & set(result.validation_index)
    assert len(result.train_index) + len(result.validation_index) == len(classification_frame)


def test_split_is_reproducible_for_a_seed(classification_frame):
    config = SplitConfig(random_seed=123)
    first = split_dataset(
        classification_frame,
        classification_frame["churned"],
        ProblemType.BINARY_CLASSIFICATION,
        config,
    )
    second = split_dataset(
        classification_frame,
        classification_frame["churned"],
        ProblemType.BINARY_CLASSIFICATION,
        config,
    )
    assert list(first.validation_index) == list(second.validation_index)


def test_different_seeds_produce_different_folds(classification_frame):
    first = split_dataset(
        classification_frame,
        classification_frame["churned"],
        ProblemType.BINARY_CLASSIFICATION,
        SplitConfig(random_seed=1),
    )
    second = split_dataset(
        classification_frame,
        classification_frame["churned"],
        ProblemType.BINARY_CLASSIFICATION,
        SplitConfig(random_seed=2),
    )
    assert list(first.validation_index) != list(second.validation_index)


def test_validation_fraction_is_honoured(classification_frame):
    result = split_dataset(
        classification_frame,
        classification_frame["churned"],
        ProblemType.BINARY_CLASSIFICATION,
        SplitConfig(validation_fraction=0.35),
    )
    assert result.summary.validation_fraction_actual == pytest.approx(0.35, abs=0.02)


def test_rare_class_falls_back_to_random_with_a_warning():
    frame = pd.DataFrame({"x": range(200)})
    target = pd.Series([0] * 199 + [1], name="y")
    result = split_dataset(frame, target, ProblemType.BINARY_CLASSIFICATION, SplitConfig())
    assert result.summary.stratified is False
    assert any(w.rule == "stratification_not_possible" for w in result.warnings)


def test_regression_uses_a_random_split(regression_frame):
    strategy = resolve_strategy(ProblemType.REGRESSION, SplitConfig())
    assert strategy.name == "random"
    result = split_dataset(
        regression_frame, regression_frame["price"], ProblemType.REGRESSION, SplitConfig()
    )
    assert result.summary.stratified is False


def test_tiny_dataset_is_rejected_clearly():
    frame = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(SplittingError, match="usable rows"):
        split_dataset(frame, pd.Series([0, 1]), ProblemType.REGRESSION, SplitConfig())


def test_unknown_strategy_is_rejected():
    with pytest.raises(SplittingError, match="Unknown split strategy"):
        resolve_strategy(ProblemType.REGRESSION, SplitConfig(strategy="time_series"))
