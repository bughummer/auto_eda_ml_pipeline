"""Train/validation splitting.

One registry, one abstraction. Temporal and grouped strategies slot in here without
touching the jobs, the state machine or the API.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_engine.contracts.common import (
    ProblemType,
    RecommendedAction,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.config import SplitConfig
from ml_engine.contracts.preparation import SplitSummary
from ml_engine.contracts.warnings import AnalysisWarning

MIN_ROWS_PER_FOLD = 2


class SplittingError(ValueError):
    """Raised when the dataset cannot be split at all."""


@dataclass(slots=True)
class SplitResult:
    train_index: pd.Index
    validation_index: pd.Index
    summary: SplitSummary
    warnings: list[AnalysisWarning] = field(default_factory=list)


class SplitStrategy(Protocol):
    name: str

    def split(self, frame: pd.DataFrame, target: pd.Series, config: SplitConfig) -> SplitResult: ...


def _class_distribution(target: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in target.astype("string").value_counts().items()}


def _summary(
    *,
    strategy: str,
    train_index: pd.Index,
    validation_index: pd.Index,
    config: SplitConfig,
    stratified: bool,
    target: pd.Series | None,
) -> SplitSummary:
    total = len(train_index) + len(validation_index)
    return SplitSummary(
        strategy=strategy,
        train_row_count=len(train_index),
        validation_row_count=len(validation_index),
        validation_fraction_actual=round(len(validation_index) / total, 6) if total else 0.0,
        random_seed=config.random_seed,
        stratified=stratified,
        train_class_distribution=_class_distribution(target.loc[train_index]) if stratified and target is not None else None,
        validation_class_distribution=_class_distribution(target.loc[validation_index]) if stratified and target is not None else None,
    )


def _validate_size(frame: pd.DataFrame, config: SplitConfig) -> None:
    rows = len(frame)
    if rows < 2 * MIN_ROWS_PER_FOLD:
        raise SplittingError(
            f"Only {rows} usable rows. At least {2 * MIN_ROWS_PER_FOLD} are required to build a "
            "training and a validation fold."
        )
    if int(round(rows * config.validation_fraction)) < MIN_ROWS_PER_FOLD:
        raise SplittingError(
            f"validation_fraction={config.validation_fraction} yields fewer than "
            f"{MIN_ROWS_PER_FOLD} validation rows for {rows} rows."
        )


class RandomSplit:
    """Uniform random split. Default for regression."""

    name = "random"

    def split(self, frame: pd.DataFrame, target: pd.Series, config: SplitConfig) -> SplitResult:
        _validate_size(frame, config)
        train_index, validation_index = train_test_split(
            frame.index,
            test_size=config.validation_fraction,
            random_state=config.random_seed,
            shuffle=True,
        )
        return SplitResult(
            train_index=pd.Index(train_index),
            validation_index=pd.Index(validation_index),
            summary=_summary(
                strategy=self.name,
                train_index=pd.Index(train_index),
                validation_index=pd.Index(validation_index),
                config=config,
                stratified=False,
                target=target,
            ),
        )


class StratifiedRandomSplit:
    """Class-proportional random split. Default for classification.

    Falls back to a uniform random split — with an explicit warning — when a class is too
    rare to appear in both folds. Silently changing the split would be worse.
    """

    name = "stratified_random"

    def split(self, frame: pd.DataFrame, target: pd.Series, config: SplitConfig) -> SplitResult:
        _validate_size(frame, config)
        warnings: list[AnalysisWarning] = []
        labels = target.astype("string")
        counts = labels.value_counts()
        rarest = int(counts.min()) if len(counts) else 0

        if rarest < 2:
            warnings.append(
                AnalysisWarning(
                    rule="stratification_not_possible",
                    category=WarningCategory.TARGET,
                    severity=Severity.HIGH,
                    message=(
                        f"The rarest target class has {rarest} row(s), so a stratified split is "
                        "impossible. A uniform random split was used instead; validation metrics "
                        "for rare classes will be unreliable."
                    ),
                    column=str(target.name),
                    recommended_action=RecommendedAction.REVIEW,
                    details={"rarest_class_count": rarest},
                )
            )
            result = RandomSplit().split(frame, target, config)
            result.summary.strategy = self.name
            result.warnings.extend(warnings)
            return result

        expected_validation = int(round(len(frame) * config.validation_fraction))
        if expected_validation < len(counts):
            warnings.append(
                AnalysisWarning(
                    rule="validation_fold_smaller_than_class_count",
                    category=WarningCategory.TARGET,
                    severity=Severity.MEDIUM,
                    message=(
                        f"The validation fold ({expected_validation} rows) is smaller than the "
                        f"number of classes ({len(counts)}). Per-class metrics will be noisy."
                    ),
                    column=str(target.name),
                )
            )

        train_index, validation_index = train_test_split(
            frame.index,
            test_size=config.validation_fraction,
            random_state=config.random_seed,
            shuffle=True,
            stratify=labels.to_numpy(),
        )
        return SplitResult(
            train_index=pd.Index(train_index),
            validation_index=pd.Index(validation_index),
            summary=_summary(
                strategy=self.name,
                train_index=pd.Index(train_index),
                validation_index=pd.Index(validation_index),
                config=config,
                stratified=True,
                target=target,
            ),
            warnings=warnings,
        )


STRATEGIES: dict[str, SplitStrategy] = {
    RandomSplit.name: RandomSplit(),
    StratifiedRandomSplit.name: StratifiedRandomSplit(),
}


def available_strategies() -> list[str]:
    return sorted(STRATEGIES)


def resolve_strategy(problem_type: ProblemType, config: SplitConfig) -> SplitStrategy:
    """Pick the strategy: the configured one if valid, otherwise the problem-type default."""
    if config.strategy in STRATEGIES:
        strategy = STRATEGIES[config.strategy]
        if strategy.name == StratifiedRandomSplit.name and not (
            problem_type.is_classification and config.stratify
        ):
            return STRATEGIES[RandomSplit.name]
        return strategy
    raise SplittingError(
        f"Unknown split strategy {config.strategy!r}. Available: {', '.join(available_strategies())}."
    )


def split_dataset(
    frame: pd.DataFrame,
    target: pd.Series,
    problem_type: ProblemType,
    config: SplitConfig,
) -> SplitResult:
    """Split a dataset with the strategy appropriate for the problem type."""
    strategy = resolve_strategy(problem_type, config)
    seed = config.random_seed
    np.random.seed(seed)  # noqa: NPY002 - some estimators still read the global seed
    return strategy.split(frame, target, config)
