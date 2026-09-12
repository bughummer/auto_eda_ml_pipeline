"""Metric contracts. Values are always produced by scikit-learn, never by an LLM."""

from pydantic import Field

from ml_engine.contracts.common import MetricDirection, StrictModel
from ml_engine.contracts.warnings import AnalysisWarning


class ConfusionMatrix(StrictModel):
    labels: list[str]
    matrix: list[list[int]] = Field(description="Row = true label, column = predicted label.")


class MetricSet(StrictModel):
    """Metrics for one model on one fold.

    ``values`` holds every successfully computed metric. A metric that could not be computed
    is absent and explained in ``warnings`` — a metric failure never fails an experiment.
    """

    values: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: ConfusionMatrix | None = None
    warnings: list[AnalysisWarning] = Field(default_factory=list)

    def get(self, name: str) -> float | None:
        return self.values.get(name)


class MetricDefinition(StrictModel):
    name: str
    display_name: str
    direction: MetricDirection
    description: str
