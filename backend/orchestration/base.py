"""The orchestration boundary.

The control plane starts workflows and reads status; it never runs ML work itself. Two
implementations exist: Step Functions (AWS) and an in-process runner (local development and
tests). Both execute the identical job code.
"""

from dataclasses import dataclass
from typing import Protocol

from ml_engine.contracts.experiment import ExperimentRecord


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    """Identifies a started workflow execution."""

    execution_id: str
    started: bool = True
    detail: str | None = None


class ExperimentOrchestrator(Protocol):
    """Starts the EDA and training workflows for an experiment."""

    name: str

    def start_eda(self, record: ExperimentRecord) -> ExecutionHandle: ...

    def start_training(self, record: ExperimentRecord) -> ExecutionHandle: ...

    def describe(self, execution_id: str) -> dict[str, str]: ...
