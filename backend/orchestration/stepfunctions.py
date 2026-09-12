"""Step Functions orchestrator.

Step Functions is the authoritative workflow engine in AWS mode: it sequences the SageMaker
jobs, retries them, isolates per-model failures and writes stage transitions straight to the
experiment record. The control plane only starts executions and reads their status.
"""

import json
import logging
from typing import Any

from backend.errors import ConfigurationError, UpstreamError
from backend.orchestration.base import ExecutionHandle
from ml_engine.contracts.experiment import ExperimentRecord
from ml_engine.io import ExperimentLayout

LOGGER = logging.getLogger("ml_factory.orchestration.stepfunctions")


class StepFunctionsOrchestrator:
    name = "stepfunctions"

    def __init__(
        self,
        client: Any,
        *,
        eda_state_machine_arn: str,
        training_state_machine_arn: str,
        experiments_table: str,
    ) -> None:
        self._client = client
        self._eda_arn = eda_state_machine_arn
        self._training_arn = training_state_machine_arn
        self._table = experiments_table

    def start_eda(self, record: ExperimentRecord) -> ExecutionHandle:
        if not self._eda_arn:
            raise ConfigurationError("No EDA state machine is configured for this environment.")
        return self._start(self._eda_arn, f"eda-{record.experiment_id}", self._eda_input(record))

    def start_training(self, record: ExperimentRecord) -> ExecutionHandle:
        if not self._training_arn:
            raise ConfigurationError(
                "No training state machine is configured for this environment."
            )
        return self._start(
            self._training_arn, f"train-{record.experiment_id}", self._training_input(record)
        )

    def describe(self, execution_id: str) -> dict[str, str]:
        try:
            response = self._client.describe_execution(executionArn=execution_id)
        except Exception as exc:
            raise UpstreamError(f"Could not describe execution {execution_id}.") from exc
        return {
            "execution_id": execution_id,
            "status": str(response.get("status", "")),
            "started_at": str(response.get("startDate", "")),
            "stopped_at": str(response.get("stopDate", "")),
            "orchestrator": self.name,
        }

    # --- inputs -----------------------------------------------------------
    def _eda_input(self, record: ExperimentRecord) -> dict[str, Any]:
        layout = ExperimentLayout(base=record.artifact_prefix)
        return {
            "experiment_id": record.experiment_id,
            "dataset_s3_uri": record.dataset.uri,
            "target_column": record.target_column,
            "output_s3_prefix": record.artifact_prefix,
            "file_format": record.dataset.file_format,
            "eda_artifact_uri": layout.eda,
            "experiments_table": self._table,
        }

    def _training_input(self, record: ExperimentRecord) -> dict[str, Any]:
        layout = ExperimentLayout(base=record.artifact_prefix)
        return {
            "experiment_id": record.experiment_id,
            "output_s3_prefix": record.artifact_prefix,
            "config_uri": layout.experiment_config,
            "comparison_uri": layout.comparison,
            # The Map state iterates this list, so the catalogue can grow without an ASL change.
            "models": sorted(record.model_statuses),
            "experiments_table": self._table,
        }

    def _start(self, arn: str, name: str, payload: dict[str, Any]) -> ExecutionHandle:
        try:
            response = self._client.start_execution(
                stateMachineArn=arn, name=name[:80], input=json.dumps(payload)
            )
        except Exception as exc:
            if type(exc).__name__ == "ExecutionAlreadyExists":
                raise UpstreamError(
                    "An execution for this experiment stage was already started."
                ) from exc
            raise UpstreamError(f"Could not start the {name} workflow execution.") from exc
        LOGGER.info("Started execution %s", response.get("executionArn"))
        return ExecutionHandle(execution_id=str(response.get("executionArn", name)))
