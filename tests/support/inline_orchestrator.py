"""An orchestrator that runs the job entrypoints in-process, for tests.

It stands in for Step Functions: same stages, same order, same artifacts, same contracts —
executed synchronously against a local object store so the whole path can be exercised without
AWS. It is a test double and deliberately lives outside the application package.
"""

from datetime import UTC, datetime

from backend.orchestration.base import ExecutionHandle
from jobs._common.experiment_state import ArtifactExperimentStateWriter
from jobs._common.runtime import JobError
from jobs.evaluation.main import run_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.training.main import run_training, write_model_failure
from ml_engine.contracts.common import ExperimentStatus
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.experiment import ExperimentRecord, WorkflowState
from ml_engine.io import ExperimentLayout, ObjectStore, read_model, write_model
from ml_engine.reporting import final_status


class InlineOrchestrator:
    """Runs each stage immediately and writes the same ``state/workflow.json`` the ASL writes."""

    name = "inline"

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def start_eda(self, record: ExperimentRecord) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        self._mark(layout, record.experiment_id, ExperimentStatus.EDA_RUNNING, "profiling")
        try:
            run_profiling(
                self._store,
                layout,
                experiment_id=record.experiment_id,
                dataset_uri=record.dataset.uri,
                target_column=record.target_column,
                file_format=record.dataset.file_format,
            )
        except Exception as error:
            self._fail(layout, record.experiment_id, "profiling", error)
            return ExecutionHandle(execution_id="inline-eda")
        self._mark(layout, record.experiment_id, ExperimentStatus.EDA_COMPLETED, "eda_completed")
        self._mark(layout, record.experiment_id, ExperimentStatus.FEATURE_REVIEW, "feature_review")
        return ExecutionHandle(execution_id="inline-eda")

    def start_training(self, record: ExperimentRecord) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        experiment_id = record.experiment_id
        self._mark(layout, experiment_id, ExperimentStatus.PREPARING, "preparation")
        try:
            run_preparation(self._store, layout, experiment_id=experiment_id)
        except Exception as error:
            self._fail(layout, experiment_id, "preparation", error)
            return ExecutionHandle(execution_id="inline-training")

        config = read_model(self._store, layout.experiment_config, ExperimentConfig)
        self._mark(layout, experiment_id, ExperimentStatus.TRAINING, "training")
        for spec in config.enabled_models_specs():
            try:
                run_training(self._store, layout, experiment_id=experiment_id, model_name=spec.name)
            except Exception as error:
                write_model_failure(
                    self._store,
                    layout,
                    experiment_id=experiment_id,
                    model_name=spec.name,
                    code=error.code if isinstance(error, JobError) else "MODEL_TRAINING_FAILED",
                    message=str(error),
                )

        self._mark(layout, experiment_id, ExperimentStatus.EVALUATING, "evaluation")
        try:
            comparison = run_evaluation(
                self._store,
                layout,
                experiment_id=experiment_id,
                state_writer=ArtifactExperimentStateWriter(self._store, layout),
            )
        except Exception as error:
            self._fail(layout, experiment_id, "evaluation", error)
            return ExecutionHandle(execution_id="inline-training")
        assert final_status(comparison) is not None
        return ExecutionHandle(execution_id="inline-training")

    def describe(self, execution_id: str) -> dict[str, str]:
        return {"execution_id": execution_id, "orchestrator": self.name}

    def wait_for_idle(self, timeout: float | None = None) -> None:
        """Every stage already ran synchronously; kept so tests read like the AWS path."""
        return None

    # --- the state writes the ASL performs --------------------------------
    def _mark(
        self, layout: ExperimentLayout, experiment_id: str, status: ExperimentStatus, stage: str
    ) -> None:
        write_model(
            self._store,
            layout.workflow_state,
            WorkflowState(
                experiment_id=experiment_id,
                updated_at=datetime.now(UTC),
                status=status,
                current_stage=stage,
            ),
        )

    def _fail(
        self, layout: ExperimentLayout, experiment_id: str, stage: str, error: Exception
    ) -> None:
        write_model(
            self._store,
            layout.workflow_state,
            WorkflowState(
                experiment_id=experiment_id,
                updated_at=datetime.now(UTC),
                status=ExperimentStatus.FAILED,
                current_stage=stage,
                failure_code=error.code if isinstance(error, JobError) else "STAGE_FAILED",
                failure_message=str(error),
            ),
        )
