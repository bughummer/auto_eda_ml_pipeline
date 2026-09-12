"""In-process orchestrator for local development and tests.

It runs the *same* job entrypoints as AWS, in the same order, writing the same artifacts to
the same layout — only the compute and the storage backend differ. That is what makes the
local path a faithful rehearsal of the AWS path rather than a separate implementation.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from ml_engine.contracts.common import ExperimentStatus, ModelRunStatus
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.experiment import ExperimentRecord
from ml_engine.io import ExperimentLayout, ObjectStore, read_model
from ml_engine.reporting import final_status

from backend.orchestration.base import ExecutionHandle
from backend.repositories import ExperimentRepository
from jobs._common.runtime import JobError
from jobs.evaluation.main import run_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.training.main import run_training, write_model_failure

LOGGER = logging.getLogger("ml_factory.orchestration.local")


class LocalOrchestrator:
    """Executes experiment stages in a background thread pool."""

    name = "local"

    def __init__(
        self,
        repository: ExperimentRepository,
        store: ObjectStore,
        *,
        max_workers: int = 2,
    ) -> None:
        self._repository = repository
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mlf-local")
        self._futures: list[Future] = []
        self._lock = threading.Lock()

    # --- workflow entry points -------------------------------------------
    def start_eda(self, record: ExperimentRecord) -> ExecutionHandle:
        return self._submit(f"local-eda-{record.experiment_id}", self._run_eda, record.experiment_id)

    def start_training(self, record: ExperimentRecord) -> ExecutionHandle:
        return self._submit(
            f"local-training-{record.experiment_id}", self._run_training, record.experiment_id
        )

    def describe(self, execution_id: str) -> dict[str, str]:
        return {"execution_id": execution_id, "orchestrator": self.name}

    def wait_for_idle(self, timeout: float | None = None) -> None:
        """Block until every submitted stage has finished. Used by tests and the demo script."""
        with self._lock:
            pending = list(self._futures)
        for future in pending:
            future.result(timeout=timeout)
        with self._lock:
            if any(not f.done() for f in self._futures):
                self.wait_for_idle(timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # --- stages -----------------------------------------------------------
    def _run_eda(self, experiment_id: str) -> None:
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        self._set(experiment_id, ExperimentStatus.EDA_RUNNING, "profiling")
        try:
            run_profiling(
                self._store,
                layout,
                experiment_id=experiment_id,
                dataset_uri=record.dataset.uri,
                target_column=record.target_column,
                file_format=record.dataset.file_format,
            )
        except Exception as error:  # noqa: BLE001 - stage boundary
            self._fail(experiment_id, error, stage="profiling")
            return
        self._set(experiment_id, ExperimentStatus.EDA_COMPLETED, "eda_completed")
        self._set(experiment_id, ExperimentStatus.FEATURE_REVIEW, "feature_review")

    def _run_training(self, experiment_id: str) -> None:
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        self._set(experiment_id, ExperimentStatus.PREPARING, "preparation")
        try:
            run_preparation(self._store, layout, experiment_id=experiment_id)
        except Exception as error:  # noqa: BLE001 - stage boundary
            self._fail(experiment_id, error, stage="preparation")
            return

        config = read_model(self._store, layout.experiment_config, ExperimentConfig)
        model_names = [spec.name for spec in config.enabled_models_specs()]
        statuses = dict.fromkeys(model_names, ModelRunStatus.QUEUED.value)
        self._set(experiment_id, ExperimentStatus.TRAINING, "training", model_statuses=statuses)

        for name in model_names:
            statuses[name] = ModelRunStatus.RUNNING.value
            self._repository.update(experiment_id, model_statuses=dict(statuses))
            try:
                run_training(self._store, layout, experiment_id=experiment_id, model_name=name)
                statuses[name] = ModelRunStatus.COMPLETED.value
            except Exception as error:  # noqa: BLE001 - one model must not fail the experiment
                LOGGER.warning("Model %s failed: %s", name, error)
                write_model_failure(
                    self._store,
                    layout,
                    experiment_id=experiment_id,
                    model_name=name,
                    code=error.code if isinstance(error, JobError) else "MODEL_TRAINING_FAILED",
                    message=str(error),
                )
                statuses[name] = ModelRunStatus.FAILED.value
            self._repository.update(experiment_id, model_statuses=dict(statuses))

        self._set(experiment_id, ExperimentStatus.EVALUATING, "evaluation")
        try:
            comparison = run_evaluation(self._store, layout, experiment_id=experiment_id)
        except Exception as error:  # noqa: BLE001 - stage boundary
            self._fail(experiment_id, error, stage="evaluation")
            return

        self._repository.update(
            experiment_id,
            status=final_status(comparison),
            current_stage="completed",
            best_model=comparison.best_model,
            best_score=comparison.best_score,
            primary_metric=comparison.primary_metric,
        )

    # --- helpers ----------------------------------------------------------
    def _submit(self, execution_id: str, target, *args) -> ExecutionHandle:
        future = self._executor.submit(target, *args)
        with self._lock:
            self._futures = [f for f in self._futures if not f.done()]
            self._futures.append(future)
        return ExecutionHandle(execution_id=execution_id)

    def _set(self, experiment_id: str, status: ExperimentStatus, stage: str, **extra) -> None:
        self._repository.update(experiment_id, status=status, current_stage=stage, **extra)

    def _fail(self, experiment_id: str, error: Exception, *, stage: str) -> None:
        code = error.code if isinstance(error, JobError) else "STAGE_FAILED"
        LOGGER.exception("Experiment %s failed during %s", experiment_id, stage)
        self._repository.update(
            experiment_id,
            status=ExperimentStatus.FAILED,
            current_stage=stage,
            failure_code=code,
            failure_message=str(error),
        )
