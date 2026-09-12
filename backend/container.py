"""Dependency wiring.

One place decides which adapters are in play. Swapping local for AWS changes this file's
output, nothing else.
"""

import logging
from dataclasses import dataclass

from ml_engine.io import LocalObjectStore, ObjectStore, S3ObjectStore

from backend.config import Settings, get_settings
from backend.orchestration import ExperimentOrchestrator, LocalOrchestrator, StepFunctionsOrchestrator
from backend.repositories import (
    DynamoExperimentRepository,
    ExperimentRepository,
    InMemoryExperimentRepository,
)
from backend.services.experiments import ExperimentService

LOGGER = logging.getLogger("ml_factory.container")


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    store: ObjectStore
    repository: ExperimentRepository
    orchestrator: ExperimentOrchestrator
    experiments: ExperimentService

    def shutdown(self) -> None:
        shutdown = getattr(self.orchestrator, "shutdown", None)
        if callable(shutdown):
            shutdown()


def build_container(settings: Settings | None = None) -> AppContainer:
    settings = settings or get_settings()
    problems = settings.validate_for_mode()
    if problems:
        for problem in problems:
            LOGGER.error("Configuration problem: %s", problem)

    if settings.is_local:
        settings.local_root.mkdir(parents=True, exist_ok=True)
        store: ObjectStore = LocalObjectStore()
        repository: ExperimentRepository = InMemoryExperimentRepository()
        orchestrator: ExperimentOrchestrator = LocalOrchestrator(
            repository, store, max_workers=settings.local_max_workers
        )
    else:
        from backend.aws import build_client, build_dynamodb_table

        store = S3ObjectStore(build_client("s3", settings.aws_region))
        repository = DynamoExperimentRepository(
            build_dynamodb_table(settings.experiments_table, settings.aws_region)
        )
        orchestrator = StepFunctionsOrchestrator(
            build_client("stepfunctions", settings.aws_region),
            eda_state_machine_arn=settings.eda_state_machine_arn,
            training_state_machine_arn=settings.training_state_machine_arn,
            experiments_table=settings.experiments_table,
        )

    service = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    LOGGER.info("ML Factory container built in %s mode", settings.mode.value)
    return AppContainer(
        settings=settings,
        store=store,
        repository=repository,
        orchestrator=orchestrator,
        experiments=service,
    )
