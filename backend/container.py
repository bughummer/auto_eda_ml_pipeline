"""Dependency wiring.

One place decides which adapters are in play: S3 for artifacts and experiment records, Step
Functions for orchestration, Bedrock for the optional reasoning layer. Everything heavier
than a validation runs in AWS.
"""

import logging
from dataclasses import dataclass

from backend.config import Settings, get_settings
from backend.orchestration import ExperimentOrchestrator, StepFunctionsOrchestrator
from backend.repositories import ExperimentRepository, ObjectStoreExperimentRepository
from backend.services.dictionary import DataDictionaryService
from backend.services.experiments import ExperimentService
from backend.services.proposals import FeatureProposalService
from backend.services.reasoning import ReasoningService
from ml_engine.io import ObjectStore, S3ObjectStore

LOGGER = logging.getLogger("ml_factory.container")


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    store: ObjectStore
    repository: ExperimentRepository
    orchestrator: ExperimentOrchestrator
    experiments: ExperimentService
    dictionary: DataDictionaryService
    reasoning: ReasoningService
    proposals: FeatureProposalService

    def shutdown(self) -> None:
        shutdown = getattr(self.orchestrator, "shutdown", None)
        if callable(shutdown):
            shutdown()


def build_container(
    settings: Settings | None = None,
    *,
    store: ObjectStore | None = None,
    repository: ExperimentRepository | None = None,
    orchestrator: ExperimentOrchestrator | None = None,
) -> AppContainer:
    """Build the application graph.

    The adapters are injectable so tests can substitute doubles without a second wiring path
    existing in the product.
    """
    settings = settings or get_settings()
    for problem in settings.validate_configuration():
        LOGGER.error("Configuration problem: %s", problem)

    if store is None or repository is None or orchestrator is None:
        from backend.aws import build_client

    if store is None:
        store = S3ObjectStore(build_client("s3", settings))
    if repository is None:
        repository = ObjectStoreExperimentRepository(store, settings.artifact_root)
    if orchestrator is None:
        orchestrator = StepFunctionsOrchestrator(
            build_client("stepfunctions", settings),
            eda_state_machine_arn=settings.eda_state_machine_arn,
            training_state_machine_arn=settings.training_state_machine_arn,
        )

    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    dictionary = DataDictionaryService(settings=settings, repository=repository, store=store)
    reasoning_client = _build_reasoning_client(settings)
    reasoning = ReasoningService(
        settings=settings, repository=repository, store=store, client=reasoning_client
    )
    proposals = FeatureProposalService(
        settings=settings, repository=repository, store=store, client=reasoning_client
    )
    LOGGER.info("ML Factory container built (artifact root %s)", settings.artifact_root)
    return AppContainer(
        settings=settings,
        store=store,
        repository=repository,
        orchestrator=orchestrator,
        experiments=experiments,
        dictionary=dictionary,
        reasoning=reasoning,
        proposals=proposals,
    )


def _build_reasoning_client(settings: Settings):
    """Bedrock is optional: without it the platform simply has no reasoning layer."""
    if not settings.bedrock_enabled or not settings.bedrock_model_id:
        return None
    from backend.aws import build_client
    from ml_engine.reasoning import BedrockReasoningClient

    return BedrockReasoningClient(
        build_client("bedrock-runtime", settings),
        model_id=settings.bedrock_model_id,
        max_tokens=settings.bedrock_max_tokens,
    )
