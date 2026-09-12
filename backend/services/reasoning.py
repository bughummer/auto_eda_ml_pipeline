"""Semantic reasoning use cases.

Runs only after deterministic artifacts exist, and stores a validated, read-only report.
Nothing here can change an experiment, a feature selection or a metric.
"""

import logging

from backend.config import Settings
from backend.errors import (
    ArtifactNotReadyError,
    ConflictError,
    FeatureDisabledError,
    UpstreamError,
)
from backend.repositories import ExperimentRepository
from backend.schemas.reasoning import ReasoningRequest
from ml_engine.contracts.comparison import ExperimentSummary
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.reasoning import ReasoningReport
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists, write_model
from ml_engine.io.layout import (
    DATA_DICTIONARY_FILE,
    EDA_FILE,
    LEAKAGE_FILE,
    SUMMARY_FILE,
)
from ml_engine.reasoning import (
    ReasoningClient,
    ReasoningOutputError,
    ReasoningUnavailableError,
    run_reasoning,
)

LOGGER = logging.getLogger("ml_factory.services.reasoning")


class ReasoningService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        client: ReasoningClient | None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._settings.bedrock_enabled and self._client is not None

    def run(self, experiment_id: str, request: ReasoningRequest) -> ReasoningReport:
        if not self.enabled:
            raise FeatureDisabledError(
                "Semantic reasoning is disabled in this environment. Enable Bedrock and "
                "configure a model id to use it."
            )
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)

        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        if eda is None:
            raise ConflictError(
                "Semantic reasoning requires the deterministic EDA to have completed first."
            )
        leakage = read_model_if_exists(self._store, layout.leakage, LeakageReport)
        dictionary = read_model_if_exists(self._store, layout.data_dictionary, DataDictionary)
        summary = read_model_if_exists(self._store, layout.summary, ExperimentSummary)

        try:
            report = run_reasoning(
                self._client,
                experiment_id=experiment_id,
                eda=eda,
                leakage=leakage,
                dictionary=dictionary,
                summary=summary,
                target_definition=request.target_definition,
                prediction_timing=request.prediction_timing,
                question=request.question,
            )
        except ReasoningUnavailableError as error:
            raise UpstreamError(str(error)) from error
        except ReasoningOutputError as error:
            raise UpstreamError(str(error)) from error

        report.input_artifacts = _artifact_names(eda, leakage, dictionary, summary)
        write_model(self._store, layout.reasoning, report)
        LOGGER.info(
            "Stored reasoning report for %s (%d semantic leakage findings)",
            experiment_id,
            len(report.semantic_leakage),
        )
        return report

    def get(self, experiment_id: str) -> ReasoningReport:
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        report = read_model_if_exists(self._store, layout.reasoning, ReasoningReport)
        if report is None:
            raise ArtifactNotReadyError("No semantic analysis has been generated yet.")
        return report


def _artifact_names(eda, leakage, dictionary, summary) -> list[str]:
    """Which artifacts fed the analysis, recorded on the report for auditability."""
    names = [EDA_FILE] if eda is not None else []
    if leakage is not None:
        names.append(LEAKAGE_FILE)
    if dictionary is not None:
        names.append(DATA_DICTIONARY_FILE)
    if summary is not None:
        names.append(SUMMARY_FILE)
    return names
