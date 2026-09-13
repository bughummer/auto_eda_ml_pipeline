"""Derived-feature proposals: generate, read, approve.

Three responsibilities, kept apart on purpose. A model *suggests*; validation decides what may
be offered; a person decides what is used. The service never promotes a suggestion on its own,
and the approved set is what preparation later computes — nothing else.
"""

import logging
from datetime import UTC, datetime

from backend.config import Settings
from backend.errors import (
    ArtifactNotReadyError,
    ConflictError,
    FeatureDisabledError,
    UpstreamError,
    ValidationError,
)
from backend.repositories import ExperimentRepository
from backend.schemas.proposals import ApproveProposalsRequest, ProposeFeaturesRequest
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.proposals import FeatureProposalReport
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists, write_model
from ml_engine.reasoning import (
    ReasoningClient,
    ReasoningOutputError,
    ReasoningUnavailableError,
    propose_features,
)

LOGGER = logging.getLogger("ml_factory.services.proposals")


class FeatureProposalService:
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

    def generate(
        self, experiment_id: str, request: ProposeFeaturesRequest
    ) -> FeatureProposalReport:
        """Ask the model for specifications. Replaces any previous, undecided proposal set."""
        if not self.enabled:
            raise FeatureDisabledError(
                "Feature proposals need Bedrock. Enable it and configure a model id to use them."
            )
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)

        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        if eda is None:
            raise ConflictError(
                "Feature proposals require the deterministic EDA to have completed first."
            )

        try:
            report = propose_features(
                self._client,
                experiment_id=experiment_id,
                eda=eda,
                target_column=record.target_column,
                leakage=read_model_if_exists(self._store, layout.leakage, LeakageReport),
                dictionary=read_model_if_exists(
                    self._store, layout.data_dictionary, DataDictionary
                ),
                target_definition=request.target_definition,
                prediction_timing=request.prediction_timing,
            )
        except (ReasoningUnavailableError, ReasoningOutputError) as error:
            raise UpstreamError(str(error)) from error

        write_model(self._store, layout.feature_proposals, report)
        LOGGER.info(
            "Proposed %d derived feature(s) for %s, refused %d",
            len(report.candidates),
            experiment_id,
            len(report.rejected),
        )
        return report

    def get(self, experiment_id: str, root: str | None = None) -> FeatureProposalReport:
        report = self.stored(experiment_id, root)
        if report is None:
            raise ArtifactNotReadyError("No derived features have been proposed yet.")
        return report

    def stored(self, experiment_id: str, root: str | None = None) -> FeatureProposalReport | None:
        """The report if it exists. Used by training start, where absence is not an error."""
        record = self._repository.get(experiment_id, root)
        layout = ExperimentLayout(base=record.artifact_prefix)
        return read_model_if_exists(self._store, layout.feature_proposals, FeatureProposalReport)

    def approve(
        self, experiment_id: str, request: ApproveProposalsRequest, user: str | None = None
    ) -> FeatureProposalReport:
        """Record exactly which candidates a person approved. Approving none is a valid answer."""
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        report = read_model_if_exists(self._store, layout.feature_proposals, FeatureProposalReport)
        if report is None:
            raise ArtifactNotReadyError("No derived features have been proposed yet.")

        offered = {candidate.name for candidate in report.candidates}
        unknown = [name for name in request.approved_names if name not in offered]
        if unknown:
            raise ValidationError(
                "Only proposals that were offered for this experiment can be approved.",
                {"unknown_proposals": unknown},
            )

        report.approved_names = list(dict.fromkeys(request.approved_names))
        report.decided_at = datetime.now(UTC)
        report.decided_by = user
        write_model(self._store, layout.feature_proposals, report)
        LOGGER.info(
            "Approved %d of %d proposed features for %s",
            len(report.approved_names),
            len(report.candidates),
            experiment_id,
        )
        return report
