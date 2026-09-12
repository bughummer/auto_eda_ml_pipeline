"""Experiment use cases.

The control plane's entire job: validate, record, start workflows, read artifacts. It never
loads a dataset, fits a transform or trains a model.
"""

# ``list`` is a use case here as well as a builtin, so annotations must stay lazy.
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from backend.config import Settings
from backend.errors import (
    ArtifactNotReadyError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from backend.orchestration.base import ExperimentOrchestrator
from backend.repositories import ExperimentRepository
from backend.schemas.experiments import (
    CreateExperimentRequest,
    FeatureReviewItem,
    FeatureReviewResponse,
    ModelRunState,
    TrainingConfigRequest,
    TrainingConfigResponse,
    TrainingStatusResponse,
    UpdateFeatureSelectionRequest,
)
from ml_engine.contracts.common import (
    ACTION_ORDER,
    ClassWeighting,
    ExperimentStatus,
    LeakageRiskLevel,
    ModelRunStatus,
    ProblemType,
    RecommendedAction,
    RequestedProblemType,
)
from ml_engine.contracts.comparison import ComparisonReport, ExperimentSummary
from ml_engine.contracts.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_FRACTION,
    ComputeConfig,
    DatasetReference,
    EnvironmentCapture,
    ExperimentConfig,
    FeatureSelection,
    ModelSpec,
    PreprocessingConfig,
)
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.experiment import ExperimentDefinition, ExperimentRecord
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.evaluation import resolve_primary_metric, selectable_primary_metrics
from ml_engine.evaluation.metrics import UnknownMetricError
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    detect_format,
    is_s3_uri,
    parse_s3_uri,
    read_model_if_exists,
    write_model,
)
from ml_engine.io.uri import InvalidS3UriError
from ml_engine.leakage import default_feature_selection
from ml_engine.models import UnknownModelError, default_model_names, get_plugin, plugins_for
from ml_engine.training.runner import environment_metadata, package_versions

LOGGER = logging.getLogger("ml_factory.services.experiments")

ML_FACTORY_VERSION = "1.0.0"

# Stages during which a training run must not be started again.
_ACTIVE_TRAINING_STATUSES = frozenset(
    {
        ExperimentStatus.PREPARING,
        ExperimentStatus.TRAINING,
        ExperimentStatus.EVALUATING,
    }
)


class ExperimentService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        orchestrator: ExperimentOrchestrator,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._orchestrator = orchestrator

    # --- lifecycle --------------------------------------------------------
    def create(self, request: CreateExperimentRequest, user: str | None = None) -> ExperimentRecord:
        """Validate, record and start EDA. Returns immediately; AWS work runs asynchronously."""
        dataset = self._validate_dataset(request.dataset_uri, request.file_format)
        experiment_id = _new_experiment_id()
        layout = ExperimentLayout.for_experiment(self._settings.artifact_root, experiment_id)

        record = self._repository.create(
            ExperimentDefinition(
                experiment_id=experiment_id,
                name=request.name,
                created_by=user,
                created_at=datetime.now(UTC),
                dataset=dataset,
                target_column=request.target_column,
                requested_problem_type=request.problem_type,
                artifact_prefix=layout.base,
            )
        )
        handle = self._orchestrator.start_eda(record)
        LOGGER.info("Started EDA for %s (%s)", experiment_id, handle.execution_id)
        # Only the execution reference is ours to record: the workflow writes every stage
        # transition itself, so the control plane cannot overwrite a stage it has passed.
        return self._repository.update(experiment_id, eda_execution_arn=handle.execution_id)

    def get(self, experiment_id: str, root: str | None = None) -> ExperimentRecord:
        return self._repository.get(experiment_id, root=self._checked_root(root))

    def list(
        self,
        limit: int | None = None,
        created_by: str | None = None,
        root: str | None = None,
    ) -> list[ExperimentRecord]:
        """List experiments found under an artifact root.

        With no root this is the platform's own bucket. With one, it is whatever previous
        experiments exist under that approved bucket — which is how the UI shows results from
        an environment other than the one it is running in.
        """
        return self._repository.list(
            limit=min(
                limit or self._settings.experiment_list_limit, self._settings.experiment_list_limit
            ),
            created_by=created_by,
            root=self._checked_root(root),
        )

    def artifact_roots(self) -> list[str]:
        """Artifact roots the UI may offer, the writable one first."""
        return self._settings.artifact_roots

    def _checked_root(self, root: str | None) -> str | None:
        """Only configured artifact roots may be read — a URI from the browser is not a key."""
        if root is None:
            return None
        if not self._settings.artifact_root_allowed(root):
            raise ValidationError(
                "That artifact location is not configured for this platform.",
                {"root": root, "allowed_roots": self._settings.artifact_roots},
            )
        return root

    def delete(self, experiment_id: str) -> None:
        """Soft delete: the record disappears from the UI, the artifacts stay auditable."""
        self._repository.get(experiment_id)
        self._repository.delete(experiment_id)

    # --- artifacts --------------------------------------------------------
    # Every read accepts an artifact root, so the UI can open an experiment produced by
    # another environment as easily as one of its own.
    def eda(self, experiment_id: str, root: str | None = None) -> EdaReport:
        return self._artifact(experiment_id, lambda layout: layout.eda, EdaReport, "EDA", root)

    def leakage(self, experiment_id: str, root: str | None = None) -> LeakageReport:
        return self._artifact(
            experiment_id, lambda layout: layout.leakage, LeakageReport, "leakage analysis", root
        )

    def comparison(self, experiment_id: str, root: str | None = None) -> ComparisonReport:
        return self._artifact(
            experiment_id,
            lambda layout: layout.comparison,
            ComparisonReport,
            "model comparison",
            root,
        )

    def summary(self, experiment_id: str, root: str | None = None) -> ExperimentSummary:
        return self._artifact(
            experiment_id,
            lambda layout: layout.summary,
            ExperimentSummary,
            "experiment summary",
            root,
        )

    def model_metadata(
        self, experiment_id: str, model_name: str, root: str | None = None
    ) -> ModelMetadata:
        layout = ExperimentLayout(base=self.get(experiment_id, root).artifact_prefix)
        metadata = read_model_if_exists(
            self._store, layout.model_metadata(model_name), ModelMetadata
        )
        if metadata is not None:
            return metadata
        failure = read_model_if_exists(self._store, layout.model_failure(model_name), ModelFailure)
        if failure is not None:
            raise ConflictError(
                f"Model '{model_name}' failed: {failure.message}",
                {"error_code": failure.error_code, "model": model_name},
            )
        raise NotFoundError(f"No result exists for model '{model_name}' in this experiment.")

    # --- feature review ---------------------------------------------------
    def feature_review(self, experiment_id: str, root: str | None = None) -> FeatureReviewResponse:
        """Join the EDA profile, the leakage verdict and any documentation into review rows."""
        record = self.get(experiment_id, root)
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = self.eda(experiment_id, root)
        leakage = read_model_if_exists(self._store, layout.leakage, LeakageReport) or LeakageReport(
            experiment_id=experiment_id,
            generated_at=eda.generated_at,
            target_column=record.target_column,
        )
        dictionary = read_model_if_exists(self._store, layout.data_dictionary, DataDictionary)
        stored = read_model_if_exists(self._store, layout.selected_features, FeatureSelection)

        if stored is not None:
            selected = set(stored.selected_features)
        else:
            proposed, _reasons = default_feature_selection(eda, leakage, record.target_column)
            selected = set(proposed)

        warnings_by_column: dict[str, list] = {}
        for warning in eda.warnings:
            if warning.column:
                warnings_by_column.setdefault(warning.column, []).append(warning)

        items: list[FeatureReviewItem] = []
        for profile in eda.columns:
            if profile.name == record.target_column:
                continue
            risk = leakage.risk_for(profile.name)
            column_warnings = warnings_by_column.get(profile.name, [])
            documentation = dictionary.for_column(profile.name) if dictionary else None
            items.append(
                FeatureReviewItem(
                    feature=profile.name,
                    selected=profile.name in selected,
                    semantic_type=profile.semantic_type,
                    dtype=profile.dtype,
                    missing_percentage=profile.missing_percentage,
                    unique_count=profile.unique_count,
                    unique_percentage=profile.unique_percentage,
                    is_constant=profile.is_constant,
                    is_high_cardinality=profile.is_high_cardinality,
                    is_likely_id=profile.is_likely_id,
                    leakage_risk=risk.risk_level if risk else LeakageRiskLevel.NONE,
                    max_severity=risk.max_severity if risk else None,
                    recommended_action=_recommended_action(risk, column_warnings),
                    reasons=(risk.reasons if risk else [])
                    + [
                        w.message
                        for w in column_warnings
                        if w.recommended_action != RecommendedAction.KEEP
                    ],
                    rules=risk.rules if risk else [],
                    warnings=column_warnings,
                    documentation=documentation.business_definition if documentation else None,
                    available_at_prediction_time=(
                        documentation.available_at_prediction_time if documentation else None
                    ),
                )
            )

        return FeatureReviewResponse(
            experiment_id=experiment_id,
            target_column=record.target_column,
            problem_type=eda.target.inferred_problem_type,
            features=items,
            selected_count=sum(1 for item in items if item.selected),
            excluded_count=sum(1 for item in items if not item.selected),
            decided=stored is not None,
        )

    def save_features(
        self, experiment_id: str, request: UpdateFeatureSelectionRequest, user: str | None = None
    ) -> FeatureSelection:
        """Persist the user's exact decision. Nothing is added or removed behind their back."""
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = self.eda(experiment_id)

        known = {profile.name for profile in eda.columns}
        unknown = [f for f in request.selected_features if f not in known]
        if unknown:
            raise ValidationError(
                "The selection contains columns that are not in the dataset.",
                {"unknown_features": unknown},
            )
        if record.target_column in request.selected_features:
            raise ValidationError(
                f"The target column '{record.target_column}' cannot be used as a feature."
            )

        selected = list(dict.fromkeys(request.selected_features))
        excluded = [f for f in known if f not in set(selected) and f != record.target_column]
        selection = FeatureSelection(
            selected_features=selected,
            excluded_features=excluded,
            exclusion_reasons={k: v for k, v in request.exclusion_reasons.items() if k in excluded},
            decided_at=datetime.now(UTC),
            decided_by=user,
        )
        write_model(self._store, layout.selected_features, selection)
        self._repository.update(
            experiment_id,
            status=ExperimentStatus.READY_FOR_TRAINING,
            current_stage="ready_for_training",
        )
        return selection

    # --- training ---------------------------------------------------------
    def training_config(
        self, experiment_id: str, root: str | None = None
    ) -> TrainingConfigResponse:
        """Defaults the UI renders: resolved problem type, metrics and the model catalogue."""
        record = self.get(experiment_id, root)
        eda = self.eda(experiment_id, root)
        problem_type = self._resolve_problem_type(record.requested_problem_type, eda)
        selection = self._stored_selection(record)
        return TrainingConfigResponse(
            experiment_id=experiment_id,
            problem_type=problem_type,
            requested_problem_type=record.requested_problem_type,
            primary_metric=resolve_primary_metric(problem_type, None),
            available_metrics=selectable_primary_metrics(problem_type),
            available_models=default_model_names(problem_type),
            selected_models=default_model_names(problem_type),
            validation_fraction=DEFAULT_VALIDATION_FRACTION,
            random_seed=DEFAULT_RANDOM_SEED,
            class_weighting=ClassWeighting.AUTO,
            preprocessing=PreprocessingConfig(),
            selected_feature_count=len(selection.selected_features) if selection else 0,
        )

    def start_training(
        self, experiment_id: str, request: TrainingConfigRequest, user: str | None = None
    ) -> ExperimentRecord:
        """Freeze the configuration, write it as an artifact and start the training workflow."""
        record = self._repository.get(experiment_id)
        if record.status in _ACTIVE_TRAINING_STATUSES:
            raise ConflictError(
                f"Training is already running for this experiment (status {record.status.value})."
            )
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = self.eda(experiment_id)

        requested = (
            request.problem_type
            if request.problem_type is not RequestedProblemType.AUTO
            else record.requested_problem_type
        )
        problem_type = self._resolve_problem_type(requested, eda)

        selection = self._stored_selection(record)
        if selection is None:
            raise ConflictError(
                "Confirm the feature selection before starting training.",
                {"endpoint": f"/experiments/{experiment_id}/features"},
            )

        try:
            primary_metric = resolve_primary_metric(problem_type, request.primary_metric)
        except UnknownMetricError as error:
            raise ValidationError(str(error)) from error

        models = self._resolve_models(request, problem_type)
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=record.name,
            created_at=record.created_at,
            created_by=user or record.created_by,
            dataset=record.dataset,
            target_column=record.target_column,
            problem_type=problem_type,
            requested_problem_type=requested,
            primary_metric=primary_metric,
            feature_selection=selection,
            split=request.to_split_config(),
            preprocessing=request.preprocessing or PreprocessingConfig(),
            models=models,
            class_weighting=request.class_weighting,
            compute=ComputeConfig(),
            environment=EnvironmentCapture(
                python_version=environment_metadata()["python_version"],
                platform=environment_metadata()["platform"],
                package_versions=package_versions(),
                ml_factory_version=ML_FACTORY_VERSION,
                captured_at=datetime.now(UTC),
            ),
            artifact_prefix=layout.base,
        )
        write_model(self._store, layout.experiment_config, config)

        updated = self._repository.update(
            experiment_id,
            status=ExperimentStatus.READY_FOR_TRAINING,
            current_stage="training_requested",
            primary_metric=primary_metric,
            requested_models=[spec.name for spec in models],
        )
        handle = self._orchestrator.start_training(updated)
        LOGGER.info("Started training for %s (%s)", experiment_id, handle.execution_id)
        return self._repository.update(experiment_id, training_execution_arn=handle.execution_id)

    def training_status(
        self, experiment_id: str, root: str | None = None
    ) -> TrainingStatusResponse:
        """Per-model progress, derived from the artifacts each training job writes.

        The artifacts are the source of truth: a model that wrote metadata succeeded, one that
        wrote a failure record failed. Nothing has to keep a per-model status field in sync.
        """
        record = self.get(experiment_id, root)
        layout = ExperimentLayout(base=record.artifact_prefix)
        states = [
            self._model_state(layout, name, record.status)
            for name in sorted(record.requested_models)
        ]
        return TrainingStatusResponse(
            experiment_id=experiment_id,
            status=record.status,
            current_stage=record.current_stage,
            models=states,
            best_model=record.best_model,
            best_score=record.best_score,
            primary_metric=record.primary_metric,
            failure_message=record.failure_message,
            updated_at=record.updated_at,
        )

    # --- helpers ----------------------------------------------------------
    def _model_state(
        self, layout: ExperimentLayout, name: str, experiment_status: ExperimentStatus
    ) -> ModelRunState:
        metadata = read_model_if_exists(self._store, layout.model_metadata(name), ModelMetadata)
        if metadata is not None:
            return ModelRunState(
                model_name=name,
                display_name=metadata.display_name,
                status=ModelRunStatus.COMPLETED,
                primary_score=metadata.primary_score,
            )
        failure = read_model_if_exists(self._store, layout.model_failure(name), ModelFailure)
        if failure is not None:
            return ModelRunState(
                model_name=name,
                display_name=_display_name(name),
                status=ModelRunStatus.FAILED,
                failure_message=failure.message,
            )
        pending = (
            ModelRunStatus.RUNNING
            if experiment_status is ExperimentStatus.TRAINING
            else ModelRunStatus.QUEUED
        )
        return ModelRunState(model_name=name, display_name=_display_name(name), status=pending)

    def _layout(self, experiment_id: str) -> ExperimentLayout:
        return ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)

    def _artifact(
        self, experiment_id: str, locator, model_type, label: str, root: str | None = None
    ):
        record = self.get(experiment_id, root)
        layout = ExperimentLayout(base=record.artifact_prefix)
        artifact = read_model_if_exists(self._store, locator(layout), model_type)
        if artifact is None:
            raise ArtifactNotReadyError(
                f"The {label} for this experiment is not available yet "
                f"(current status: {record.status.value}).",
                {"status": record.status.value, "stage": record.current_stage},
            )
        return artifact

    def _stored_selection(self, record: ExperimentRecord) -> FeatureSelection | None:
        layout = ExperimentLayout(base=record.artifact_prefix)
        return read_model_if_exists(self._store, layout.selected_features, FeatureSelection)

    def _resolve_problem_type(self, requested: RequestedProblemType, eda: EdaReport) -> ProblemType:
        if requested is not RequestedProblemType.AUTO:
            return ProblemType(requested.value)
        inferred = eda.target.inferred_problem_type
        if inferred is None:
            raise ConflictError(
                "The problem type could not be inferred from the target column. Choose one "
                "explicitly in the training configuration.",
                {"target_column": eda.target.column},
            )
        return inferred

    def _resolve_models(
        self, request: TrainingConfigRequest, problem_type: ProblemType
    ) -> list[ModelSpec]:
        names = request.models or default_model_names(problem_type)
        if not names:
            raise ValidationError(
                f"No model plugin is available for {problem_type.value} in this environment."
            )
        specs: list[ModelSpec] = []
        for name in dict.fromkeys(names):
            try:
                plugin = get_plugin(name)
            except UnknownModelError as error:
                raise ValidationError(str(error)) from error
            if not plugin.supports(problem_type):
                raise ValidationError(
                    f"Model '{name}' does not support {problem_type.value}.",
                    {"supported_models": [p.name for p in plugins_for(problem_type)]},
                )
            available, reason = plugin.is_available()
            if not available:
                raise ValidationError(f"Model '{name}' is not available: {reason}")
            specs.append(ModelSpec(name=name, params=dict(request.model_parameters.get(name, {}))))
        return specs

    def _validate_dataset(self, uri: str, file_format: str | None) -> DatasetReference:
        """Format, allow-list and existence checks — all before any workflow is started."""
        candidate = uri.strip()
        if is_s3_uri(candidate):
            try:
                parse_s3_uri(candidate)
            except InvalidS3UriError as error:
                raise ValidationError(str(error)) from error
        else:
            raise ValidationError(
                "The dataset URI must be an S3 URI of the form s3://bucket/key.",
                {"dataset_uri": candidate},
            )

        if not self._settings.dataset_uri_allowed(candidate):
            raise ValidationError(
                "This dataset location is not on the approved list for the ML Factory.",
                {
                    "dataset_uri": candidate,
                    "allowed_prefixes": self._settings.allowed_dataset_prefixes,
                },
            )

        resolved_format = (file_format or detect_format(candidate)).lower()
        if resolved_format == "unknown":
            raise ValidationError(
                "Could not determine the dataset format. Supported: csv, tsv, parquet.",
                {"dataset_uri": candidate},
            )

        metadata = self._store.stat(candidate) if not candidate.endswith("/") else None
        if metadata is None and not candidate.endswith("/"):
            raise ValidationError(
                "The dataset could not be found, or the platform has no access to it.",
                {"dataset_uri": candidate},
            )
        return DatasetReference(
            uri=candidate,
            file_format=resolved_format,
            version_id=metadata.version_id if metadata else None,
            etag=metadata.etag if metadata else None,
            size_bytes=metadata.size_bytes if metadata else None,
            last_modified=metadata.last_modified if metadata else None,
        )


def _recommended_action(risk, column_warnings) -> RecommendedAction:
    """The strongest recommendation across leakage findings and EDA warnings.

    A column can be uninteresting for reasons that have nothing to do with leakage — it is
    constant, it is free text, it is almost entirely missing — and the review table must say
    so rather than showing "keep" next to a warning.
    """
    candidates = [risk.recommended_action] if risk else []
    candidates.extend(warning.recommended_action for warning in column_warnings)
    if not candidates:
        return RecommendedAction.KEEP
    return max(candidates, key=lambda action: ACTION_ORDER[action])


def _new_experiment_id() -> str:
    return f"exp-{uuid.uuid4().hex[:12]}"


def _display_name(model_name: str) -> str:
    try:
        return get_plugin(model_name).display_name
    except UnknownModelError:
        return model_name
