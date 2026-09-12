"""Training job: fit exactly one model.

Runs as a SageMaker Training job, one job per model, so a single model failure is isolated
to its own branch of the state machine and never fails the experiment.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime

from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.model import ModelArtifacts, ModelFailure, ModelMetadata
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_joblib,
    read_parquet,
    save_joblib,
    write_model,
)
from ml_engine.models import ModelNotAvailableError, UnknownModelError, get_plugin
from ml_engine.training import TrainingError, train_model

from jobs._common.runtime import JobError, base_parser, run_entrypoint

LOGGER = logging.getLogger("ml_factory.jobs.training")


def run_training(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
) -> ModelMetadata:
    """Train one configured model and write its artifact and metadata."""
    config = read_model_config(store, layout)
    spec = next((m for m in config.enabled_models_specs() if m.name == model_name), None)
    if spec is None:
        raise JobError(
            "MODEL_NOT_CONFIGURED",
            f"Model '{model_name}' is not part of this experiment's configuration.",
        )

    try:
        plugin = get_plugin(model_name)
    except UnknownModelError as error:
        raise JobError("UNKNOWN_MODEL", str(error)) from error

    try:
        pipeline = load_joblib(store, layout.preprocessor(plugin.preprocessing_strategy))
    except FileNotFoundError as error:
        raise JobError(
            "PREPROCESSING_ARTIFACT_MISSING",
            "The fitted preprocessing pipeline is missing. The preparation stage must run first.",
        ) from error

    train_frame = read_parquet(store, layout.train_dataset)
    validation_frame = read_parquet(store, layout.validation_dataset)
    LOGGER.info(
        "Training %s on %d rows (%d validation rows)",
        model_name,
        len(train_frame),
        len(validation_frame),
    )

    try:
        metadata, estimator = train_model(
            plugin,
            experiment_id=experiment_id,
            problem_type=config.problem_type,
            pipeline=pipeline,
            train_frame=train_frame,
            validation_frame=validation_frame,
            target_column=config.target_column,
            param_overrides=dict(spec.params),
            primary_metric=config.primary_metric,
            random_seed=config.split.random_seed,
            class_weighting=config.class_weighting,
        )
    except (TrainingError, ModelNotAvailableError) as error:
        raise JobError("MODEL_TRAINING_FAILED", str(error), {"model": model_name}) from error

    model_uri = save_joblib(store, layout.model_artifact(model_name), estimator)
    metadata.artifacts = ModelArtifacts(
        model_uri=model_uri,
        metadata_uri=layout.model_metadata(model_name),
        preprocessor_uri=layout.preprocessor(plugin.preprocessing_strategy),
    )
    write_model(store, layout.model_metadata(model_name), metadata)
    LOGGER.info(
        "%s completed: %s=%s", model_name, metadata.primary_metric, metadata.primary_score
    )
    return metadata


def read_model_config(store: ObjectStore, layout: ExperimentLayout) -> ExperimentConfig:
    from ml_engine.io import read_model

    try:
        return read_model(store, layout.experiment_config, ExperimentConfig)
    except FileNotFoundError as error:
        raise JobError(
            "EXPERIMENT_CONFIG_MISSING",
            "The experiment configuration artifact is missing; training cannot start.",
        ) from error


def write_model_failure(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    code: str,
    message: str,
) -> None:
    """Per-model failure record, read by the evaluation job when it builds the comparison."""
    write_model(
        store,
        layout.model_failure(model_name),
        ModelFailure(
            experiment_id=experiment_id,
            model_name=model_name,
            error_code=code,
            message=message,
            failed_at=datetime.now(UTC),
        ),
    )


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_training(store, layout, experiment_id=args.experiment_id, model_name=args.model_name)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory training job (one model per job)")
    parser.prog = "training-job"
    parser.add_argument("--model-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda args, layout: layout.model_failure(args.model_name),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
