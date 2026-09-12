"""Evaluation job: aggregate every model result into a comparison and an experiment summary.

Runs as a SageMaker Processing job after the training Map state. Failed models are read from
their ``failure.json`` artifacts so the comparison shows the complete picture.
"""

import argparse
import logging
import sys

from jobs._common.experiment_state import ExperimentStateWriter, build_state_writer
from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.comparison import ComparisonReport
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model,
    read_model_if_exists,
    write_model,
)
from ml_engine.reporting import build_comparison, build_summary, final_status

LOGGER = logging.getLogger("ml_factory.jobs.evaluation")


def collect_model_results(
    store: ObjectStore, layout: ExperimentLayout, model_names: list[str]
) -> tuple[list[ModelMetadata], list[ModelFailure]]:
    """Read every configured model's outcome: metadata when it succeeded, failure otherwise."""
    successes: list[ModelMetadata] = []
    failures: list[ModelFailure] = []
    for name in model_names:
        metadata = read_model_if_exists(store, layout.model_metadata(name), ModelMetadata)
        if metadata is not None:
            successes.append(metadata)
            continue
        failure = read_model_if_exists(store, layout.model_failure(name), ModelFailure)
        failures.append(
            failure
            or ModelFailure(
                experiment_id=layout.base.rsplit("/", 1)[-1],
                model_name=name,
                error_code="MODEL_RESULT_MISSING",
                message=(
                    "The model produced neither metrics nor a failure record. Its training job "
                    "most likely terminated abnormally."
                ),
                failed_at=_now(),
            )
        )
    return successes, failures


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def run_evaluation(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    state_writer: ExperimentStateWriter | None = None,
) -> ComparisonReport:
    """Build ``comparison.json`` and ``experiment_summary.json``."""
    try:
        config = read_model(store, layout.experiment_config, ExperimentConfig)
    except FileNotFoundError as error:
        raise JobError(
            "EXPERIMENT_CONFIG_MISSING", "The experiment configuration artifact is missing."
        ) from error

    model_names = [spec.name for spec in config.enabled_models_specs()]
    successes, failures = collect_model_results(store, layout, model_names)
    LOGGER.info("Collected %d successful and %d failed models", len(successes), len(failures))

    comparison = build_comparison(
        experiment_id=experiment_id,
        problem_type=config.problem_type,
        primary_metric=config.primary_metric,
        successes=successes,
        failures=failures,
    )
    write_model(store, layout.comparison, comparison)

    summary = build_summary(
        config=config,
        status=final_status(comparison),
        comparison=comparison,
        eda=read_model_if_exists(store, layout.eda, EdaReport),
        leakage=read_model_if_exists(store, layout.leakage, LeakageReport),
        models=successes,
    )
    write_model(store, layout.summary, summary)

    if state_writer is not None:
        state_writer.record_completion(experiment_id, comparison)

    LOGGER.info(
        "Best model: %s (%s=%s)",
        comparison.best_model,
        comparison.primary_metric,
        comparison.best_score,
    )
    return comparison


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_evaluation(
        store,
        layout,
        experiment_id=args.experiment_id,
        state_writer=build_state_writer(store, layout, enabled=not args.no_record_state),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory evaluation job (comparison and experiment summary)")
    parser.prog = "evaluation-job"
    parser.add_argument(
        "--no-record-state",
        action="store_true",
        help="Build the reports without writing the terminal experiment state.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda _args, layout: layout.path("comparison/failure.json"),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
