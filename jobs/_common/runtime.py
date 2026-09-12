"""Shared runtime for the SageMaker job entrypoints.

Every job is a thin shell: parse arguments, build an object store, call a pure function from
``ml_engine``, write contracts back. Business logic never lives in this layer.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from ml_engine.contracts.model import ModelFailure
from ml_engine.io import (
    ExperimentLayout,
    LocalObjectStore,
    ObjectStore,
    S3ObjectStore,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.jobs")


class JobError(RuntimeError):
    """A job failure with a stable, user-safe error code."""

    def __init__(self, code: str, message: str, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.environ.get("ML_FACTORY_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def build_store(artifact_base: str, client: Any | None = None) -> ObjectStore:
    """S3 for ``s3://`` locations, the local filesystem otherwise. One rule, everywhere."""
    if artifact_base.startswith("s3://"):
        return S3ObjectStore(client)
    return LocalObjectStore()


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--artifact-base",
        required=True,
        help="Base URI of this experiment's artifacts (s3://... or a local directory).",
    )
    return parser


def run_entrypoint(
    parser: argparse.ArgumentParser,
    handler: Callable[[argparse.Namespace, ObjectStore, ExperimentLayout], Any],
    argv: Sequence[str] | None = None,
    *,
    failure_uri: Callable[[argparse.Namespace, ExperimentLayout], str] | None = None,
) -> int:
    """Run a job, converting any failure into a ``failure.json`` artifact and a non-zero exit.

    Step Functions catches the non-zero exit; the artifact gives the UI a user-safe reason
    without exposing a stack trace.
    """
    configure_logging()
    args = parser.parse_args(argv)
    layout = ExperimentLayout(base=args.artifact_base)
    store = build_store(args.artifact_base)
    job_name = parser.prog

    try:
        handler(args, store, layout)
    except JobError as error:
        LOGGER.exception("%s failed: %s", job_name, error.message)
        _write_failure(store, layout, args, error.code, error.message, error.details, failure_uri)
        return 1
    except Exception as error:
        LOGGER.exception("%s failed unexpectedly", job_name)
        _write_failure(
            store,
            layout,
            args,
            "INTERNAL_JOB_ERROR",
            f"{type(error).__name__}: {error}",
            {},
            failure_uri,
        )
        return 1
    LOGGER.info("%s completed", job_name)
    return 0


def _write_failure(
    store: ObjectStore,
    layout: ExperimentLayout,
    args: argparse.Namespace,
    code: str,
    message: str,
    details: dict[str, str],
    failure_uri: Callable[[argparse.Namespace, ExperimentLayout], str] | None,
) -> None:
    uri = failure_uri(args, layout) if failure_uri else layout.path("failure.json")
    failure = ModelFailure(
        experiment_id=args.experiment_id,
        model_name=getattr(args, "model_name", "") or "experiment",
        error_code=code,
        message=message,
        failed_at=datetime.now(UTC),
        details=details,
    )
    try:
        write_model(store, uri, failure)
    except Exception:
        LOGGER.exception("Could not write the failure artifact to %s", uri)
