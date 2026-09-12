"""Profiling job: deterministic EDA plus deterministic leakage screening.

Runs as a SageMaker Processing job. Reads the source dataset, writes ``eda/eda.json`` and
``eda/leakage.json``. No model is trained here and no LLM is involved.
"""

import argparse
import logging
import sys

from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.io import ExperimentLayout, ObjectStore, load_dataset, write_model
from ml_engine.leakage import LeakageConfig, analyze_leakage
from ml_engine.profiling import ProfilingConfig, profile_dataset

LOGGER = logging.getLogger("ml_factory.jobs.profiling")


def run_profiling(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    dataset_uri: str,
    target_column: str,
    file_format: str | None = None,
    profiling_config: ProfilingConfig | None = None,
    leakage_config: LeakageConfig | None = None,
) -> tuple[EdaReport, LeakageReport]:
    """Profile a dataset and screen it for leakage. Writes both artifacts, returns both."""
    profiling_config = profiling_config or ProfilingConfig()
    LOGGER.info("Loading dataset %s", dataset_uri)
    dataset = load_dataset(store, dataset_uri, file_format, max_rows=profiling_config.max_rows)

    if target_column not in dataset.frame.columns:
        raise JobError(
            "TARGET_COLUMN_MISSING",
            f"Target column '{target_column}' is not present in the dataset. Available columns: "
            + ", ".join(map(str, dataset.frame.columns[:50])),
        )

    LOGGER.info("Profiling %s rows x %s columns", *dataset.frame.shape)
    eda = profile_dataset(
        dataset.frame,
        experiment_id=experiment_id,
        target_column=target_column,
        source_uri=dataset.source_uri,
        file_format=dataset.file_format,
        config=profiling_config,
        source_size_bytes=dataset.source_size_bytes,
        sampled=dataset.sampled,
    )
    write_model(store, layout.eda, eda)
    LOGGER.info("Wrote %s", layout.eda)

    problem_type = eda.target.inferred_problem_type
    if problem_type is None:
        leakage = LeakageReport(
            experiment_id=experiment_id,
            generated_at=eda.generated_at,
            target_column=target_column,
            checks_skipped={
                "all": "the target column cannot support supervised learning, so leakage "
                "analysis was not meaningful"
            },
        )
    else:
        leakage = analyze_leakage(
            dataset.frame,
            experiment_id=experiment_id,
            target_column=target_column,
            problem_type=problem_type,
            eda=eda,
            config=leakage_config or LeakageConfig(),
        )
    write_model(store, layout.leakage, leakage)
    LOGGER.info("Wrote %s (%d findings)", layout.leakage, len(leakage.findings))
    return eda, leakage


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_profiling(
        store,
        layout,
        experiment_id=args.experiment_id,
        dataset_uri=args.dataset_uri,
        target_column=args.target_column,
        file_format=args.file_format,
        profiling_config=ProfilingConfig(max_rows=args.max_rows) if args.max_rows else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory profiling job (EDA + deterministic leakage screening)")
    parser.prog = "profiling-job"
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--file-format", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda _args, layout: layout.path("eda/failure.json"),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
