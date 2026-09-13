"""Run the semantic reasoning pass and validate what comes back.

Three guarantees enforced here, not merely requested in the prompt:
1. the output must parse as the ``ReasoningReport`` schema, or the run fails;
2. findings about columns that do not exist are dropped as hallucinations;
3. nothing in the result can change an experiment — the report is read-only output.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from ml_engine.contracts.comparison import ExperimentSummary
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.reasoning import ReasoningReport
from ml_engine.reasoning.client import ReasoningClient, ReasoningUnavailableError
from ml_engine.reasoning.context import build_context, referenced_columns
from ml_engine.reasoning.json_io import ReasoningOutputError, parse_json_object
from ml_engine.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt

LOGGER = logging.getLogger("ml_factory.reasoning")


def run_reasoning(
    client: ReasoningClient,
    *,
    experiment_id: str,
    eda: EdaReport,
    leakage: LeakageReport | None = None,
    dictionary: DataDictionary | None = None,
    summary: ExperimentSummary | None = None,
    target_definition: str | None = None,
    prediction_timing: str | None = None,
    question: str | None = None,
) -> ReasoningReport:
    """Produce a validated :class:`ReasoningReport` from deterministic artifacts."""
    context = build_context(
        eda=eda,
        leakage=leakage,
        dictionary=dictionary,
        summary=summary,
        target_definition=target_definition,
        prediction_timing=prediction_timing,
    )
    raw = client.invoke(SYSTEM_PROMPT, build_user_prompt(context, question))
    payload = parse_json_object(raw)
    report = _validate(payload, experiment_id=experiment_id, model_id=client.model_id)
    return _drop_unknown_columns(report, referenced_columns(context))


def _validate(payload: dict[str, Any], *, experiment_id: str, model_id: str) -> ReasoningReport:
    payload = dict(payload)
    payload.pop("schema_version", None)
    payload.pop("disclaimer", None)
    payload["experiment_id"] = experiment_id
    payload["model_id"] = model_id
    payload["generated_at"] = datetime.now(UTC)
    payload.setdefault("executive_summary", "")
    payload.setdefault("input_artifacts", [])
    try:
        return ReasoningReport.model_validate(payload)
    except PydanticValidationError as exc:
        raise ReasoningOutputError(
            "The reasoning model's output did not match the required schema and was discarded: "
            f"{exc.error_count()} validation error(s)."
        ) from exc


def _drop_unknown_columns(report: ReasoningReport, known_columns: set[str]) -> ReasoningReport:
    """Remove references to columns the model invented. Silence beats a confident fiction."""
    if not known_columns:
        return report
    dropped: list[str] = []

    semantic = []
    for finding in report.semantic_leakage:
        if finding.feature in known_columns:
            semantic.append(finding)
        else:
            dropped.append(finding.feature)
    interpretations = []
    for interpretation in report.feature_interpretations:
        if interpretation.feature in known_columns:
            interpretations.append(interpretation)
        else:
            dropped.append(interpretation.feature)
    for concern in report.data_quality_concerns:
        concern.affected_columns = [c for c in concern.affected_columns if c in known_columns]

    if dropped:
        LOGGER.warning("Dropped reasoning findings for unknown columns: %s", sorted(set(dropped)))
        report.limitations.append(
            "Some findings referenced columns that do not exist in the dataset and were removed: "
            + ", ".join(sorted(set(dropped)))
        )
    report.semantic_leakage = semantic
    report.feature_interpretations = interpretations
    return report


__all__ = ["ReasoningOutputError", "ReasoningUnavailableError", "run_reasoning"]
