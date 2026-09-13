"""Asking a model to propose derived features, and refusing everything it gets wrong.

The model reads the deterministic profile and, where one was supplied, the data dictionary —
the only input that says what a column *means*, which no statistic can supply. What comes back
is a list of specifications from a closed vocabulary, never code.

Nothing here is trusted. Every returned object is validated against the specification schema
and then against the same rules preparation applies, so a proposal that names a missing column,
reads the target or misuses a type is refused before anyone is shown it. Refusals are kept and
returned: a reviewer should see what the model suggested and why it was discarded.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.proposals import (
    MAX_PROPOSALS,
    FeatureProposal,
    FeatureProposalReport,
    ProposalRejection,
    RejectedProposal,
)
from ml_engine.features import column_types_from_eda, validate_proposals
from ml_engine.reasoning.client import ReasoningClient
from ml_engine.reasoning.context import build_context
from ml_engine.reasoning.json_io import parse_json_object

LOGGER = logging.getLogger("ml_factory.reasoning.proposals")

_PROPOSAL_ADAPTER: TypeAdapter[FeatureProposal] = TypeAdapter(FeatureProposal)

PROPOSAL_SYSTEM_PROMPT = f"""\
You are a senior data scientist inside an internal ML platform. You are given the DETERMINISTIC
profile of a dataset and, when it exists, documentation describing what each column means.

Your task is to propose a small number of derived features that a domain expert would build by
hand. Your advantage over the platform's automatic preprocessing is meaning: the platform
already imputes, scales, one-hot encodes and extracts date parts, so proposing those is wasted.
Propose features that require knowing what the columns represent.

You do not write code. You return specifications using a fixed vocabulary of five operations,
and nothing else can be expressed:
- ratio: numerator / denominator, for per-unit quantities
- difference: left - right, for gaps and headroom
- date_difference: elapsed time between two DATE COLUMNS of this dataset
- map_categories: group category values by meaning, e.g. country to region
- is_missing: flag whether a column is missing, where absence is itself informative

Hard rules:
- NEVER use the target column as an input, in any position, in any operation. A feature derived
  from the target is leakage and will be refused.
- Use only column names that appear in the supplied context. Do not invent columns.
- Respect types: ratio and difference need numeric columns; date_difference needs date columns;
  map_categories needs a categorical column.
- There is no "today" or "now". date_difference takes two columns, both from the dataset.
- Propose at most {MAX_PROPOSALS} features. Fewer, well-reasoned ones are better than many.
- Every proposal needs a rationale a reviewer can judge, in one or two sentences.
- State available_at_prediction_time honestly: true only if every input column would be known
  at the moment a prediction is actually made. If you are unsure, say false and explain why in
  the rationale. A human confirms this; guessing true to look useful is the worst thing you can
  do here.

Return a single JSON object and nothing else: no prose before or after, no markdown fences.
"""

PROPOSAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "proposals": [
        {
            "name": "string - new column name, snake_case, must not already exist",
            "op": "one of: ratio, difference, date_difference, map_categories, is_missing",
            "rationale": "string - why this should help, in domain terms",
            "available_at_prediction_time": "boolean - see the hard rules",
            "//ratio": {"numerator": "column", "denominator": "column"},
            "//difference": {"left": "column", "right": "column"},
            "//date_difference": {
                "start": "date column",
                "end": "date column",
                "unit": "one of: days, months, years",
            },
            "//map_categories": {
                "column": "categorical column",
                "mapping": {"raw value": "group"},
                "default": "group for unmapped values, or null",
            },
            "//is_missing": {"column": "any column"},
        }
    ]
}

_EXAMPLE = {
    "proposals": [
        {
            "name": "charges_per_tenure_month",
            "op": "ratio",
            "numerator": "monthly_charges",
            "denominator": "tenure_months",
            "on_zero_denominator": "null",
            "rationale": "Spend normalised by relationship length separates expensive new "
            "customers from long-standing ones on the same plan.",
            "available_at_prediction_time": True,
        },
        {
            "name": "country_region",
            "op": "map_categories",
            "column": "country",
            "mapping": {"DE": "EU", "FR": "EU", "US": "NA"},
            "default": "other",
            "rationale": "The documentation describes country as an ISO code; grouping to "
            "sales region reduces cardinality without losing the geographic signal.",
            "available_at_prediction_time": True,
        },
    ]
}


def build_proposal_prompt(context: dict[str, Any]) -> str:
    """Evidence, the required shape, and one worked example. Deterministic."""
    return "\n".join(
        [
            "## Deterministic dataset context",
            json.dumps(context, indent=2, sort_keys=True, default=str),
            "",
            "## Required JSON output shape",
            "Keys prefixed with '//' document the fields each operation needs; use the fields "
            "themselves, not the '//' keys.",
            json.dumps(PROPOSAL_OUTPUT_SCHEMA, indent=2, sort_keys=True),
            "",
            "## Example of a well-formed response",
            json.dumps(_EXAMPLE, indent=2, sort_keys=True),
            "",
            "## Instructions",
            "- Return only the JSON object.",
            "- Return an empty 'proposals' list rather than inventing something marginal.",
            "- Do not propose scaling, imputation, one-hot encoding or date-part extraction: "
            "the platform already does those.",
        ]
    )


def propose_features(
    client: ReasoningClient,
    *,
    experiment_id: str,
    eda: EdaReport,
    target_column: str,
    leakage: LeakageReport | None = None,
    dictionary: DataDictionary | None = None,
    target_definition: str | None = None,
    prediction_timing: str | None = None,
) -> FeatureProposalReport:
    """Ask for derived-feature specifications and return only the ones that survive checking."""
    context = build_context(
        eda=eda,
        leakage=leakage,
        dictionary=dictionary,
        summary=None,
        target_definition=target_definition,
        prediction_timing=prediction_timing,
    )
    raw = client.invoke(PROPOSAL_SYSTEM_PROMPT, build_proposal_prompt(context))
    payload = parse_json_object(raw)

    parsed, malformed = _parse_specifications(payload.get("proposals"))
    accepted, refused = validate_proposals(
        parsed, column_types_from_eda(eda), target_column=target_column
    )

    LOGGER.info(
        "Feature proposals for %s: %d accepted, %d refused, %d malformed",
        experiment_id,
        len(accepted),
        len(refused),
        len(malformed),
    )
    return FeatureProposalReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        proposed_by=client.model_id,
        candidates=accepted,
        rejected=[*malformed, *refused],
    )


def _parse_specifications(
    raw: Any,
) -> tuple[list[FeatureProposal], list[RejectedProposal]]:
    """Validate each returned object against the specification schema, one at a time.

    One unparseable entry must not discard the rest, so they are validated individually rather
    than as a list.
    """
    if not isinstance(raw, list):
        return [], []

    parsed: list[FeatureProposal] = []
    malformed: list[RejectedProposal] = []
    for entry in raw:
        if not isinstance(entry, dict):
            malformed.append(
                RejectedProposal(
                    name="<unnamed>",
                    reason=ProposalRejection.MALFORMED_SPECIFICATION,
                    message="The model returned something that is not a specification object.",
                )
            )
            continue
        try:
            parsed.append(_PROPOSAL_ADAPTER.validate_python(entry))
        except PydanticValidationError as error:
            malformed.append(
                RejectedProposal(
                    name=str(entry.get("name") or "<unnamed>"),
                    reason=ProposalRejection.MALFORMED_SPECIFICATION,
                    message=(
                        f"The specification does not match any supported operation: "
                        f"{error.error_count()} validation error(s)."
                    ),
                )
            )
    return parsed, malformed
