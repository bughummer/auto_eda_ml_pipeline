"""Prompt construction for the semantic reasoning layer.

The instructions are part of the platform's safety boundary: the model interprets, it never
calculates, ranks, or changes anything.
"""

import json
from typing import Any

SYSTEM_PROMPT = """\
You are a senior machine-learning reviewer inside an internal ML platform. You are given the
DETERMINISTIC outputs of an automated analysis: statistics computed with pandas, leakage
checks computed with explicit rules, and model metrics computed with scikit-learn.

Your role is interpretation, not computation. You must:
- reason about business meaning, data collection timing and plausibility
- identify features that are unlikely to be available at prediction time (semantic leakage)
- explain what the deterministic findings imply for the modelling approach
- state assumptions and limitations explicitly

You must NOT:
- invent, recompute, adjust or estimate any statistic, metric, score or ranking
- claim a feature is or is not leaking without grounding it in the supplied evidence
- recommend silently removing features; every recommendation is for human review
- refer to any column that does not appear in the supplied context

Quantities you mention must be copied verbatim from the context. If evidence is insufficient,
say so. Respond with a single JSON object and nothing else: no prose before or after, no
markdown fences.
"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "executive_summary": "string - 3 to 6 sentences for a data scientist",
    "semantic_leakage": [
        {
            "feature": "string - must be a column from the context",
            "severity": "one of: info, low, medium, high, critical",
            "confidence": "one of: low, medium, high",
            "reasoning": "string - why this may not be available at prediction time",
            "recommended_action": (
                "one of: keep, review, consider_excluding, strongly_consider_excluding"
            ),
            "evidence_basis": ["string - which context facts support this"],
        }
    ],
    "feature_interpretations": [
        {"feature": "string", "interpretation": "string", "concern": "string or null"}
    ],
    "data_quality_concerns": [
        {
            "topic": "string",
            "severity": "one of: info, low, medium, high, critical",
            "explanation": "string",
            "affected_columns": ["string"],
        }
    ],
    "model_behaviour_narrative": "string or null - only if model results are present",
    "assumptions": ["string"],
    "limitations": ["string"],
    "proposed_experiments": [
        {
            "title": "string",
            "hypothesis": "string",
            "proposed_changes": ["string"],
            "expected_insight": "string",
            "priority": "one of: low, medium, high",
        }
    ],
}


def build_user_prompt(context: dict[str, Any], question: str | None = None) -> str:
    """Evidence plus the required output shape. Deterministic — no randomness in the prompt."""
    sections = [
        "## Deterministic analysis context",
        json.dumps(context, indent=2, sort_keys=True, default=str),
        "",
        "## Required JSON output schema",
        json.dumps(OUTPUT_SCHEMA, indent=2, sort_keys=True),
        "",
        "## Instructions",
        "- Use only the context above as evidence.",
        "- Every 'feature' value must be one of the column names in the context.",
        "- Omit a list rather than filling it with speculation.",
        "- Return only the JSON object.",
    ]
    if question:
        sections.extend(["", "## Additional question from the user", question.strip()])
    return "\n".join(sections)
