"""The reasoning layer must interpret, never compute — and never invent a column."""

import json

import pytest

from ml_engine.contracts.common import ProblemType
from ml_engine.reasoning import ReasoningOutputError, build_context, run_reasoning
from ml_engine.reasoning.prompts import SYSTEM_PROMPT


class FakeClient:
    """Returns a canned payload and records what it was asked."""

    model_id = "fake.model-v1"

    def __init__(self, payload: dict | str) -> None:
        self.payload = payload
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def _valid_payload(**overrides) -> dict:
    payload = {
        "executive_summary": "The dataset is usable with reservations.",
        "semantic_leakage": [
            {
                "feature": "customer_id",
                "severity": "medium",
                "confidence": "high",
                "reasoning": "Identifiers cannot generalize.",
                "recommended_action": "consider_excluding",
                "evidence_basis": ["likely_identifier warning"],
            }
        ],
        "feature_interpretations": [{"feature": "contract", "interpretation": "Contract type."}],
        "data_quality_concerns": [
            {
                "topic": "missingness",
                "severity": "medium",
                "explanation": "Charges are often missing.",
                "affected_columns": ["monthly_charges"],
            }
        ],
        "assumptions": ["The target is known only after the billing period closes."],
        "limitations": [],
        "proposed_experiments": [
            {
                "title": "Drop identifier",
                "hypothesis": "Performance is unchanged without it.",
                "proposed_changes": ["exclude customer_id"],
                "expected_insight": "Confirms the model is not memorizing.",
                "priority": "high",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_valid_output_is_validated_and_attributed(classification_eda):
    client = FakeClient(_valid_payload())
    report = run_reasoning(client, experiment_id="exp-1", eda=classification_eda)
    assert report.experiment_id == "exp-1"
    assert report.model_id == "fake.model-v1"
    assert report.semantic_leakage[0].feature == "customer_id"
    assert report.disclaimer


def test_context_contains_results_not_rows(classification_eda):
    context = build_context(eda=classification_eda, leakage=None, dictionary=None, summary=None)
    serialized = json.dumps(context)
    assert "row_count" in serialized
    assert "C00001" not in serialized  # no raw identifiers
    assert context["target"]["problem_type"] == ProblemType.BINARY_CLASSIFICATION.value


def test_prompt_forbids_computation(classification_eda):
    client = FakeClient(_valid_payload())
    run_reasoning(client, experiment_id="exp-1", eda=classification_eda)
    assert "must NOT" in client.system_prompt
    assert "recompute" in SYSTEM_PROMPT


def test_invented_columns_are_dropped(classification_eda):
    payload = _valid_payload(
        semantic_leakage=[
            {
                "feature": "column_that_does_not_exist",
                "severity": "critical",
                "confidence": "high",
                "reasoning": "Invented.",
                "recommended_action": "strongly_consider_excluding",
                "evidence_basis": [],
            }
        ]
    )
    report = run_reasoning(FakeClient(payload), experiment_id="exp-1", eda=classification_eda)
    assert report.semantic_leakage == []
    assert any("do not exist" in limitation for limitation in report.limitations)


def test_json_wrapped_in_prose_is_recovered(classification_eda):
    raw = "Here is the analysis:\n" + json.dumps(_valid_payload()) + "\nHope that helps."
    report = run_reasoning(FakeClient(raw), experiment_id="exp-1", eda=classification_eda)
    assert report.executive_summary


def test_non_json_output_is_discarded(classification_eda):
    with pytest.raises(ReasoningOutputError, match="did not return JSON"):
        run_reasoning(FakeClient("I refuse."), experiment_id="exp-1", eda=classification_eda)


def test_schema_violations_are_discarded(classification_eda):
    payload = _valid_payload(semantic_leakage=[{"feature": "customer_id", "severity": "extreme"}])
    with pytest.raises(ReasoningOutputError, match="did not match the required schema"):
        run_reasoning(FakeClient(payload), experiment_id="exp-1", eda=classification_eda)


def test_model_supplied_metrics_cannot_enter_the_report(classification_eda):
    """Unknown keys are rejected outright, so a model cannot smuggle in numbers."""
    payload = _valid_payload(roc_auc=0.99, best_model="its_favourite")
    with pytest.raises(ReasoningOutputError):
        run_reasoning(FakeClient(payload), experiment_id="exp-1", eda=classification_eda)
