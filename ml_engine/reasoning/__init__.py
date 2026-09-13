"""Semantic reasoning over deterministic results. Never a source of numbers."""

from ml_engine.reasoning.analyzer import run_reasoning
from ml_engine.reasoning.client import (
    BedrockReasoningClient,
    ReasoningClient,
    ReasoningUnavailableError,
)
from ml_engine.reasoning.context import build_context, referenced_columns
from ml_engine.reasoning.json_io import ReasoningOutputError
from ml_engine.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt
from ml_engine.reasoning.proposals import (
    PROPOSAL_SYSTEM_PROMPT,
    build_proposal_prompt,
    propose_features,
)

__all__ = [
    "PROPOSAL_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "BedrockReasoningClient",
    "ReasoningClient",
    "ReasoningOutputError",
    "ReasoningUnavailableError",
    "build_context",
    "build_proposal_prompt",
    "build_user_prompt",
    "propose_features",
    "referenced_columns",
    "run_reasoning",
]
