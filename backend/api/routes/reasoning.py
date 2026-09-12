"""Semantic reasoning endpoints.

Available only once deterministic artifacts exist, and only when Bedrock is configured.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from backend.api.dependencies import get_container
from backend.schemas.reasoning import ReasoningRequest
from backend.services.reasoning import ReasoningService
from ml_engine.contracts.reasoning import ReasoningReport

router = APIRouter(prefix="/experiments", tags=["reasoning"])


def get_reasoning_service(request: Request) -> ReasoningService:
    return get_container(request).reasoning


ReasoningServiceDep = Annotated[ReasoningService, Depends(get_reasoning_service)]


@router.post("/{experiment_id}/reasoning", response_model=ReasoningReport)
def run_reasoning(
    experiment_id: str, request: ReasoningRequest, service: ReasoningServiceDep
) -> ReasoningReport:
    """Interpret the deterministic results. The model never computes or changes anything."""
    return service.run(experiment_id, request)


@router.get("/{experiment_id}/reasoning", response_model=ReasoningReport)
def get_reasoning(experiment_id: str, service: ReasoningServiceDep) -> ReasoningReport:
    return service.get(experiment_id)
