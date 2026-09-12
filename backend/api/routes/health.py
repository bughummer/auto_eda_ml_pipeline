"""Liveness and configuration reporting."""

from fastapi import APIRouter

from backend.api.dependencies import ContainerDep
from backend.schemas.experiments import HealthResponse
from ml_engine.contracts.common import ProblemType
from ml_engine.models import default_model_names

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: ContainerDep) -> HealthResponse:
    settings = container.settings
    problems = settings.validate_for_mode()
    models = sorted(
        {name for problem_type in ProblemType for name in default_model_names(problem_type)}
    )
    return HealthResponse(
        status="degraded" if problems else "ok",
        mode=settings.mode.value,
        orchestrator=container.orchestrator.name,
        artifact_root=settings.artifact_root,
        credential_source=settings.credential_source(),
        bedrock_enabled=settings.bedrock_enabled,
        configuration_problems=problems,
        available_models=models,
    )
