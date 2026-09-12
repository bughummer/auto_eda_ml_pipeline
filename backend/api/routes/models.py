"""Model catalogue endpoint. Reads the plugin registry; no model names are hardcoded."""

from fastapi import APIRouter, Query

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.model import ModelDescriptor
from ml_engine.evaluation import selectable_primary_metrics
from ml_engine.models import catalogue

router = APIRouter(tags=["catalogue"])


@router.get("/models", response_model=list[ModelDescriptor])
def list_models(
    problem_type: ProblemType | None = Query(default=None),
) -> list[ModelDescriptor]:
    return catalogue(problem_type)


@router.get("/metrics", response_model=list[str])
def list_metrics(problem_type: ProblemType) -> list[str]:
    """Metrics that may be selected as the primary (ranking) metric for a problem type."""
    return selectable_primary_metrics(problem_type)
