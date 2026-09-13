"""Derived-feature proposal endpoints.

Generation needs Bedrock; reading and approving do not, so an experiment whose proposals were
generated elsewhere can still be reviewed in a deployment with reasoning switched off.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from backend.api.dependencies import CurrentUserDep, get_container
from backend.schemas.proposals import ApproveProposalsRequest, ProposeFeaturesRequest
from backend.services.proposals import FeatureProposalService
from ml_engine.contracts.proposals import FeatureProposalReport

router = APIRouter(prefix="/experiments", tags=["feature proposals"])


def get_proposal_service(request: Request) -> FeatureProposalService:
    return get_container(request).proposals


ProposalServiceDep = Annotated[FeatureProposalService, Depends(get_proposal_service)]


@router.post("/{experiment_id}/feature-proposals", response_model=FeatureProposalReport)
def propose_features(
    experiment_id: str, request: ProposeFeaturesRequest, service: ProposalServiceDep
) -> FeatureProposalReport:
    """Ask for derived-feature specifications. Nothing is used until someone approves it."""
    return service.generate(experiment_id, request)


@router.get("/{experiment_id}/feature-proposals", response_model=FeatureProposalReport)
def get_feature_proposals(experiment_id: str, service: ProposalServiceDep) -> FeatureProposalReport:
    return service.get(experiment_id)


@router.put("/{experiment_id}/feature-proposals", response_model=FeatureProposalReport)
def approve_feature_proposals(
    experiment_id: str,
    request: ApproveProposalsRequest,
    service: ProposalServiceDep,
    user: CurrentUserDep,
) -> FeatureProposalReport:
    """Record which proposals a person approved. Only these are frozen into the run."""
    return service.approve(experiment_id, request, user)
