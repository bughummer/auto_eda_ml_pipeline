"""FastAPI dependencies. Everything resolves from the container on ``app.state``."""

from typing import Annotated

from fastapi import Depends, Header, Request

from backend.config import Settings
from backend.container import AppContainer
from backend.services.experiments import ExperimentService


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_settings_dep(request: Request) -> Settings:
    return get_container(request).settings


def get_experiment_service(request: Request) -> ExperimentService:
    return get_container(request).experiments


def get_current_user(
    x_remote_user: Annotated[str | None, Header(alias="X-Remote-User")] = None,
) -> str | None:
    """Identity supplied by the corporate reverse proxy.

    The platform does not authenticate users itself; the proxy in front of it does, and
    passes the resolved identity through this header.
    """
    return x_remote_user


ContainerDep = Annotated[AppContainer, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ExperimentServiceDep = Annotated[ExperimentService, Depends(get_experiment_service)]
CurrentUserDep = Annotated[str | None, Depends(get_current_user)]
