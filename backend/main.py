"""FastAPI application: the ML Factory control plane.

It validates requests, records experiment state, starts Step Functions executions and serves
artifacts. It never performs heavy data or ML work, and the browser never talks to AWS.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import experiments, health, models
from backend.config import Settings, get_settings
from backend.container import build_container
from backend.errors import register_error_handlers

LOGGER = logging.getLogger("ml_factory.api")

DESCRIPTION = """
Internal platform that accelerates classical-ML POCs: automated EDA, deterministic data
quality and leakage screening, supervised feature review, classical model training and
comparison, and — strictly after the deterministic results exist — semantic interpretation.
"""


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = build_container(settings)
        LOGGER.info("ML Factory API ready (mode=%s)", settings.mode.value)
        try:
            yield
        finally:
            app.state.container.shutdown()

    app = FastAPI(
        title="ML Factory",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    register_error_handlers(app, LOGGER)

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(models.router, prefix=settings.api_prefix)
    app.include_router(experiments.router, prefix=settings.api_prefix)
    return app


app = create_app()
