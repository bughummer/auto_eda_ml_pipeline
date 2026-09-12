"""FastAPI application: the ML Factory control plane.

It validates requests, records experiment state, starts Step Functions executions and serves
artifacts. It never performs heavy data or ML work, and the browser never talks to AWS.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import dictionary, experiments, health, models, reasoning
from backend.config import Settings, get_settings
from backend.container import AppContainer, build_container
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


def create_app(
    settings: Settings | None = None, *, container: AppContainer | None = None
) -> FastAPI:
    """Build the application.

    ``container`` lets a caller supply an already-wired graph; when it does, the caller owns
    its lifetime. Without one the app builds its own from the settings.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or build_container(settings)
        LOGGER.info("ML Factory API ready (artifacts %s)", settings.artifact_root)
        try:
            yield
        finally:
            if owned:
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
    app.include_router(dictionary.router, prefix=settings.api_prefix)
    app.include_router(reasoning.router, prefix=settings.api_prefix)

    if settings.static_dir is not None:
        mount_frontend(app, settings.static_dir)
    return app


def mount_frontend(app: FastAPI, directory: Path) -> None:
    """Serve the built single-page app alongside the API from one process.

    This is what lets the platform ship as a single container: uvicorn answers /api/v1/* and
    everything else falls through to the SPA's index.html so client-side routes survive a
    page reload. Routers are registered first, so the mount can never shadow the API.
    """
    root = directory.expanduser()
    index = root / "index.html"
    if not index.is_file():
        LOGGER.warning("No frontend build at %s; serving the API only", root)
        return

    assets = root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def serve_spa(spa_path: str) -> FileResponse:
        candidate = (root / spa_path).resolve()
        if spa_path and candidate.is_file() and candidate.is_relative_to(root.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)

    LOGGER.info("Serving the frontend from %s", root)


app = create_app()
