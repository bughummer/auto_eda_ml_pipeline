"""One error model for the whole API.

Users get a stable code and a safe message. Stack traces go to the logs, never to the browser.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class AppError(Exception):
    """Base class for every expected failure. Subclasses map to an HTTP status."""

    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def envelope(self, request_id: str | None = None) -> ErrorEnvelope:
        return ErrorEnvelope(
            error=ErrorDetail(
                code=self.code, message=self.message, details=self.details, request_id=request_id
            )
        )


class ValidationError(AppError):
    status_code = 422
    code = "VALIDATION_ERROR"


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class ConflictError(AppError):
    """The request is valid but the experiment is not in a state that allows it."""

    status_code = 409
    code = "INVALID_STATE"


class ArtifactNotReadyError(AppError):
    """The artifact does not exist yet because its stage has not completed."""

    status_code = 404
    code = "ARTIFACT_NOT_READY"


class UpstreamError(AppError):
    """An AWS dependency failed."""

    status_code = 502
    code = "UPSTREAM_ERROR"


class ConfigurationError(AppError):
    status_code = 500
    code = "CONFIGURATION_ERROR"


class FeatureDisabledError(AppError):
    status_code = 503
    code = "FEATURE_DISABLED"


def register_error_handlers(app: FastAPI, logger) -> None:
    """Install handlers so every error response has the same shape."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        if exc.status_code >= 500:
            logger.exception("%s: %s", exc.code, exc.message)
        else:
            logger.info("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code, content=exc.envelope(request_id).model_dump()
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        error = ValidationError(
            "The request payload is invalid.", {"errors": _readable_errors(exc.errors())}
        )
        return JSONResponse(status_code=422, content=error.envelope(request_id).model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        envelope = ErrorEnvelope(
            error=ErrorDetail(code=code, message=str(exc.detail), request_id=request_id)
        )
        return JSONResponse(status_code=exc.status_code, content=envelope.model_dump())

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error while handling %s", request.url.path)
        error = AppError(
            "An unexpected error occurred. The incident was logged with the request id."
        )
        return JSONResponse(status_code=500, content=error.envelope(request_id).model_dump())


def _readable_errors(errors: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "field": ".".join(str(part) for part in error.get("loc", ()) if part != "body"),
            "message": str(error.get("msg", "")),
        }
        for error in errors
    ]
