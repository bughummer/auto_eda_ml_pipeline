"""AWS adapters for the control plane."""

from backend.aws.clients import (
    build_client,
    build_dynamodb_table,
    build_session,
    credential_kwargs,
    proxy_definitions,
)

__all__ = [
    "build_client",
    "build_dynamodb_table",
    "build_session",
    "credential_kwargs",
    "proxy_definitions",
]
