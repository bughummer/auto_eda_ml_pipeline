"""boto3 client construction.

One place builds AWS clients, so the corporate proxy, the retry policy, the region and the
credential source are applied consistently. ``HTTPS_PROXY``/``NO_PROXY`` are honoured
explicitly rather than relied upon.

Credentials come from the standard AWS chain by default — an instance role on the corporate
server, or ``~/.aws``. Static keys are used only when they are configured, which is the
fallback for hosts where no role is available.
"""

import logging
import os
from typing import Any

from backend.config import Settings

LOGGER = logging.getLogger("ml_factory.aws")


def proxy_definitions() -> dict[str, str]:
    """Corporate proxy settings, read from the standard environment variables."""
    proxies: dict[str, str] = {}
    for scheme, variables in (
        ("https", ("HTTPS_PROXY", "https_proxy")),
        ("http", ("HTTP_PROXY", "http_proxy")),
    ):
        for variable in variables:
            value = os.environ.get(variable)
            if value:
                proxies[scheme] = value
                break
    return proxies


def credential_kwargs(settings: Settings) -> dict[str, str]:
    """Explicit credentials for boto3, or nothing at all so the default chain applies."""
    if settings.has_static_credentials:
        kwargs = {
            "aws_access_key_id": settings.aws_access_key_id.get_secret_value(),
            "aws_secret_access_key": settings.aws_secret_access_key.get_secret_value(),
        }
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token.get_secret_value()
        return kwargs
    if settings.aws_profile:
        return {"profile_name": settings.aws_profile}
    return {}


def build_session(settings: Settings) -> Any:
    """A boto3 session configured with the region and whichever credential source applies."""
    import boto3

    LOGGER.info("AWS credentials: %s", settings.credential_source())
    return boto3.session.Session(region_name=settings.aws_region, **credential_kwargs(settings))


def botocore_config(settings: Settings, **overrides: Any) -> Any:
    from botocore.config import Config

    proxies = proxy_definitions()
    return Config(
        region_name=settings.aws_region,
        retries={"max_attempts": 5, "mode": "standard"},
        proxies=proxies or None,
        proxies_config={"proxy_use_forwarding_for_https": True} if proxies else None,
        user_agent_extra="ml-factory/1.0",
        **overrides,
    )


def build_client(service: str, settings: Settings, **overrides: Any) -> Any:
    """Create a proxy-aware, retry-configured, credential-aware boto3 client."""
    return build_session(settings).client(service, config=botocore_config(settings, **overrides))


def build_dynamodb_table(table_name: str, settings: Settings) -> Any:
    resource = build_session(settings).resource("dynamodb", config=botocore_config(settings))
    return resource.Table(table_name)
