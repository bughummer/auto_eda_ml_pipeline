"""boto3 client construction.

One place builds AWS clients, so the corporate proxy, the retry policy, the region and the
credential source are applied consistently. ``HTTPS_PROXY``/``NO_PROXY`` are honoured
explicitly rather than relied upon.

Credentials come from the standard AWS chain by default — normally ``~/.aws`` on the
corporate server, mounted into the container. Static keys are used only when they are
configured, which is the fallback for hosts with no profile. If ``aws_role_arn`` is set,
whichever of those resolves is used only to call ``sts:AssumeRole``, and every client is built
from that role's temporary credentials instead — refreshed automatically before they expire, so
this keeps working for the life of a long-running process rather than failing an hour after
startup.
"""

import logging
import os
from typing import Any

from botocore.session import Session as BotocoreSession

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
    base_kwargs = credential_kwargs(settings)
    if not settings.aws_role_arn:
        return boto3.session.Session(region_name=settings.aws_region, **base_kwargs)
    return _assumed_role_session(settings.aws_role_arn, settings.aws_region, base_kwargs)


def _assumed_role_session(role_arn: str, region: str, base_kwargs: dict[str, str]) -> Any:
    """A session whose credentials are ``role_arn``'s, re-assumed automatically before expiry.

    This is the same credential provider botocore builds for you from a ``role_arn`` profile in
    ``~/.aws/config`` — ``AssumeRoleCredentialFetcher`` plus ``DeferredRefreshableCredentials``
    — constructed by hand so the role can be configured from ``config/secrets/config.py``
    instead of a second file on disk. ``botocore_session._credentials`` has no public setter for
    an already-built ``Credentials`` object; assigning it directly is what botocore's own
    profile-based provider does internally, and is the documented recipe for this exact case.
    """
    import boto3
    from botocore.credentials import AssumeRoleCredentialFetcher, DeferredRefreshableCredentials

    base = BotocoreSession()
    base.set_config_variable("region", region)
    if "aws_access_key_id" in base_kwargs:
        base.set_credentials(
            base_kwargs["aws_access_key_id"],
            base_kwargs["aws_secret_access_key"],
            base_kwargs.get("aws_session_token"),
        )
    elif "profile_name" in base_kwargs:
        base.set_config_variable("profile", base_kwargs["profile_name"])
    # else: no explicit base credentials — botocore's own default chain resolves the identity
    # that assumes the role, exactly as it would for any other AWS call in this process.

    fetcher = AssumeRoleCredentialFetcher(
        client_creator=base.create_client,
        source_credentials=base.get_credentials(),
        role_arn=role_arn,
        extra_args={"RoleSessionName": "ml-factory-backend"},
    )
    assumed = BotocoreSession()
    assumed.set_config_variable("region", region)
    assumed._credentials = DeferredRefreshableCredentials(
        method="assume-role", refresh_using=fetcher.fetch_credentials
    )
    return boto3.Session(botocore_session=assumed)


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
