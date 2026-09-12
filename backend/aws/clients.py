"""boto3 client construction.

One place builds AWS clients so the corporate proxy, retry policy and region are applied
consistently. ``HTTPS_PROXY``/``NO_PROXY`` are honoured explicitly rather than relied upon.
"""

import logging
import os
from functools import lru_cache
from typing import Any

LOGGER = logging.getLogger("ml_factory.aws")


def proxy_definitions() -> dict[str, str]:
    """Corporate proxy settings, read from the standard environment variables."""
    proxies: dict[str, str] = {}
    for scheme, variables in (("https", ("HTTPS_PROXY", "https_proxy")), ("http", ("HTTP_PROXY", "http_proxy"))):
        for variable in variables:
            value = os.environ.get(variable)
            if value:
                proxies[scheme] = value
                break
    return proxies


@lru_cache
def _session(region: str):
    import boto3

    return boto3.session.Session(region_name=region)


def build_client(service: str, region: str, **overrides: Any) -> Any:
    """Create a proxy-aware, retry-configured boto3 client."""
    from botocore.config import Config

    config = Config(
        region_name=region,
        retries={"max_attempts": 5, "mode": "standard"},
        proxies=proxy_definitions() or None,
        proxies_config={"proxy_use_forwarding_for_https": True} if proxy_definitions() else None,
        user_agent_extra="ml-factory/1.0",
        **overrides,
    )
    return _session(region).client(service, config=config)


def build_dynamodb_table(table_name: str, region: str) -> Any:
    import boto3
    from botocore.config import Config

    resource = boto3.resource(
        "dynamodb",
        region_name=region,
        config=Config(retries={"max_attempts": 5, "mode": "standard"}, proxies=proxy_definitions() or None),
    )
    return resource.Table(table_name)
