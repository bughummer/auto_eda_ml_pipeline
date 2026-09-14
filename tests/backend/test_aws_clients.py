"""backend/aws/clients.py — credential resolution, including the assume-role path.

The important property is not "a session gets built" but "the session's credentials are the
assumed role's, and they refresh on their own" — a naive one-shot assume-role call would look
identical in a quick manual test and then start failing every request an hour later. These
tests exercise the real botocore credential provider rather than mocking around it, so a
regression that silently drops the refresh behaviour actually fails.
"""

from unittest.mock import MagicMock, patch

import pytest
from botocore.credentials import DeferredRefreshableCredentials, ReadOnlyCredentials

from backend.aws.clients import build_session, credential_kwargs
from backend.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(artifact_bucket="s3://x", aws_region="eu-central-1", **overrides)


# --- credential_kwargs: unchanged behaviour without a role -----------------------


def test_static_keys_are_passed_through_directly():
    kwargs = credential_kwargs(
        _settings(aws_access_key_id="AKIAEXAMPLE", aws_secret_access_key="secret")
    )
    assert kwargs == {"aws_access_key_id": "AKIAEXAMPLE", "aws_secret_access_key": "secret"}


def test_a_profile_name_is_passed_through_directly():
    kwargs = credential_kwargs(_settings(aws_profile="ml-factory"))
    assert kwargs == {"profile_name": "ml-factory"}


def test_no_configuration_falls_through_to_the_default_chain():
    assert credential_kwargs(_settings()) == {}


# --- build_session: the plain path is untouched when no role is configured -------


def test_without_a_role_arn_the_session_is_built_directly_from_credential_kwargs():
    settings = _settings(aws_access_key_id="AKIAEXAMPLE", aws_secret_access_key="secret")
    fake_boto3 = MagicMock()

    with patch.dict("sys.modules", {"boto3": fake_boto3}):
        build_session(settings)

    fake_boto3.session.Session.assert_called_once_with(
        region_name="eu-central-1", aws_access_key_id="AKIAEXAMPLE", aws_secret_access_key="secret"
    )


# --- build_session: the assume-role path is the real botocore mechanism ----------


def test_a_role_arn_produces_a_session_whose_credentials_are_the_assumed_roles():
    """End to end against real botocore credential objects, with only the network call stubbed."""
    settings = _settings(
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="secret",
        aws_role_arn="arn:aws:iam::123456789012:role/MlFactoryBackendRole",
    )

    sts_client = MagicMock()
    sts_client.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIAROLE",
            "SecretAccessKey": "role-secret",
            "SessionToken": "role-session-token",
            "Expiration": "2030-01-01T00:00:00Z",
        }
    }

    def create_client(service_name, **kwargs):
        assert service_name == "sts"
        return sts_client

    with patch("backend.aws.clients.BotocoreSession.create_client", side_effect=create_client):
        session = build_session(settings)

    credentials = session.get_credentials()
    assert isinstance(credentials, DeferredRefreshableCredentials)
    frozen = credentials.get_frozen_credentials()
    assert frozen == ReadOnlyCredentials("ASIAROLE", "role-secret", "role-session-token")
    sts_client.assume_role.assert_called_once_with(
        RoleArn="arn:aws:iam::123456789012:role/MlFactoryBackendRole",
        RoleSessionName="ml-factory-backend",
    )


def test_the_role_is_assumed_using_the_static_keys_as_the_base_identity():
    """The static keys are the identity that calls AssumeRole — they never reach a client directly."""
    settings = _settings(
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="base-secret",
        aws_role_arn="arn:aws:iam::123456789012:role/MlFactoryBackendRole",
    )
    seen_base_credentials = {}

    def capturing_create_client(self, service_name, **kwargs):
        if service_name == "sts":
            frozen = self.get_credentials().get_frozen_credentials()
            seen_base_credentials["access_key"] = frozen.access_key
            seen_base_credentials["secret_key"] = frozen.secret_key
            client = MagicMock()
            client.assume_role.return_value = {
                "Credentials": {
                    "AccessKeyId": "ASIAROLE",
                    "SecretAccessKey": "role-secret",
                    "SessionToken": "role-token",
                    "Expiration": "2030-01-01T00:00:00Z",
                }
            }
            return client
        raise AssertionError(f"unexpected client requested: {service_name}")

    with patch("backend.aws.clients.BotocoreSession.create_client", capturing_create_client):
        session = build_session(settings)
        session.get_credentials().get_frozen_credentials()

    assert seen_base_credentials == {"access_key": "AKIAEXAMPLE", "secret_key": "base-secret"}


def test_a_role_arn_with_no_static_credentials_assumes_from_the_default_chain():
    """No key pair configured: the base identity is whatever botocore's own chain resolves."""
    settings = _settings(aws_role_arn="arn:aws:iam::123456789012:role/MlFactoryBackendRole")

    sts_client = MagicMock()
    sts_client.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIAROLE",
            "SecretAccessKey": "role-secret",
            "SessionToken": "role-token",
            "Expiration": "2030-01-01T00:00:00Z",
        }
    }

    with patch(
        "backend.aws.clients.BotocoreSession.create_client",
        lambda self, service_name, **kw: sts_client,
    ):
        session = build_session(settings)
        frozen = session.get_credentials().get_frozen_credentials()

    assert frozen.access_key == "ASIAROLE"


def test_credential_source_names_the_role_without_leaking_a_secret():
    settings = _settings(
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="secret",
        aws_role_arn="arn:aws:iam::123456789012:role/MlFactoryBackendRole",
    )

    description = settings.credential_source()

    assert "arn:aws:iam::123456789012:role/MlFactoryBackendRole" in description
    assert "secret" not in description
    assert "AKIAEXAMPLE" not in description


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aws_access_key_id": "AKIAEXAMPLE", "aws_secret_access_key": "secret"},
        {"aws_profile": "ml-factory"},
        {},
    ],
)
def test_credential_source_reports_the_role_on_top_of_every_base_identity(kwargs):
    description = _settings(
        aws_role_arn="arn:aws:iam::123456789012:role/X", **kwargs
    ).credential_source()
    assert description.endswith("assuming arn:aws:iam::123456789012:role/X")
