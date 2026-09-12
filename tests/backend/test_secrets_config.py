"""Credentials come from one obvious file, and cannot leak by accident."""

import subprocess
from pathlib import Path

import pytest

from backend.aws.clients import credential_kwargs
from backend.config import SecretsFileError, Settings, secrets_file_path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "config" / "secrets" / "config.sample.py"


@pytest.fixture
def secrets_file(tmp_path, monkeypatch):
    def write(body: str) -> Path:
        path = tmp_path / "config.py"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setenv("ML_FACTORY_SECRETS_FILE", str(path))
        return path

    return write


def test_values_are_read_from_the_python_file(secrets_file):
    secrets_file(
        'AWS_REGION = "eu-west-1"\n'
        'ARTIFACT_BUCKET = "s3://from-python-file"\n'
        'AWS_ACCESS_KEY_ID = "AKIAEXAMPLE"\n'
        'AWS_SECRET_ACCESS_KEY = "shh"\n'
    )
    settings = Settings()
    assert settings.aws_region == "eu-west-1"
    assert settings.artifact_bucket == "s3://from-python-file"
    assert settings.has_static_credentials is True


def test_the_prefixed_spelling_works_too(secrets_file):
    secrets_file('ML_FACTORY_AWS_REGION = "ap-south-1"\n')
    assert Settings().aws_region == "ap-south-1"


def test_environment_variables_win_over_the_file(secrets_file, monkeypatch):
    """A deployment can override one value without editing the credentials file."""
    secrets_file('AWS_REGION = "eu-west-1"\n')
    monkeypatch.setenv("ML_FACTORY_AWS_REGION", "us-east-1")
    assert Settings().aws_region == "us-east-1"


def test_a_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_FACTORY_SECRETS_FILE", str(tmp_path / "absent.py"))
    assert Settings().aws_region == "eu-central-1"


def test_a_broken_file_fails_loudly(secrets_file):
    secrets_file("raise ValueError('typo in the credentials file')\n")
    with pytest.raises(SecretsFileError, match="could not be executed"):
        Settings()


def test_unknown_names_and_helpers_are_ignored(secrets_file):
    secrets_file(
        "import os\n"
        "_note = 'internal'\n"
        "def helper():\n"
        "    return 1\n"
        "SOMETHING_ELSE = 'ignored'\n"
        'AWS_REGION = "eu-north-1"\n'
    )
    assert Settings().aws_region == "eu-north-1"


def test_secrets_are_masked_everywhere_they_could_leak(secrets_file):
    secrets_file('AWS_ACCESS_KEY_ID = "AKIAEXAMPLE"\nAWS_SECRET_ACCESS_KEY = "top-secret"\n')
    settings = Settings()
    assert "top-secret" not in repr(settings)
    assert "top-secret" not in str(settings.model_dump())
    assert "top-secret" not in settings.model_dump_json()
    assert settings.aws_secret_access_key.get_secret_value() == "top-secret"


def test_a_half_configured_key_pair_is_reported(secrets_file):
    secrets_file('AWS_ACCESS_KEY_ID = "AKIAEXAMPLE"\n')
    problems = Settings().validate_configuration()
    assert any("must be set together" in problem for problem in problems)


def test_static_keys_reach_boto3(secrets_file):
    secrets_file('AWS_ACCESS_KEY_ID = "AKIAEXAMPLE"\nAWS_SECRET_ACCESS_KEY = "shh"\n')
    kwargs = credential_kwargs(Settings())
    assert kwargs["aws_access_key_id"] == "AKIAEXAMPLE"
    assert kwargs["aws_secret_access_key"] == "shh"
    assert "aws_session_token" not in kwargs


def test_a_profile_is_passed_instead_of_keys(secrets_file):
    secrets_file('AWS_PROFILE = "ml-factory"\n')
    assert credential_kwargs(Settings()) == {"profile_name": "ml-factory"}


def test_no_credentials_configured_leaves_the_default_chain_alone(secrets_file):
    secrets_file('AWS_REGION = "eu-central-1"\n')
    settings = Settings()
    assert credential_kwargs(settings) == {}
    assert "default AWS chain" in settings.credential_source()


def test_the_credential_source_description_never_contains_a_secret(secrets_file):
    secrets_file('AWS_ACCESS_KEY_ID = "AKIAEXAMPLE"\nAWS_SECRET_ACCESS_KEY = "top-secret"\n')
    description = Settings().credential_source()
    assert "top-secret" not in description and "AKIAEXAMPLE" not in description


def test_the_default_location_is_the_documented_one(monkeypatch):
    monkeypatch.delenv("ML_FACTORY_SECRETS_FILE", raising=False)
    assert secrets_file_path() == Path("config/secrets/config.py")


def test_the_sample_is_tracked_and_the_real_file_is_ignored():
    """The sample must stay in Git; the real file must be impossible to add by accident."""
    assert SAMPLE.is_file()
    tracked = subprocess.run(
        ["git", "ls-files", "config/secrets/config.sample.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "config.sample.py" in tracked

    ignored = subprocess.run(
        ["git", "check-ignore", "config/secrets/config.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0, "config/secrets/config.py is not gitignored"


def test_the_sample_contains_no_real_credentials():
    body = SAMPLE.read_text(encoding="utf-8")
    assert "AWS_ACCESS_KEY_ID = None" in body
    assert "AWS_SECRET_ACCESS_KEY = None" in body
    # A real key would start with AKIA followed by capitals; the sample only mentions it
    # inside a comment.
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        assert "AKIA" not in code, f"possible real key in the sample: {line}"


def test_the_sample_loads_and_produces_usable_defaults(monkeypatch):
    """The sample is a working starting point: it parses, and its gaps are reported clearly."""
    monkeypatch.setenv("ML_FACTORY_SECRETS_FILE", str(SAMPLE))
    settings = Settings()
    assert settings.aws_region
    assert credential_kwargs(settings) == {}
    problems = settings.validate_configuration()
    assert any("ARTIFACT_BUCKET" in problem for problem in problems)
