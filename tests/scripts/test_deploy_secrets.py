"""config/secrets/deploy.py — deploy-time settings without exporting anything.

Mirrors tests/backend/test_secrets_config.py: same shape of file, same "environment wins,
file is the fallback, None means unset" contract, checked the same way. This is the file
scripts/deploy_aws.py and deploy_aws.sh both read, so a value typed here alone — no `export`
anywhere — is enough to run `make deploy-aws-nocli` or `make deploy-aws`.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
deploy_aws = importlib.import_module("deploy_aws")

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "config" / "secrets" / "deploy.sample.py"

_ALL_RECOGNISED_KEYS = deploy_aws._RECOGNISED_KEYS


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """A deterministic slate: none of these are set, regardless of the ambient shell."""
    for key in _ALL_RECOGNISED_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ML_FACTORY_DEPLOY_SECRETS_FILE", raising=False)


@pytest.fixture
def secrets_file(tmp_path, monkeypatch):
    def write(body: str) -> Path:
        path = tmp_path / "deploy.py"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setenv("ML_FACTORY_DEPLOY_SECRETS_FILE", str(path))
        return path

    return write


def test_values_are_read_from_the_python_file(secrets_file):
    secrets_file('AWS_REGION = "eu-west-1"\nARTIFACT_BUCKET = "s3-bucket-name"\n')

    assert deploy_aws.env("AWS_REGION") == "eu-west-1"
    assert deploy_aws.env("ARTIFACT_BUCKET") == "s3-bucket-name"


def test_environment_variables_win_over_the_file(secrets_file, monkeypatch):
    """A single invocation can override one value without editing the credentials file."""
    secrets_file('AWS_REGION = "eu-west-1"\n')
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    assert deploy_aws.env("AWS_REGION") == "us-east-1"


def test_a_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_FACTORY_DEPLOY_SECRETS_FILE", str(tmp_path / "absent.py"))

    assert deploy_aws.load_deploy_secrets() == {}
    assert deploy_aws.env("AWS_REGION", "eu-central-1") == "eu-central-1"


def test_a_required_variable_missing_from_both_fails_loudly(secrets_file):
    secrets_file("# nothing here\n")

    with pytest.raises(SystemExit):
        deploy_aws.env("ARTIFACT_BUCKET", required=True)


def test_none_means_unset_not_an_empty_value(secrets_file):
    """The sample sets everything to None; that must fall through, not resolve to 'None'."""
    secrets_file("AWS_REGION = None\n")

    assert deploy_aws.load_deploy_secrets().get("AWS_REGION") is None
    assert deploy_aws.env("AWS_REGION", "eu-central-1") == "eu-central-1"


def test_imports_underscored_names_and_helpers_are_not_treated_as_settings(secrets_file):
    """Only ``load_deploy_secrets`` needs to ignore these; ``print_shell_exports`` filters
    everything else by name, tested separately below."""
    secrets_file(
        "import os\n_note = 'internal'\ndef helper():\n    return 1\nAWS_REGION = \"eu-north-1\"\n"
    )

    secrets = deploy_aws.load_deploy_secrets()

    assert secrets["AWS_REGION"] == "eu-north-1"
    assert "os" not in secrets
    assert "_note" not in secrets
    assert "helper" not in secrets


def test_a_stray_name_is_returned_but_harmless(secrets_file):
    """load_deploy_secrets stays permissive, like config.py's loader; only the shell bridge
    (print_shell_exports) needs to filter to recognised settings, and does."""
    secrets_file("SOMETHING_ELSE = 'ignored by everything downstream'\n")

    assert deploy_aws.load_deploy_secrets()["SOMETHING_ELSE"] == "ignored by everything downstream"


def test_a_broken_file_fails_loudly(secrets_file):
    secrets_file("raise ValueError('typo in the deploy settings file')\n")

    with pytest.raises(ValueError, match="typo in the deploy settings file"):
        deploy_aws.load_deploy_secrets()


def test_the_default_location_is_the_documented_one(monkeypatch):
    monkeypatch.delenv("ML_FACTORY_DEPLOY_SECRETS_FILE", raising=False)

    assert deploy_aws._deploy_secrets_file() == deploy_aws.ROOT / "config" / "secrets" / "deploy.py"


# --- the sample -------------------------------------------------------------------


def test_the_sample_is_tracked_and_the_real_file_is_ignored():
    assert SAMPLE.is_file()
    tracked = subprocess.run(
        ["git", "ls-files", "config/secrets/deploy.sample.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "deploy.sample.py" in tracked

    ignored = subprocess.run(
        ["git", "check-ignore", "config/secrets/deploy.py"], cwd=REPO_ROOT, capture_output=True
    )
    assert ignored.returncode == 0, "config/secrets/deploy.py is not gitignored"


def test_the_sample_contains_no_real_credentials():
    body = SAMPLE.read_text(encoding="utf-8")
    assert "AWS_ACCESS_KEY_ID = None" in body
    assert "AWS_SECRET_ACCESS_KEY = None" in body
    for line in body.splitlines():
        code = line.split("#", 1)[0]
        assert "AKIA" not in code, f"possible real key in the sample: {line}"


def test_the_sample_loads_and_every_setting_it_carries_is_recognised(monkeypatch):
    """The sample is a working starting point: it parses, and nothing in it is a stray name."""
    monkeypatch.setenv("ML_FACTORY_DEPLOY_SECRETS_FILE", str(SAMPLE))

    secrets = deploy_aws.load_deploy_secrets()

    assert secrets["AWS_REGION"] == "eu-central-1"
    for name in secrets:
        assert name in _ALL_RECOGNISED_KEYS, f"{name} is in the sample but not a known setting"


def test_every_recognised_key_appears_in_the_sample():
    """A setting added to the script without a line in the sample is invisible to an operator."""
    body = SAMPLE.read_text(encoding="utf-8")
    for key in _ALL_RECOGNISED_KEYS:
        assert f"{key} " in body or f"{key}=" in body, f"{key} is not documented in the sample"


# --- the shell bridge (deploy_aws.sh's `eval "$(... --print-shell-exports)"`) -----


def test_print_shell_exports_emits_every_setting_from_the_file(secrets_file, capsys):
    secrets_file('AWS_REGION = "eu-west-1"\nARTIFACT_BUCKET = "my-bucket"\n')

    deploy_aws.print_shell_exports()

    out = capsys.readouterr().out
    assert "export AWS_REGION=eu-west-1" in out
    assert "export ARTIFACT_BUCKET=my-bucket" in out


def test_print_shell_exports_never_overrides_a_value_already_in_the_environment(
    secrets_file, monkeypatch, capsys
):
    secrets_file('AWS_REGION = "eu-west-1"\n')
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    deploy_aws.print_shell_exports()

    assert "AWS_REGION" not in capsys.readouterr().out


def test_print_shell_exports_skips_unrecognised_names(secrets_file, capsys):
    secrets_file('SOMETHING_ELSE = "not a real setting"\n')

    deploy_aws.print_shell_exports()

    assert capsys.readouterr().out == ""


def test_print_shell_exports_output_is_safe_to_eval_in_a_shell(tmp_path):
    """A value containing shell metacharacters must not escape its own export statement."""
    marker = tmp_path / "injection-marker"
    danger = f"bucket'; touch {marker}; echo pwned"
    (tmp_path / "deploy.py").write_text(f"ARTIFACT_BUCKET = {danger!r}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            'eval "$("$0" scripts/deploy_aws.py --print-shell-exports)" '
            '&& printf "%s" "$ARTIFACT_BUCKET"',
            sys.executable,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "ML_FACTORY_DEPLOY_SECRETS_FILE": str(tmp_path / "deploy.py")},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == danger
    assert not marker.exists(), "the embedded shell command ran instead of staying inert text"


# --- credential_source (printed, never logged with a secret) ----------------------


def test_credential_source_reports_static_keys(secrets_file):
    secrets_file('AWS_ACCESS_KEY_ID = "AKIAEXAMPLE"\nAWS_SECRET_ACCESS_KEY = "shh"\n')

    assert deploy_aws.credential_source() == "static access key"


def test_credential_source_reports_the_default_chain_when_nothing_is_configured(secrets_file):
    secrets_file("AWS_REGION = 'eu-west-1'\n")

    assert "default AWS chain" in deploy_aws.credential_source()


def test_credential_source_names_the_assumed_role(secrets_file):
    secrets_file('DEPLOYER_ROLE_ARN = "arn:aws:iam::123456789012:role/Deployer"\n')

    assert deploy_aws.credential_source().endswith(
        "assuming arn:aws:iam::123456789012:role/Deployer"
    )


def test_credential_source_never_contains_a_secret(secrets_file):
    secrets_file('AWS_ACCESS_KEY_ID = "AKIAEXAMPLE"\nAWS_SECRET_ACCESS_KEY = "top-secret"\n')

    description = deploy_aws.credential_source()

    assert "AKIAEXAMPLE" not in description
    assert "top-secret" not in description
