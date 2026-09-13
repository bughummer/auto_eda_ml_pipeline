"""Control-plane configuration.

Everything environment-specific is here. Nothing else in the backend reads ``os.environ``.

Three sources, highest precedence first:

1. environment variables (``ML_FACTORY_*``) — what the container and systemd unit set
2. the ``.env`` file next to the application
3. ``config/secrets/config.py`` — a plain Python file for credentials, kept out of Git

The Python file exists so an operator has one obvious place to type credentials. Environment
variables still win over it, so a deployment can override a single value without editing the
file. Secret values are held as ``SecretStr`` so they cannot be printed, logged or serialized
by accident.
"""

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ml_engine.io.uri import uri_matches_prefix

LOGGER = logging.getLogger("ml_factory.config")

#: Where the operator types credentials. Tracked sample: ``config/secrets/config.sample.py``.
SECRETS_FILE = Path("config/secrets/config.py")
SECRETS_FILE_ENV_VAR = "ML_FACTORY_SECRETS_FILE"


class SecretsFileError(RuntimeError):
    """Raised when the secrets file exists but cannot be loaded."""


def secrets_file_path() -> Path:
    """The secrets file location, overridable for containers and tests."""
    return Path(os.environ.get(SECRETS_FILE_ENV_VAR, str(SECRETS_FILE)))


class PythonSecretsSource(PydanticBaseSettingsSource):
    """Reads settings from a plain Python file.

    Module-level names are matched case-insensitively, with or without the ``ML_FACTORY_``
    prefix, so both ``AWS_SECRET_ACCESS_KEY`` and ``ML_FACTORY_AWS_SECRET_ACCESS_KEY`` work.
    Names starting with an underscore are ignored, as are imported modules and callables, so
    the file can contain helpers without them leaking into the configuration.
    """

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self.path = path

    def get_field_value(self, _field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Required by the source interface; the file is read in one pass by ``__call__``.

        pydantic-settings calls this positionally, so the unused first argument is named with
        a leading underscore rather than suppressed.
        """
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        values = _load_python_settings(self.path)
        known = set(self.settings_cls.model_fields)
        prefix = str(self.config.get("env_prefix", "")).lower()
        resolved: dict[str, Any] = {}
        for raw_name, value in values.items():
            name = raw_name.lower()
            if prefix and name.startswith(prefix):
                name = name[len(prefix) :]
            if name in known:
                resolved[name] = value
        if resolved:
            LOGGER.info(
                "Loaded %d setting(s) from %s: %s",
                len(resolved),
                self.path,
                ", ".join(sorted(resolved)),  # names only — never the values
            )
        return resolved


def _load_python_settings(path: Path) -> dict[str, Any]:
    """Import the secrets file by path and return its public module-level constants."""
    import types

    spec = importlib.util.spec_from_file_location("ml_factory_secrets", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SecretsFileError(f"Could not load the secrets file at {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SecretsFileError(f"The secrets file at {path} could not be executed: {exc}") from exc
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType)
        and not callable(value)
    }


class Settings(BaseSettings):
    """Configuration for the FastAPI control plane."""

    model_config = SettingsConfigDict(
        env_prefix="ML_FACTORY_", env_file=".env", extra="ignore", case_sensitive=False
    )

    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    static_dir: Path | None = Field(
        default=None,
        description=(
            "Directory holding the built frontend. Set in the container image so one service "
            "serves both the API and the UI; unset during development, where Vite serves the UI."
        ),
    )

    # --- aws credentials --------------------------------------------------
    # Leave these empty to use the standard AWS chain (normally ~/.aws mounted into the
    # container), which is the preferred option: nothing to rotate and nothing to leak.
    # Set them only where no profile is available.
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    aws_session_token: SecretStr | None = None
    aws_profile: str | None = None

    # --- aws resources ----------------------------------------------------
    aws_region: str = "eu-central-1"
    artifact_bucket: str = ""
    allowed_dataset_prefixes: list[str] = Field(default_factory=list)
    additional_artifact_roots: list[str] = Field(
        default_factory=list,
        description=(
            "Extra artifact buckets the UI may browse read-only, for experiments produced by "
            "another environment. New experiments are always written to artifact_bucket."
        ),
    )
    eda_state_machine_arn: str = ""
    training_state_machine_arn: str = ""
    kms_key_id: str = ""

    # --- bedrock ----------------------------------------------------------
    bedrock_enabled: bool = False
    bedrock_model_id: str = ""
    bedrock_max_tokens: int = Field(default=4096, ge=256, le=32_000)

    # --- limits -----------------------------------------------------------
    max_dictionary_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    experiment_list_limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("artifact_bucket")
    @classmethod
    def _normalize_bucket(cls, value: str) -> str:
        return value.rstrip("/")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment first, then .env, then the Python secrets file, then the defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            PythonSecretsSource(settings_cls, secrets_file_path()),
            file_secret_settings,
        )

    @property
    def has_static_credentials(self) -> bool:
        """True when an access key pair was supplied instead of relying on a role."""
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    def credential_source(self) -> str:
        """How AWS credentials will be obtained. Safe to log — no values."""
        if self.has_static_credentials:
            return "static access key from configuration"
        if self.aws_profile:
            return f"shared profile {self.aws_profile!r}"
        return "default AWS chain (~/.aws, or an attached role)"

    @property
    def artifact_root(self) -> str:
        """Where new experiments are written."""
        return self.artifact_bucket

    @property
    def artifact_roots(self) -> list[str]:
        """Every artifact root the platform may read, the writable one first."""
        roots = [self.artifact_bucket] if self.artifact_bucket else []
        roots.extend(root for root in self.additional_artifact_roots if root not in roots)
        return roots

    def artifact_root_allowed(self, root: str) -> bool:
        """Guards the artifact browser: only configured roots may be read."""
        candidate = root.rstrip("/")
        return any(candidate == configured.rstrip("/") for configured in self.artifact_roots)

    def dataset_uri_allowed(self, uri: str) -> bool:
        """Allow-list check performed before any AWS call is made.

        An empty allow-list denies everything, deliberately.
        """
        return any(uri_matches_prefix(uri, prefix) for prefix in self.allowed_dataset_prefixes)

    def validate_configuration(self) -> list[str]:
        """Configuration problems that would make the platform unusable at runtime."""
        problems: list[str] = []
        if bool(self.aws_access_key_id) != bool(self.aws_secret_access_key):
            problems.append(
                "ML_FACTORY_AWS_ACCESS_KEY_ID and ML_FACTORY_AWS_SECRET_ACCESS_KEY must be set "
                "together, or both left empty to use the default AWS chain"
            )
        if not self.artifact_bucket.startswith("s3://"):
            problems.append("ML_FACTORY_ARTIFACT_BUCKET must be an s3:// URI")
        if not self.allowed_dataset_prefixes:
            problems.append(
                "ML_FACTORY_ALLOWED_DATASET_PREFIXES is empty; every dataset would be rejected"
            )
        if not self.eda_state_machine_arn:
            problems.append("ML_FACTORY_EDA_STATE_MACHINE_ARN is required")
        if not self.training_state_machine_arn:
            problems.append("ML_FACTORY_TRAINING_STATE_MACHINE_ARN is required")
        if self.bedrock_enabled and not self.bedrock_model_id:
            problems.append("ML_FACTORY_BEDROCK_MODEL_ID is required when Bedrock is enabled")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
