"""Control-plane configuration.

Everything environment-specific is here. Nothing else in the backend reads ``os.environ``.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml_engine.io.uri import uri_matches_prefix


class DeploymentMode(StrEnum):
    LOCAL = "local"
    AWS = "aws"


class Settings(BaseSettings):
    """Configuration for the FastAPI control plane."""

    model_config = SettingsConfigDict(
        env_prefix="ML_FACTORY_", env_file=".env", extra="ignore", case_sensitive=False
    )

    mode: DeploymentMode = DeploymentMode.LOCAL
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

    # --- local mode -------------------------------------------------------
    local_root: Path = Path("./var/ml-factory")
    local_max_workers: int = Field(default=2, ge=1, le=16)

    # --- aws mode ---------------------------------------------------------
    aws_region: str = "eu-central-1"
    artifact_bucket: str = ""
    allowed_dataset_prefixes: list[str] = Field(default_factory=list)
    eda_state_machine_arn: str = ""
    training_state_machine_arn: str = ""
    experiments_table: str = "ml-factory-experiments"
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

    @property
    def is_local(self) -> bool:
        return self.mode is DeploymentMode.LOCAL

    @property
    def artifact_root(self) -> str:
        """Where experiment artifacts live, in whichever mode is active."""
        if self.is_local:
            return str(self.local_root.expanduser().resolve())
        return self.artifact_bucket

    def dataset_uri_allowed(self, uri: str) -> bool:
        """Allow-list check performed before any AWS call is made.

        In local mode any readable path is allowed; in AWS mode the URI must sit under a
        configured approved prefix. An empty allow-list denies everything, deliberately.
        """
        if self.is_local:
            return True
        return any(uri_matches_prefix(uri, prefix) for prefix in self.allowed_dataset_prefixes)

    def validate_for_mode(self) -> list[str]:
        """Configuration problems that would make this mode unusable at runtime."""
        problems: list[str] = []
        if self.is_local:
            return problems
        if not self.artifact_bucket.startswith("s3://"):
            problems.append("ML_FACTORY_ARTIFACT_BUCKET must be an s3:// URI in aws mode")
        if not self.allowed_dataset_prefixes:
            problems.append(
                "ML_FACTORY_ALLOWED_DATASET_PREFIXES is empty; every dataset would be rejected"
            )
        if not self.eda_state_machine_arn:
            problems.append("ML_FACTORY_EDA_STATE_MACHINE_ARN is required in aws mode")
        if not self.training_state_machine_arn:
            problems.append("ML_FACTORY_TRAINING_STATE_MACHINE_ARN is required in aws mode")
        if self.bedrock_enabled and not self.bedrock_model_id:
            problems.append("ML_FACTORY_BEDROCK_MODEL_ID is required when Bedrock is enabled")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
