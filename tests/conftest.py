"""Shared fixtures. Every test runs against the local adapters — no AWS, no network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import Settings
from backend.container import AppContainer, build_container
from backend.repositories import ObjectStoreExperimentRepository
from ml_engine.contracts.common import ProblemType
from ml_engine.io import ExperimentLayout
from ml_engine.profiling import profile_dataset
from tests.support.inline_orchestrator import InlineOrchestrator
from tests.support.memory_store import InMemoryObjectStore


@pytest.fixture(autouse=True)
def _isolated_secrets_file(tmp_path_factory, monkeypatch):
    """Tests never read a developer's real config/secrets/config.py."""
    monkeypatch.setenv(
        "ML_FACTORY_SECRETS_FILE", str(tmp_path_factory.mktemp("no-secrets") / "config.py")
    )


ARTIFACT_BUCKET = "s3://ml-factory-test-artifacts"
DATA_BUCKET = "s3://ml-factory-test-data"


@pytest.fixture
def artifact_root() -> str:
    return ARTIFACT_BUCKET


@pytest.fixture
def store() -> InMemoryObjectStore:
    """An object store addressed exactly as S3 is, so URI handling is under test too."""
    return InMemoryObjectStore()


@pytest.fixture
def layout(artifact_root: str) -> ExperimentLayout:
    return ExperimentLayout.for_experiment(artifact_root, "exp-test")


@pytest.fixture
def settings(artifact_root: str) -> Settings:
    """A fully configured platform whose artifact root is a temporary directory.

    The artifact root is the only thing that differs from production: every other component,
    including the repository, is the code that ships.
    """
    return Settings(
        artifact_bucket=artifact_root,
        allowed_dataset_prefixes=[DATA_BUCKET],
        eda_state_machine_arn="arn:aws:states:eu-central-1:000000000000:stateMachine:test-eda",
        training_state_machine_arn="arn:aws:states:eu-central-1:000000000000:stateMachine:test-train",
    )


@pytest.fixture
def container(settings: Settings, store: InMemoryObjectStore) -> AppContainer:
    """The real container, with the storage and orchestration adapters substituted."""
    built = build_container(
        settings,
        store=store,
        repository=ObjectStoreExperimentRepository(store, settings.artifact_root),
        orchestrator=InlineOrchestrator(store),
    )
    yield built
    built.shutdown()


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    """A small dataset carrying every problem the platform is meant to detect."""
    rng = np.random.default_rng(42)
    rows = 400
    tenure = rng.integers(1, 60, rows)
    charges = rng.normal(70, 20, rows)
    contract = rng.choice(["monthly", "annual"], rows, p=[0.6, 0.4])
    risk = 0.05 * charges - 0.04 * tenure + np.where(contract == "monthly", 1.5, -1.0)
    target = (risk + rng.normal(0, 1.0, rows) > 2.0).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(rows)],
            "tenure_months": tenure,
            "monthly_charges": np.round(charges, 2),
            "contract": contract,
            "is_business": rng.choice([True, False], rows),
            "signup_date": pd.date_range("2021-01-01", periods=rows, freq="D"),
            "region_code": "EU",
            "notes": rng.choice(
                [
                    "customer provided a long free text comment about the service quality",
                    "the technician visit was handled professionally and on time",
                    "billing was unclear and the price increase was not communicated",
                ],
                rows,
            ),
            "churn_copy": target,
            "churned": target,
        }
    )
    frame.loc[rng.random(rows) < 0.4, "monthly_charges"] = np.nan
    return frame


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = 300
    area = rng.normal(90, 25, rows).clip(20, 200)
    rooms = np.maximum(1, (area / 30).round())
    district = rng.choice(["centre", "north"], rows)
    price = 1000 * area + 5000 * rooms + rng.normal(0, 15000, rows)
    return pd.DataFrame(
        {
            "area_sqm": np.round(area, 1),
            "rooms": rooms,
            "district": district,
            "price": np.round(price, 2),
        }
    )


@pytest.fixture
def classification_eda(classification_frame: pd.DataFrame):
    return profile_dataset(
        classification_frame,
        experiment_id="exp-test",
        target_column="churned",
        source_uri="memory://classification",
        file_format="csv",
    )


@pytest.fixture
def dataset_csv(
    tmp_path: Path, store: InMemoryObjectStore, classification_frame: pd.DataFrame
) -> str:
    """The classification dataset, uploaded to the approved data bucket."""
    path = tmp_path / "churn.csv"
    classification_frame.to_csv(path, index=False)
    return store.put_file(f"{DATA_BUCKET}/curated/churn.csv", path)


@pytest.fixture
def regression_csv(
    tmp_path: Path, store: InMemoryObjectStore, regression_frame: pd.DataFrame
) -> str:
    path = tmp_path / "prices.csv"
    regression_frame.to_csv(path, index=False)
    return store.put_file(f"{DATA_BUCKET}/curated/prices.csv", path)


@pytest.fixture
def binary_problem() -> ProblemType:
    return ProblemType.BINARY_CLASSIFICATION
