"""HTTP surface: validation, error envelopes and status reporting."""

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture
def client(settings) -> TestClient:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_reports_the_active_mode(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "local"
    assert body["orchestrator"] == "local"
    assert "logistic_regression" in body["available_models"]


def test_model_catalogue_can_be_filtered(client):
    names = [m["name"] for m in client.get("/api/v1/models").json()]
    assert "xgboost" in names and "elastic_net" in names
    regression = client.get("/api/v1/models", params={"problem_type": "regression"}).json()
    assert {m["name"] for m in regression} == {
        "catboost_regressor",
        "elastic_net",
        "random_forest_regressor",
        "xgboost_regressor",
    }


def test_primary_metrics_are_listed_per_problem_type(client):
    metrics = client.get("/api/v1/metrics", params={"problem_type": "regression"}).json()
    assert "rmse" in metrics
    assert "bias" not in metrics  # signed metrics cannot rank models


def test_create_experiment_returns_immediately(client, dataset_csv):
    response = client.post(
        "/api/v1/experiments",
        json={
            "name": "churn baseline",
            "dataset_uri": str(dataset_csv),
            "target_column": "churned",
        },
        headers={"X-Remote-User": "analyst@corp.example"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["experiment_id"].startswith("exp-")
    assert body["status"] in {"CREATED", "EDA_RUNNING"}


def test_unknown_dataset_is_rejected_with_the_error_envelope(client, tmp_path):
    response = client.post(
        "/api/v1/experiments",
        json={
            "name": "bad",
            "dataset_uri": str(tmp_path / "missing.csv"),
            "target_column": "y",
        },
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert "could not be found" in error["message"]
    assert error["request_id"]


def test_malformed_request_is_rejected(client):
    response = client.post("/api/v1/experiments", json={"name": "", "dataset_uri": "x"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unsupported_dataset_format_is_rejected(client, tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("nope")
    response = client.post(
        "/api/v1/experiments",
        json={"name": "bad", "dataset_uri": str(path), "target_column": "y"},
    )
    assert response.status_code == 422
    assert "format" in response.json()["error"]["message"]


def test_unknown_experiment_returns_not_found(client):
    response = client.get("/api/v1/experiments/exp-does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_artifact_not_ready_is_distinguishable(client, dataset_csv, container):
    experiment_id = client.post(
        "/api/v1/experiments",
        json={"name": "x", "dataset_uri": str(dataset_csv), "target_column": "churned"},
    ).json()["experiment_id"]
    response = client.get(f"/api/v1/experiments/{experiment_id}/comparison")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ARTIFACT_NOT_READY"


def test_training_requires_a_confirmed_feature_selection(client, dataset_csv):
    experiment_id = client.post(
        "/api/v1/experiments",
        json={"name": "x", "dataset_uri": str(dataset_csv), "target_column": "churned"},
    ).json()["experiment_id"]
    client.app.state.container.orchestrator.wait_for_idle(timeout=120)
    response = client.post(f"/api/v1/experiments/{experiment_id}/training", json={})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATE"


def test_reasoning_is_disabled_without_bedrock(client, dataset_csv):
    experiment_id = client.post(
        "/api/v1/experiments",
        json={"name": "x", "dataset_uri": str(dataset_csv), "target_column": "churned"},
    ).json()["experiment_id"]
    response = client.post(f"/api/v1/experiments/{experiment_id}/reasoning", json={})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "FEATURE_DISABLED"


def test_every_response_carries_a_request_id(client):
    response = client.get("/api/v1/health")
    assert response.headers["X-Request-Id"]


def test_experiments_can_be_listed_and_soft_deleted(client, dataset_csv):
    experiment_id = client.post(
        "/api/v1/experiments",
        json={"name": "x", "dataset_uri": str(dataset_csv), "target_column": "churned"},
    ).json()["experiment_id"]
    listing = client.get("/api/v1/experiments").json()
    assert listing["count"] == 1
    assert client.delete(f"/api/v1/experiments/{experiment_id}").status_code == 204
    assert client.get("/api/v1/experiments").json()["count"] == 0
    assert client.get(f"/api/v1/experiments/{experiment_id}").status_code == 404
