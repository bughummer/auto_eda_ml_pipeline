"""The single-container setup: one process serves the API and the built UI."""

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app


@pytest.fixture
def built_frontend(tmp_path):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>ML Factory</title>")
    (root / "assets" / "app.js").write_text("console.log('app');")
    return root


def test_api_still_answers_when_the_spa_is_mounted(settings, built_frontend):
    with TestClient(
        create_app(settings.model_copy(update={"static_dir": built_frontend}))
    ) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"


def test_index_is_served_at_the_root(settings, built_frontend):
    with TestClient(
        create_app(settings.model_copy(update={"static_dir": built_frontend}))
    ) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "ML Factory" in response.text


def test_client_side_routes_fall_through_to_the_spa(settings, built_frontend):
    """A page reload on /experiments/exp-1 must not 404."""
    with TestClient(
        create_app(settings.model_copy(update={"static_dir": built_frontend}))
    ) as client:
        response = client.get("/experiments/exp-123")
        assert response.status_code == 200
        assert "ML Factory" in response.text


def test_assets_are_served_as_files(settings, built_frontend):
    with TestClient(
        create_app(settings.model_copy(update={"static_dir": built_frontend}))
    ) as client:
        assert "console.log" in client.get("/assets/app.js").text


def test_path_traversal_is_refused(settings, built_frontend):
    """A path outside the build directory falls back to index.html, never to the filesystem."""
    with TestClient(
        create_app(settings.model_copy(update={"static_dir": built_frontend}))
    ) as client:
        response = client.get("/../../etc/passwd")
        assert response.status_code == 200
        assert "root:" not in response.text


def test_missing_build_leaves_the_api_working(settings, tmp_path):
    with TestClient(
        create_app(settings.model_copy(update={"static_dir": tmp_path / "absent"}))
    ) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/").status_code == 404


def test_no_static_dir_configured_is_the_development_default(settings):
    assert Settings().static_dir is None
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").status_code == 200
