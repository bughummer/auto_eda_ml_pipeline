VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help install install-backend install-frontend lint format test test-fast build-front dev api front demo openapi clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

install: install-backend install-frontend ## install python and node dependencies

install-backend: ## create the venv and install the python project
	python3.12 -m venv $(VENV)
	$(PIP) install -q --upgrade pip setuptools wheel
	$(PIP) install -q -e ".[backend,models,dictionary,dev]"

install-frontend: ## install frontend dependencies
	cd frontend && npm install --no-audit --no-fund

lint: ## ruff check
	$(VENV)/bin/ruff check ml_engine backend jobs tests scripts

format: ## ruff format + autofix
	$(VENV)/bin/ruff format ml_engine backend jobs tests scripts
	$(VENV)/bin/ruff check --fix ml_engine backend jobs tests scripts

test: ## full pytest suite
	$(VENV)/bin/pytest

test-fast: ## skip the slow end-to-end tests
	$(VENV)/bin/pytest -m "not slow"

build-front: ## typecheck and build the frontend
	cd frontend && npm run build

api: ## run the FastAPI control plane
	$(VENV)/bin/uvicorn backend.main:app --reload --port 8000

front: ## run the Vite dev server
	cd frontend && npm run dev

dev: ## run API and frontend together
	$(MAKE) -j2 api front

demo: ## generate a sample dataset and run a full local experiment
	$(PY) scripts/run_local_demo.py

openapi: ## export the OpenAPI document for the frontend
	$(PY) scripts/export_openapi.py

clean:
	rm -rf .pytest_cache .ruff_cache var/ml-factory frontend/dist
