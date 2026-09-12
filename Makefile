VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help install install-backend install-frontend lint format test test-fast build-front dev api front openapi deploy-aws docker-build docker-up docker-down docker-logs clean

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

deploy-aws: ## build and push the job image, upload the workflows, deploy the stack
	./scripts/deploy_aws.sh

sample-data: ## generate sample datasets to upload to an approved S3 prefix
	$(PY) scripts/sample_data.py

openapi: ## export the OpenAPI document for the frontend
	$(PY) scripts/export_openapi.py

docker-build: ## build the single-service image (honours http_proxy / https_proxy)
	docker compose build

docker-up: ## start the platform on http://localhost:7570
	docker compose up -d

docker-down: ## stop it
	docker compose down

docker-logs: ## follow the container logs
	docker compose logs -f ml-factory

clean:
	rm -rf .pytest_cache .ruff_cache var frontend/dist
