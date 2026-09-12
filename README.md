# ML Factory

Internal platform that accelerates classical-ML POCs and model development — automated EDA,
deterministic data-quality and leakage screening, supervised feature review, classical model
training and comparison, and reproducible experiment artifacts — without removing
data-scientist oversight.

A data scientist supplies an approved S3 dataset and a target column. The platform does the
mechanical work and surfaces what it found. Every decision that changes what the model sees
stays with the person.

* Architecture and its rationale: [`architecture.md`](architecture.md)
* What is built and what is deliberately deferred: [`implementation_plan.md`](implementation_plan.md)
* Rules for working in this repository: [`AGENTS.md`](AGENTS.md)
* AWS deployment: [`infrastructure/README.md`](infrastructure/README.md)

## What it does

1. **EDA** — row/column counts, duplicates, per-column semantic type, missingness, cardinality,
   numeric quantiles, capped categorical frequencies, datetime ranges, target distribution and
   class imbalance.
2. **Deterministic screening** — exact target duplication, inverse binary targets, duplicate
   columns, single features that reproduce the target, identifier-like columns, post-outcome
   naming patterns, and missingness that predicts the outcome. Each finding carries a rule id,
   a severity, an explanation and a recommended action.
3. **Feature review** — the profile, the risk and the recommendation in one table. Nothing is
   removed automatically except the target; the selection that is saved is exactly the one the
   user chose, and it is stored with the experiment.
4. **Training** — logistic regression, elastic net, random forest, XGBoost and CatBoost, for
   binary, multiclass and regression problems. Conservative defaults, no hyperparameter search,
   one job per model so a single failure cannot lose the rest of the experiment.
5. **Comparison** — direction-aware ranking on a chosen primary metric, per-model metrics,
   confusion matrices, feature importance and failure reasons.
6. **Semantic analysis** (optional) — Bedrock reads the deterministic artifacts and any
   attached column documentation and interprets them. It never computes a number and never
   changes a result.

## Non-negotiables

| Rule | Why |
|---|---|
| The LLM never computes a statistic, metric or ranking | numbers must be reproducible and auditable |
| Learned transforms fit on the training fold only | anything else quietly inflates validation scores |
| The browser never calls AWS | one control plane, one audit point, no credentials in a tab |
| Step Functions owns workflow state | the backend and UI report it, they never invent it |
| Nothing is excluded automatically except the target | the platform recommends; the data scientist decides |
| Every artifact is a versioned Pydantic contract | producer and consumer cannot drift apart |

## What runs where

Exactly one thing runs on the corporate server: this container, serving the API and the UI
from one process. Everything else is AWS, and only two AWS data services are involved.

| | Where |
|---|---|
| Web UI + API | the container on the corporate server |
| EDA, preprocessing, training, evaluation | ephemeral SageMaker jobs |
| Stage sequencing, retries, per-model isolation | Step Functions |
| Datasets | S3, read-only, allow-listed prefixes |
| Artifacts **and experiment records** | S3, one prefix per experiment |
| Semantic analysis (optional) | Bedrock |

There is no database. An experiment is a self-contained S3 prefix holding its record, EDA,
configuration, models, comparison and reports — copy the prefix and you have copied the
experiment.

The "corporate server" is just a host running `docker compose`; nothing about the container
assumes on-prem. Running it on an EC2 instance is the natural AWS-native choice — the
CloudFormation stack creates an instance profile for exactly that — and is walked through step
by step in [`infrastructure/README.md`](infrastructure/README.md#running-docker-compose-on-aws),
alongside the account permissions needed to deploy and to run it.

## Deploying

Two steps, in this order. The first is AWS-side and is needed once per release; the second
runs the control plane.

```bash
# 1. Build and push the job image SageMaker runs, upload the workflow definitions, and
#    create or update the stack (S3 + KMS + IAM + the two state machines).
AWS_REGION=eu-central-1 AWS_ACCOUNT_ID=123456789012 \
ARTIFACT_BUCKET=my-ml-factory-artifacts \
APPROVED_DATA_BUCKET=my-approved-data \
DEFINITIONS_BUCKET=my-deploy-bucket \
make deploy-aws

# 2. Put the stack outputs into config/secrets/config.py, then run the control plane.
cp config/secrets/config.sample.py config/secrets/config.py   # fill in the outputs
cp .env.example .env
docker network create dev_network            # once per host, if it does not exist yet
docker compose up -d --build
# http://<host>:7570
```

`make deploy-aws` exists because `docker compose` cannot do that work: compose runs containers
on this host, while SageMaker needs its image in ECR and Step Functions needs its state
machines to exist in AWS. It is one idempotent script (`scripts/deploy_aws.sh`) — re-run it to
ship a new job image.

The job image is a *different* image from the one compose builds: compose builds the control
plane (FastAPI + the React app); `infrastructure/docker/Dockerfile` builds what SageMaker runs.

## The container

One service: FastAPI serves the API and the built React app from the same process, so there is
one container, one port and one origin — no CORS and no separate web server.

| Detail | Value |
|---|---|
| Published port | `7570` on the host, `8520` in the container |
| Network | the external `dev_network`, joined by the service |
| Configuration | `./.env` and `./config/secrets` mounted read-only |
| Health | `curl http://localhost:7570/api/v1/health`, also wired as a container `HEALTHCHECK` |

Behind the corporate proxy, `http_proxy` and `https_proxy` are passed as build arguments (npm
and pip need them at build time) and as environment variables (boto3 needs them at runtime).
`no_proxy` keeps loopback off the proxy so the healthcheck works. The image also writes an apt
proxy file from the same build argument, so the proxy is defined in one place:

```bash
export http_proxy=http://10.0.139.93:8080
export https_proxy=http://10.0.139.93:8080
docker compose up -d --build
```

Nothing else is installed on the host: the image builds the React app itself, so no Node and
no Python venv are needed to run the platform — those are development conveniences only.

## Credentials and configuration

Configuration has one module, `backend/config.py`, and three sources — highest precedence first:

| Source | What belongs there |
|---|---|
| `ML_FACTORY_*` environment variables | per-deployment overrides; what Compose and systemd set |
| `.env` (copy of `.env.example`) | non-secret settings |
| `config/secrets/config.py` | **credentials** — copied from the tracked sample, never committed |

```bash
cp config/secrets/config.sample.py config/secrets/config.py   # then fill it in
```

`config/secrets/config.py` is in `.gitignore`, so `git add .` will not pick it up and
`git status` will not offer it. `config/secrets/config.sample.py` is tracked and carries no
real values; a test asserts both of those facts, and that the sample stays credential-free.

Secrets are held as `SecretStr`, so they are masked in logs, reprs and `model_dump()` and can
only be read by asking for the value explicitly. The credentials file is executed as Python,
so treat it as code: it may contain comments and helpers, and names starting with `_` are
ignored.

**Prefer an instance role.** Leave `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` as `None`. On
EC2, attach the stack's `MlFactoryBackendInstanceProfile` when you launch the host — boto3's
default chain finds it automatically and there is nothing to rotate. Off EC2, mount `~/.aws`
into the container instead. Static keys in `config/secrets/config.py` are the fallback for
neither of those being available, and must be set as a pair — a half-configured pair is
reported by `GET /api/v1/health`, which also states which credential source is in effect
without ever returning a credential.

The settings that matter:

| Variable | Meaning |
|---|---|
| `ML_FACTORY_ARTIFACT_BUCKET` | `s3://…` root for artifacts **and experiment records** |
| `ML_FACTORY_ADDITIONAL_ARTIFACT_ROOTS` | other approved buckets the UI may browse read-only |
| `ML_FACTORY_ALLOWED_DATASET_PREFIXES` | dataset allow-list; an empty list denies everything, deliberately |
| `ML_FACTORY_EDA_STATE_MACHINE_ARN` / `…_TRAINING_…` | the two workflows |
| `ML_FACTORY_BEDROCK_ENABLED` / `…_MODEL_ID` | enables the semantic analysis tab |
| `HTTPS_PROXY` / `NO_PROXY` | corporate proxy; applied explicitly to every boto3 client |

`GET /api/v1/health` reports the active mode and lists any configuration that would fail at
runtime, so a misconfigured deployment says so instead of failing on the first experiment.

## Browsing previous experiments

The experiment list shows every experiment in this platform's artifact bucket — status,
target, best model and score — and opening one shows its full EDA, feature review, training
result and comparison. Experiments persist across restarts and redeploys because the record
lives in the bucket, not in the container.

`ML_FACTORY_ADDITIONAL_ARTIFACT_ROOTS` is empty by default and most deployments should leave
it that way. It exists for the case where you run **more than one** ML Factory — a production
instance and a development one, or a second AWS account — and want one UI to read the other's
results. Listing a bucket there adds a bucket selector to the experiment list; those buckets
are read-only, and new experiments are always written to this platform's own bucket. With no
extra buckets configured, no selector appears.

## Repository layout

```
ml_engine/        deterministic ML core — contracts, IO, profiling, leakage, preprocessing,
                  splitting, models, evaluation, training, reporting, dictionary, reasoning
jobs/             thin entrypoints run inside SageMaker containers
backend/          FastAPI control plane: validate, record, start workflows, serve artifacts
frontend/         React + TypeScript + Vite + Ant Design + TanStack Query
infrastructure/   Step Functions definitions, IAM policies, CloudFormation, the job image
tests/            pytest suites, including the architecture boundary tests
scripts/          local demo, sample data, OpenAPI export
```

Dependency direction is one-way and enforced by a test:
`frontend → backend → ml_engine`, `jobs → ml_engine`, and `ml_engine` depends on nothing above it.

## API

Base path `/api/v1`; full table in [`architecture.md` §6](architecture.md#6-api-contract-v1).
The shape of the flow:

```
POST /experiments                     -> {experiment_id, status}   (returns immediately)
GET  /experiments/{id}                -> status and stage
GET  /experiments/{id}/eda            -> EdaReport
GET  /experiments/{id}/features       -> profile + risk + recommendation per feature
PUT  /experiments/{id}/features       -> persist the user's decision
POST /experiments/{id}/training       -> validate config, freeze it, start the workflow
GET  /experiments/{id}/training-status-> per-model progress
GET  /experiments/{id}/comparison     -> ranked models and the best one
```

Errors always use one envelope: `{"error": {"code", "message", "details", "request_id"}}`.
Stack traces go to the logs, never to the browser.

## Artifacts

Every experiment writes a self-describing set of artifacts under
`s3://<bucket>/ml-factory/experiments/<experiment_id>/` — EDA, leakage findings, the frozen
configuration, the feature decision, the fitted preprocessing pipelines, the materialized
folds, one metadata document per model, the comparison and the summary. The source dataset is
referenced (URI, version id, etag, size), never copied.

Reproducing a run means reading `config/experiment_config.json`: it carries the dataset
identity, the selected and excluded features, the split configuration and seed, the
preprocessing configuration, every hyperparameter and the package versions used.

## Adding a model

Implement `ModelPlugin` in `ml_engine/models/<name>.py`, register it in
`ml_engine/models/registry.py`, and add it to the plugin test. Nothing else changes — not the
state machine, not the API, not the UI. See [`AGENTS.md`](AGENTS.md).

## Testing

```bash
make install       # python 3.12 venv + npm install
make test          # everything
make test-fast     # skips the slow end-to-end experiments
make lint          # ruff
make build-front   # tsc --noEmit + vite build
```

The suite needs no AWS. It substitutes two adapters — an in-memory object store addressed with
real `s3://` URIs, and an orchestrator that runs the same job entrypoints in-process — so the
production code paths, including URI validation, the dataset allow-list and the artifact
layout, are the ones under test. Both live in `tests/support/`; neither ships.

The suite covers the engine unit by unit, the job entrypoints, the control plane, the
documented architecture boundaries, and a complete local experiment from creation to
comparison — including a deliberately failed model, to prove one failure does not lose the run.
