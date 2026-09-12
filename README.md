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

## Quick start (no AWS required)

```bash
make install                 # python 3.12 venv + npm install
make demo                    # a full experiment end to end against the local adapters
make test                    # 312 tests
make dev                     # API on :8000, UI on :5173
```

`make demo` generates a sample dataset that deliberately contains an identifier, a constant
column, a high-missingness column, a leaked copy of the target and a post-outcome column, then
runs a complete experiment. It prints what the platform found, what it recommended, and the
model comparison that results once those recommendations are acted on.

In local mode (`ML_FACTORY_MODE=local`, the default) the platform runs the *same* job code in
a background thread pool against a directory tree that mirrors the S3 layout. Switching to
`ML_FACTORY_MODE=aws` swaps two adapters — storage and orchestration — and nothing else.

## Configuration

Copy `.env.example` to `.env`. The settings that matter:

| Variable | Meaning |
|---|---|
| `ML_FACTORY_MODE` | `local` (filesystem + in-process runner) or `aws` (S3 + Step Functions + DynamoDB) |
| `ML_FACTORY_ARTIFACT_BUCKET` | `s3://…` root for experiment artifacts |
| `ML_FACTORY_ALLOWED_DATASET_PREFIXES` | dataset allow-list; an empty list denies everything, deliberately |
| `ML_FACTORY_EDA_STATE_MACHINE_ARN` / `…_TRAINING_…` | the two workflows |
| `ML_FACTORY_EXPERIMENTS_TABLE` | DynamoDB table holding experiment records |
| `ML_FACTORY_BEDROCK_ENABLED` / `…_MODEL_ID` | enables the semantic analysis tab |
| `HTTPS_PROXY` / `NO_PROXY` | corporate proxy; applied explicitly to every boto3 client |

`GET /api/v1/health` reports the active mode and lists any configuration that would fail at
runtime, so a misconfigured deployment says so instead of failing on the first experiment.

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
make test          # everything
make test-fast     # skips the slow end-to-end experiments
make lint          # ruff
make build-front   # tsc --noEmit + vite build
```

The suite covers the engine unit by unit, the job entrypoints, the control plane, the
documented architecture boundaries, and a complete local experiment from creation to
comparison — including a deliberately failed model, to prove one failure does not lose the run.
