# ML Factory — Architecture

Status: authoritative. Version 1.0 (first usable platform version).

This document describes the **current intended architecture**. It is not immutable: when
implementation reveals a materially better design, the change is recorded in
[Architecture decisions](#12-architecture-decisions) and this document, the contracts, the
implementation, the tests and the documentation are updated together.

---

## 1. Purpose

Internal enterprise platform that accelerates classical-ML POC and model-development work
without removing data-scientist oversight.

A data scientist points the platform at an approved S3 dataset and a target column. The
platform performs deterministic EDA, deterministic data-quality and leakage screening,
supervised feature review, deterministic preprocessing, training of several classical models,
evaluation, comparison, and — only afterwards — LLM-based *semantic* reasoning over those
deterministic results.

Non-goals for version 1.0: deep learning, time-series/forecasting, NLP/vectorization,
real-time inference serving, automated model deployment, hyperparameter search.

## 2. Principles

1. **Deterministic first, semantic second.** Every number the user sees is computed by
   pandas / scikit-learn / the model libraries. The LLM never computes a statistic or a
   metric; it reads deterministic artifacts and produces prose and structured judgements.
2. **The control plane never does heavy compute.** FastAPI validates, records, launches and
   reads artifacts. All data touching compute is ephemeral AWS compute.
3. **One authoritative workflow engine.** Step Functions owns stage sequencing and retries.
   The backend and the frontend never invent workflow state.
4. **The browser never touches AWS.** React talks only to FastAPI. No AWS credentials,
   no presigned-URL-driven logic, no direct S3 reads from the browser in v1.0.
5. **Everything reproducible.** Every experiment writes an immutable, self-describing set of
   JSON artifacts under a predictable S3 prefix, including dataset identity, seeds, package
   versions and hyperparameters.
6. **Suspicion is surfaced, not enforced.** Deterministic checks recommend; the data
   scientist decides. Only the target column is excluded from features automatically.
7. **Vertical slices over scaffolding.** Each phase delivers a path that works end to end.

## 3. System context

```
 ┌──────────────┐   HTTPS    ┌──────────────────────┐
 │   Browser    │──────────▶ │  FastAPI control     │
 │ React + AntD │ ◀──────────│  plane (corp server) │
 └──────────────┘   JSON     └──────────┬───────────┘
                                        │ boto3 (corporate proxy)
                        ┌───────────────┼───────────────┬───────────────┐
                        ▼               ▼               ▼               ▼
                 ┌────────────┐  ┌─────────────┐  ┌──────────┐   ┌──────────┐
                 │  Step      │  │  DynamoDB   │  │    S3    │   │ Bedrock  │
                 │  Functions │  │ experiments │  │artifacts │   │(reasoning)│
                 └─────┬──────┘  └─────────────┘  └────┬─────┘   └──────────┘
                       │ ephemeral jobs                │
        ┌──────────────┼───────────────────┐           │
        ▼              ▼                   ▼           │
 ┌─────────────┐ ┌─────────────┐   ┌──────────────┐    │
 │ SageMaker   │ │ SageMaker   │   │ SageMaker    │────┘
 │ Processing  │ │ Training    │   │ Processing   │
 │ (profiling, │ │ (1 job per  │   │ (evaluation, │
 │ preprocess) │ │  model)     │   │  comparison) │
 └─────────────┘ └─────────────┘   └──────────────┘
```

The React app and the FastAPI app run on a permanently available corporate server
(systemd / a single container behind the corporate reverse proxy). No ECS, no Fargate, no
Kubernetes, no always-on AWS ML compute, no SageMaker endpoints.

## 4. Repository layout

Monorepo. Three Python import roots plus the frontend:

```
ml_engine/        deterministic ML core. Pure Python. NO aws, NO fastapi imports.
  contracts/      Pydantic v2 models — the single source of truth for every artifact schema
  io/             dataset loading (csv/parquet), object store abstraction, S3 URI handling
  profiling/      EDA computation
  leakage/        deterministic leakage & feature-risk rules
  preprocessing/  ColumnTransformer construction, fit/transform, persistence
  splitting/      train/validation split strategies
  models/         model plugins (base + registry + implementations)
  evaluation/     metric computation per problem type
  training/       single-model training use case (plugin + fitted preprocessing + metrics)
  reporting/      comparison + experiment summary assembly
  dictionary/     data dictionary parsing & normalization (JSON/CSV/XLSX)
  reasoning/      Bedrock prompt construction + strict response validation (client injected)

jobs/             thin entrypoints executed inside SageMaker containers
  _common/        argument parsing, artifact IO, logging, failure reporting
  profiling/      Processing job: EDA + deterministic leakage
  preprocessing/  Processing job: validation, split, fit preprocessing, materialize folds
  training/       Training job: one model, one job
  evaluation/     Processing job: aggregation, comparison, experiment summary

backend/          FastAPI control plane
  api/routes/     HTTP surface
  schemas/        request/response models (API-only; artifact schemas come from ml_engine)
  services/       use cases (experiment lifecycle, artifacts, dictionary, reasoning)
  repositories/   experiment record persistence (DynamoDB | in-memory)
  orchestration/  ExperimentOrchestrator protocol: Step Functions | local runner
  aws/            boto3 session/client construction, proxy config, S3/SFN/DDB/Bedrock adapters

frontend/         React 18 + TypeScript + Vite + Ant Design + TanStack Query
infrastructure/   Step Functions ASL, IAM policies, CloudFormation, container images
tests/            pytest suites mirroring the packages
scripts/          developer utilities (OpenAPI export, sample data generation)
```

Dependency direction is strictly one-way:

```
frontend → backend → { ml_engine.contracts, aws adapters }
jobs     → ml_engine
backend  → ml_engine (contracts + local runner only; never model training inline)
ml_engine→ (pandas, numpy, sklearn, model libs) only
```

`ml_engine` must never import `fastapi`, `backend` or `jobs`. It must not import `boto3`
either, with exactly one documented exception: `ml_engine/io/s3.py`, which imports the SDK
lazily so that the control plane and the jobs share one S3 adapter instead of growing two
(AD-8). The Bedrock adapter needs no exception — its client is injected. All of this is
enforced by `tests/test_architecture_boundaries.py`, as is the rule that artifact paths are
defined only in `ml_engine/io/layout.py`.

## 5. Experiment lifecycle and state model

```
CREATED
  └─▶ EDA_RUNNING ──▶ EDA_COMPLETED ──▶ FEATURE_REVIEW
                                             │ (user confirms features + training config)
                                             ▼
                                      READY_FOR_TRAINING
                                             ▼
                                        PREPARING            (preprocessing job)
                                             ▼
                                         TRAINING            (N parallel training jobs)
                                             ▼
                                        EVALUATING           (comparison job)
                                             ▼
                             COMPLETED | COMPLETED_WITH_WARNINGS
any stage ─▶ FAILED
```

Rules:

* The backend writes `CREATED` and `READY_FOR_TRAINING` (user-driven transitions).
* Step Functions writes every execution-driven transition via the DynamoDB SDK integration.
* `COMPLETED_WITH_WARNINGS` is used when the experiment produced a usable comparison but at
  least one model failed, or an evaluation/warning threshold was crossed.
* The frontend renders status; it never derives or guesses it.
* `current_stage` is a free-form human-readable stage label; `status` is the enum above.

State ownership summary:

| Data | Owner | Store |
|---|---|---|
| Experiment record (status, stage, dataset, target, best model) | Backend + Step Functions | DynamoDB |
| Workflow execution position, retries | Step Functions | Step Functions |
| All ML artifacts (EDA, config, models, metrics, reports) | Jobs | S3 |
| Nothing | Frontend | — |

## 6. API contract (v1)

Base path `/api/v1`. All errors use the shared envelope in §10.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + dependency mode report |
| POST | `/experiments` | create experiment, start EDA. Returns `{experiment_id, status}` immediately |
| GET | `/experiments` | list experiment records (paged) |
| GET | `/experiments/{id}` | experiment record incl. status, stage, best model |
| DELETE | `/experiments/{id}` | soft-delete the record (artifacts retained) |
| GET | `/experiments/{id}/eda` | `EdaReport` artifact |
| GET | `/experiments/{id}/leakage` | `LeakageReport` artifact |
| GET | `/experiments/{id}/features` | feature review view = columns × profile × leakage × recommendation |
| PUT | `/experiments/{id}/features` | persist selected/excluded feature list |
| GET | `/experiments/{id}/training-config` | current or default `TrainingConfig` |
| POST | `/experiments/{id}/training` | validate config, persist, start training workflow |
| GET | `/experiments/{id}/training-status` | per-model status list (from the workflow/artifacts) |
| GET | `/experiments/{id}/comparison` | `ComparisonReport` |
| GET | `/experiments/{id}/models/{model}` | `ModelMetadata` incl. metrics + importance |
| GET | `/experiments/{id}/summary` | `ExperimentSummary` |
| POST | `/experiments/{id}/data-dictionary` | upload JSON/CSV/XLSX dictionary → normalized schema |
| GET | `/experiments/{id}/data-dictionary` | normalized `DataDictionary` |
| POST | `/experiments/{id}/reasoning` | run Bedrock semantic analysis over existing artifacts |
| GET | `/experiments/{id}/reasoning` | latest `ReasoningReport` |
| GET | `/models` | model plugin catalogue (name, problem types, defaults) |

Requests and responses are Pydantic v2 models. Artifact-shaped responses reuse the exact
`ml_engine.contracts` models so a schema can never drift between producer and consumer.

## 7. S3 artifact layout

```
s3://<artifact-bucket>/ml-factory/experiments/<experiment_id>/
  eda/eda.json                          EdaReport
  eda/leakage.json                      LeakageReport
  config/experiment_config.json         ExperimentConfig (frozen at training start)
  config/selected_features.json         FeatureSelection
  config/data_dictionary.json           DataDictionary (optional)
  validation/validation.json            PreparationReport (split sizes, dropped cols, warnings)
  preprocessing/preprocessor_<strategy>.joblib   fitted sklearn pipeline, one per strategy
  preprocessing/preprocessing.json      PreprocessingMetadata (columns per transformer, versions)
  datasets/train.parquet                materialized training fold (transformed-ready raw slice)
  datasets/validation.parquet           materialized validation fold
  models/<model_name>/model.joblib      serialized fitted model
  models/<model_name>/metadata.json     ModelMetadata (metrics, importance, hyperparams, versions)
  models/<model_name>/failure.json      present only when that model failed
  comparison/comparison.json            ComparisonReport
  report_data/experiment_summary.json   ExperimentSummary (everything needed by reporting)
  reasoning/reasoning.json              ReasoningReport (Bedrock output, validated)
  reports/                              rendered narrative reports (markdown/HTML)
```

The source dataset is **never copied**. Identity is captured as
`{uri, version_id, etag, size_bytes, last_modified}` in the experiment config.

## 8. Compute topology

| Stage | Compute | Entrypoint | Default instance |
|---|---|---|---|
| Profiling (EDA + leakage) | SageMaker Processing | `jobs/profiling/main.py` | `ml.m5.large` |
| Preparation (validate, split, fit preprocessing) | SageMaker Processing | `jobs/preprocessing/main.py` | `ml.m5.large` |
| Training (per model) | SageMaker Training | `jobs/training/main.py` | `ml.m5.large` |
| Evaluation / comparison | SageMaker Processing | `jobs/evaluation/main.py` | `ml.m5.large` |

All CPU by default; instance type and volume size are configuration, not code. Every job is
ephemeral — it starts, writes artifacts to S3, and terminates.

Step Functions state machine (`infrastructure/stepfunctions/experiment_state_machine.asl.json`):

```
StartAt: MarkEdaRunning (DynamoDB UpdateItem)
  → RunProfiling (SageMaker CreateProcessingJob.sync)
  → MarkEdaCompleted → wait for the user (execution ends; feature review is human time)

Training execution (second state machine execution, started by POST /training):
StartAt: MarkPreparing
  → RunPreparation (Processing .sync)
  → MarkTraining
  → TrainModels (Map over config.models, MaxConcurrency configurable,
                 per-branch Catch → writes models/<name>/failure.json, branch result = FAILED)
  → MarkEvaluating
  → RunEvaluation (Processing .sync; the job writes the terminal state, see AD-9)
Catch (any) → MarkFailed
```

The model list is a `Map` input, so adding a model plugin requires **no** state-machine change.

## 9. Model plugin architecture

`ml_engine/models/base.py` defines `ModelPlugin`:

| Member | Meaning |
|---|---|
| `name` | stable identifier used in S3 paths, API and UI (`logistic_regression`, …) |
| `display_name` | human label |
| `supported_problem_types` | subset of {binary, multiclass, regression} |
| `requires_dense_numeric` | if true the preprocessing pipeline must one-hot + impute + scale |
| `supports_native_categorical` | CatBoost-style native handling |
| `supports_class_weighting` | whether `class_weighting=auto` is honoured |
| `default_params(problem_type, context)` | conservative defaults, given row/class counts |
| `build(problem_type, params, context)` | returns an unfitted estimator |
| `fit(estimator, X, y, context)` | training (allows per-library specifics) |
| `predict` / `predict_proba` | inference |
| `feature_importance(estimator, feature_names)` | normalized `FeatureImportance` |
| `library_versions()` | `{library: version}` for the metadata artifact |

Plugins self-register in `ml_engine/models/registry.py`. Orchestration code (jobs, backend,
state machine) contains **no** model-specific branching. Optional dependencies (XGBoost,
CatBoost) register only when importable, and the API catalogue reports availability.

## 10. Error handling

Single error envelope for the whole API:

```json
{ "error": { "code": "EXPERIMENT_NOT_FOUND", "message": "…", "details": {…},
             "request_id": "…" } }
```

* `AppError` hierarchy in `backend/errors.py` maps to HTTP status codes
  (`ValidationError`→422, `NotFoundError`→404, `ConflictError`→409, `UpstreamError`→502,
  `ConfigurationError`→500). Unhandled exceptions become `INTERNAL_ERROR` with a request id;
  stack traces go to logs only, never to the browser.
* Job failures write `failure.json` next to the artifact they were producing and exit non-zero.
  Step Functions catches, records the cause on the experiment record, and (for model training)
  isolates the failure to that model branch.
* Metric computation failures never abort an experiment: the failing metric is omitted and a
  `Warning` with category `metric` is attached.
* The frontend shows `message` and, for developers, `code`; never a stack trace.

## 11. Security

* IAM: three roles — `MlFactoryBackendRole` (control plane: `states:StartExecution`,
  `states:DescribeExecution`, DynamoDB CRUD on one table, `s3:GetObject/PutObject` limited to
  the artifact prefix, `s3:ListBucket`/`GetObject` on approved dataset buckets,
  `bedrock:InvokeModel` on allow-listed model ids),
  `MlFactoryWorkflowRole` (Step Functions: create/describe SageMaker jobs, `iam:PassRole` to
  the job role, DynamoDB UpdateItem on the experiments table),
  `MlFactoryJobRole` (SageMaker containers: read approved dataset prefixes, read/write the
  experiment artifact prefix, CloudWatch Logs). Policies in `infrastructure/iam/`.
* KMS: artifact bucket and DynamoDB table use a customer-managed key; the job role and the
  backend role get `kms:Decrypt`/`GenerateDataKey` on that key only.
* Dataset allow-list: the backend rejects any dataset URI whose bucket/prefix is not in
  `ML_FACTORY_ALLOWED_DATASET_PREFIXES`. This is enforced before any AWS call.
* No credentials in the repository. The corporate proxy is configured through
  `HTTPS_PROXY`/`NO_PROXY` and honoured by an explicit botocore `Config`.
* Bedrock input is assembled from deterministic artifacts and user-supplied documentation
  only; its output is parsed as JSON and validated against a Pydantic schema before it is
  stored or shown. It can never mutate an experiment, feature selection, or metric.

## 12. Architecture decisions

| # | Decision | Rationale |
|---|---|---|
| AD-1 | Contracts live in `ml_engine/contracts` and are reused verbatim by the API | one schema, no producer/consumer drift |
| AD-2 | Two Step Functions executions per experiment (EDA, then training) rather than one with a callback wait | feature review is human-scale time; a `waitForTaskToken` execution open for days is fragile and costs nothing to avoid |
| AD-3 | DynamoDB from v1 instead of deriving state from Step Functions | listing experiments, user ownership and best-model summary are queries Step Functions cannot answer cheaply |
| AD-4 | `ObjectStore` + `ExperimentOrchestrator` protocols with local implementations | the entire vertical slice runs and is tested without AWS; the AWS path is one adapter, not a fork of the logic |
| AD-5 | Preprocessing is fitted in the preparation job and persisted, not refitted per model | guarantees every model sees identical features and prevents fit-on-validation leakage |
| AD-6 | One SageMaker Training job per model via a `Map` state | isolates failures, parallelizes, and keeps the state machine independent of the model catalogue |
| AD-7 | Optional model libraries register conditionally | a missing CatBoost wheel degrades the catalogue, it does not break the platform |
| AD-8 | The S3 adapter lives inside `ml_engine/io` with a lazy `boto3` import | the control plane and the jobs share one implementation instead of duplicating it; `import ml_engine` still works without the AWS SDK. The Bedrock client is injected, so it needs no such exception |
| AD-9 | The evaluation job writes the terminal experiment state; Step Functions writes every other transition | whether a run is `COMPLETED` or `COMPLETED_WITH_WARNINGS` depends on the comparison the job computes. Re-deriving that in the state machine would duplicate the judgement; `ml_engine.reporting.final_status` stays the single place it is made |
| AD-10 | The preparation job fits one pipeline per required preprocessing strategy (dense one-hot, native categorical) rather than one overall | CatBoost consumes categories natively while linear models need a dense matrix. Both are still fitted on the training fold only, so the guarantee in AD-5 holds for every model |

## 13. Local development

`ML_FACTORY_MODE=local` (default in `.env.example`) selects the local adapters:
`LocalObjectStore` (a directory tree that mirrors the S3 layout) and `LocalOrchestrator`
(runs the same `jobs/*` entrypoints in a background thread pool, in dependency order).
The same contracts, the same job code, the same artifacts — only the boundary adapters differ.
`ML_FACTORY_MODE=aws` selects S3 + Step Functions + DynamoDB.

`make dev` runs the API on :8000 and Vite on :5173 with a proxy to the API.
`make test` runs pytest; `make lint` runs ruff. See `README.md`.
