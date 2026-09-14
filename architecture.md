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
5. **Everything reproducible, in one place.** Every experiment — its record, its EDA, its
   configuration, its models and its reports — is a set of JSON objects under one predictable
   S3 prefix, including dataset identity, seeds, package versions and hyperparameters. There
   is no separate state store to keep in sync, and copying a prefix copies the experiment.
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
                        ┌───────────────┬───────────────────┬───────────────┐
                        ▼               ▼                   ▼               │
                 ┌────────────┐  ┌────────────────┐  ┌───────────┐          │
                 │  Step      │  │       S3       │  │  Bedrock  │   no database:
                 │  Functions │  │ artifacts AND  │  │(reasoning)│   records are
                 └─────┬──────┘  │ experiment     │  └───────────┘   objects in S3
                       │         │ records        │
                       │         └────┬───────────┘
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
  features/       derived-feature specifications: validation and row-wise execution
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
  preprocessing/  Processing job: validation, derived features, split, fit preprocessing,
                  materialize folds
  training/       Training job: one model, one job
  evaluation/     Processing job: aggregation, comparison, experiment summary

backend/          FastAPI control plane
  api/routes/     HTTP surface
  schemas/        request/response models (API-only; artifact schemas come from ml_engine)
  services/       use cases (experiment lifecycle, artifacts, dictionary, reasoning)
  repositories/   experiment record persistence (S3 objects beside the artifacts)
  orchestration/  ExperimentOrchestrator protocol, implemented by Step Functions
  aws/            boto3 session/client construction, proxy and credential config

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
* Step Functions writes every execution-driven transition to `state/workflow.json` via the
  `s3:putObject` SDK integration.
* `COMPLETED_WITH_WARNINGS` is used when the experiment produced a usable comparison but at
  least one model failed, or an evaluation/warning threshold was crossed.
* The frontend renders status; it never derives or guesses it.
* `current_stage` is a free-form human-readable stage label; `status` is the enum above.

State ownership summary:

Every experiment record is three small JSON objects under the experiment's own prefix, each
with exactly one writer, so nothing ever performs a read-modify-write on shared state:

| Object | Written by | Contents |
|---|---|---|
| `experiment.json` | the control plane, once | name, dataset identity, target, owner, prefix |
| `state/control.json` | the control plane | user-driven transitions, execution ARNs, requested models |
| `state/workflow.json` | Step Functions and the evaluation job | execution-driven status, stage, best model, failure |

The API returns the composition of the three. Ownership of the *status* is explicit: the
control plane claims it only when it actually sets a status, so recording an execution ARN
cannot take a stage back from a workflow that has moved on.

Per-model progress is not stored at all — it is derived from the model artifacts, which are
the source of truth: metadata means the model succeeded, a failure record means it failed.

| Data | Owner | Store |
|---|---|---|
| Experiment records | Backend + Step Functions | S3, beside the artifacts |
| Workflow execution position, retries | Step Functions | Step Functions |
| All ML artifacts (EDA, config, models, metrics, reports) | Jobs | S3 |
| Nothing | Frontend | — |

## 6. API contract (v1)

Base path `/api/v1`. All errors use the shared envelope in §10.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + dependency mode report |
| POST | `/experiments` | create experiment, start EDA. Returns `{experiment_id, status}` immediately |
| GET | `/experiments` | list experiment records; `?root=` lists another approved artifact bucket |
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

Every read endpoint accepts an optional `root` query parameter naming an approved artifact
bucket, which is how the UI shows experiments produced by another environment. Only configured
roots are accepted, and writes always target the platform's own bucket.

Requests and responses are Pydantic v2 models. Artifact-shaped responses reuse the exact
`ml_engine.contracts` models so a schema can never drift between producer and consumer.

## 7. S3 artifact layout

```
s3://<artifact-bucket>/ml-factory/experiments/<experiment_id>/
  experiment.json                       ExperimentDefinition (immutable)
  state/control.json                    ControlPlaneState
  state/workflow.json                   WorkflowState
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
StartAt: MarkEdaRunning (s3:putObject -> state/workflow.json)
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

* IAM: three **runtime** roles — the platform's own credentials, never assumed by hand —
  `MlFactoryBackendRole` (control plane: `states:StartExecution`,
  `states:DescribeExecution`, `s3:GetObject/PutObject` limited to the artifact prefix, `s3:ListBucket`/`GetObject` on approved dataset buckets,
  `bedrock:InvokeModel` on allow-listed model ids),
  `MlFactoryWorkflowRole` (Step Functions: create/describe SageMaker jobs, `iam:PassRole` to
  the job role, `s3:PutObject` limited to `…/experiments/*/state/*`),
  `MlFactoryJobRole` (SageMaker containers: read approved dataset prefixes, read/write the
  experiment artifact prefix, CloudWatch Logs). Policies in `infrastructure/iam/`.
  `MlFactoryBackendRole` is trusted for `sts:AssumeRole` from this account, so the corporate
  server running `docker compose` acts as it without any permission living on the identity
  whose keys are configured — see `infrastructure/README.md`. Separate from all three:
  `iam/deployer_policy.json` is what the *operator's own* AWS identity needs to create the
  stack and the roles in the first place — a distinct, narrower permission set that does not
  let the deployer act as the running platform.
* KMS: the artifact bucket uses a customer-managed key; the job role and the
  backend role get `kms:Decrypt`/`GenerateDataKey` on that key only.
* Dataset allow-list: the backend rejects any dataset URI whose bucket/prefix is not in
  `ML_FACTORY_ALLOWED_DATASET_PREFIXES`. This is enforced before any AWS call.
* No credentials in the repository. Configuration resolves in one place
  (`backend/config.py`) from environment variables, then `.env`, then
  `config/secrets/config.py` — the last of which is gitignored and is the only place an
  operator types a credential. Secret fields are `SecretStr`, so they cannot be logged or
  serialized by accident. A mounted `~/.aws` profile is preferred over static keys; when
  neither is configured, boto3's default chain applies unchanged.
* The corporate proxy is configured through `HTTPS_PROXY`/`NO_PROXY` and honoured by an
  explicit botocore `Config`.
* Bedrock input is assembled from deterministic artifacts and user-supplied documentation
  only; its output is parsed as JSON and validated against a Pydantic schema before it is
  stored or shown. It can never mutate an experiment, feature selection, or metric.

## 12. Architecture decisions

| # | Decision | Rationale |
|---|---|---|
| AD-1 | Contracts live in `ml_engine/contracts` and are reused verbatim by the API | one schema, no producer/consumer drift |
| AD-2 | Two Step Functions executions per experiment (EDA, then training) rather than one with a callback wait | feature review is human-scale time; a `waitForTaskToken` execution open for days is fragile and costs nothing to avoid |
| AD-3 | ~~DynamoDB for experiment records~~ — **superseded by AD-11** | listing experiments is a query Step Functions cannot answer cheaply, so some record store was needed |
| AD-4 | `ObjectStore` + `ExperimentOrchestrator` protocols, with the substitutes living in the test suite | the entire vertical slice is testable without AWS, while the product keeps exactly one wiring (see AD-12) |
| AD-5 | Preprocessing is fitted in the preparation job and persisted, not refitted per model | guarantees every model sees identical features and prevents fit-on-validation leakage |
| AD-6 | One SageMaker Training job per model via a `Map` state | isolates failures, parallelizes, and keeps the state machine independent of the model catalogue |
| AD-7 | Optional model libraries register conditionally | a missing CatBoost wheel degrades the catalogue, it does not break the platform |
| AD-8 | The S3 adapter lives inside `ml_engine/io` with a lazy `boto3` import | the control plane and the jobs share one implementation instead of duplicating it; `import ml_engine` still works without the AWS SDK. The Bedrock client is injected, so it needs no such exception |
| AD-9 | The evaluation job writes the terminal experiment state; Step Functions writes every other transition | whether a run is `COMPLETED` or `COMPLETED_WITH_WARNINGS` depends on the comparison the job computes. Re-deriving that in the state machine would duplicate the judgement; `ml_engine.reporting.final_status` stays the single place it is made |
| AD-10 | The preparation job fits one pipeline per required preprocessing strategy (dense one-hot, native categorical) rather than one overall | CatBoost consumes categories natively while linear models need a dense matrix. Both are still fitted on the training fold only, so the guarantee in AD-5 holds for every model |
| AD-11 | Experiment records live in the artifact bucket as three single-writer JSON objects; DynamoDB is removed (supersedes AD-3) | the platform then uses exactly two AWS data services — S3 and SageMaker — instead of three. An experiment becomes one self-contained prefix: copyable, auditable and restorable without a database. Listing is a prefix listing with a delimiter, and Step Functions writes stage transitions with the `s3:putObject` SDK integration, so the "cheap query" argument behind AD-3 no longer holds |
| AD-12 | No local execution mode: the platform is AWS-only, and the in-process runner is a test double under `tests/support/` | two supported runtimes meant two behaviours to reason about for no production benefit. The end-to-end tests still exercise the real job code through the same contracts, against an in-memory store addressed exactly as S3 is |
| AD-13 | Any configured artifact bucket can be browsed read-only from the UI | experiment results are already self-contained in their prefix, so showing another environment's results costs one query parameter rather than a second deployment or a shared database. Writes always target the platform's own bucket |
| AD-14 | Derived features are declared as specifications from a closed vocabulary, never as generated code | a specification can be validated before it runs, reproduced exactly from the frozen configuration, and read by a human asking where a column came from. Generated code can do none of those, and executing it would put model output inside the trust boundary the dataset allow-list and job role exist to draw. The vocabulary is `ratio`, `difference`, `date_difference`, `map_categories` and `is_missing` — all row-wise, so derivation before the split cannot move information across the fold boundary. Stateful operations (group aggregates, quantile bins, target encoding) are excluded: they need fold-aware fitting and belong with the preprocessing pipelines |
| AD-15 | The preparation job re-validates derived-feature specifications against the frame it actually loaded | the specification was checked when it was approved, but the dataset can change underneath a frozen configuration. A specification that reads the target must never be computed regardless of how it entered the config, and a refusal becomes a `HIGH` warning in the preparation report rather than a silently missing column |
| AD-16 | A proposed feature reaches a run only through an explicit human approval, recorded separately from validation | validation decides what may be *offered*; a person decides what is *used*. The proposal report keeps both, so the record shows what was suggested, what was refused and why, and what someone chose — and the frozen configuration carries only the last of those. Approving nothing is a normal outcome and is recorded as a decision, not as an absence |
| AD-17 | One semantic-type classifier serves both validation paths | a specification is checked against the EDA report when it is offered and against the dataframe in preparation. Reading dtypes directly in the second path disagreed with the profiler in the first — a 0/1 integer column is `int64` to pandas and a boolean to the profiler — which would show a reviewer a feature the platform was never going to build. Both paths now run `infer_semantic_type`, and a test asserts the two descriptions match |
| AD-18 | `AWS_ROLE_ARN` assumes a role with auto-refreshing credentials, built from botocore's own `AssumeRoleCredentialFetcher`/`DeferredRefreshableCredentials` rather than a one-shot `sts:AssumeRole` call | a role's temporary credentials expire in 1–12 hours; every boto3 client the container uses is built once at startup (`build_container`), so a naive assume-role call would work in testing and then fail every request after the first expiry with no restart to recover. This is the same mechanism a `role_arn` profile in `~/.aws/config` gets automatically, made configurable from `config/secrets/config.py` instead of a second file on disk. `scripts/deploy_aws.py`'s `DEPLOYER_ROLE_ARN` is deliberately the simpler one-shot version — that script runs for a few minutes and exits, so refresh is not needed there |
| AD-19 | Each of the three runtime roles can be supplied by ARN instead of created, via a `CloudFormation` `Condition` per role, not a second template | some organizations require every IAM role to be created and reviewed outside of CloudFormation. A parallel "bring your own roles" template would drift from the one that creates them; a per-role `Fn::If` between the created resource's `Arn` and the supplied `Ref` keeps both paths in the one file staying in sync by construction, and lets the three roles be created, supplied, or mixed independently. The stack never attaches a policy to a role it did not create — the supplied role must already carry the matching `infrastructure/iam/*_role_policy.json`. `tests/infrastructure/test_cloudformation_template.py` walks every `Fn::GetAtt` onto one of the three role resources and asserts it is reachable only behind that role's own condition, because a reference left unguarded deploys fine when every role is created and fails only in the one case this feature exists for |

## 13. Development

There is one runtime: AWS. No second code path exists for running the ML anywhere else.

Development and the test suite substitute two *adapters* — never a second application wiring:

* `tests/support/memory_store.py` — an object store addressed with real `s3://` URIs, so URI
  parsing, the dataset allow-list, prefix listing and the artifact layout are all under test.
* `tests/support/inline_orchestrator.py` — runs the same `jobs/*` entrypoints in-process, in
  the same order, writing the same `state/workflow.json` the state machine writes.

`build_container()` and `create_app()` accept those adapters as arguments, which is why the
product needs no local mode: the seam already exists for tests, and nothing more is required.

`make dev` runs the API on :8000 and Vite on :5173 with a proxy to the API. `make test` runs
pytest; `make lint` runs ruff; `make deploy-aws` builds and pushes the job image, uploads the
workflow definitions and deploys the stack. See `README.md`.
