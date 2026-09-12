# ML Factory — Implementation Plan

Companion to `architecture.md`. Tracks what is built, in what order, and what is deliberately
deferred. Checked items are implemented and covered by tests in this repository.

## Sequencing rule

Vertical slices, not horizontal scaffolding. A phase is done when a user-visible path works
end to end (UI → API → orchestration → job → artifact → UI) and is tested.

---

## Phase A — architecture and contracts ✅

- [x] Inspect repository (empty repository — greenfield; nothing to preserve or migrate)
- [x] `architecture.md`, `implementation_plan.md`, `AGENTS.md`, `README.md`
- [x] API contract (`architecture.md` §6) and Pydantic v2 contracts in `ml_engine/contracts/`
- [x] Experiment state model (§5) with backend/Step Functions ownership split
- [x] S3 artifact layout (§7)
- [x] IAM boundaries (§11) + policy documents in `infrastructure/iam/`
- [x] Model plugin architecture (§9)
- [x] EDA output schema (`ml_engine/contracts/eda.py`)
- [x] Error-handling strategy (§10) + `backend/errors.py`
- [x] Local development strategy (§13): `ObjectStore` / `ExperimentOrchestrator` protocols

## Phase B — EDA vertical slice ✅

- [x] `ml_engine/io`: S3 URI parsing, CSV/Parquet loading, `ObjectStore` (local + S3)
- [x] `ml_engine/profiling`: dataset summary, per-column profile, numeric/categorical stats,
      target analysis, deterministic warnings
- [x] `jobs/profiling`: Processing entrypoint writing `eda/eda.json` (+ `eda/leakage.json`)
- [x] `POST /api/v1/experiments`: validate → allow-list check → create record → start
      workflow → return `{experiment_id, status}` without blocking
- [x] `GET /experiments/{id}`, `/eda` served from artifacts
- [x] Local orchestrator so the slice runs without AWS
- [x] Frontend: create experiment, experiment list, EDA view with TanStack Query polling

## Phase C — deterministic ML engine ✅

- [x] Problem types: binary / multiclass / regression, with `auto` inference + user override
- [x] `ModelPlugin` base, registry, conditional registration of optional libraries
- [x] Classification: logistic regression, random forest, XGBoost, CatBoost
- [x] Regression: elastic net, random forest, XGBoost, CatBoost
- [x] Conservative defaults; no hyperparameter search

## Phase D — deterministic preprocessing ✅

- [x] Numeric imputation (+ scaling only where the algorithm needs it)
- [x] Categorical missing handling + one-hot for dense models, native categories for CatBoost
- [x] Boolean normalization; datetime detection → `year`/`month`/`day_of_week` derivation
- [x] Text-like detection → excluded by default with a warning
- [x] `sklearn` `Pipeline`/`ColumnTransformer`, fitted on the training fold only, persisted

## Phase E — deterministic leakage and feature review ✅

- [x] `ml_engine/leakage`: exact target duplication, inverse binary target, duplicate columns,
      near-deterministic single-feature relationship, suspicious names, identifier-like,
      post-outcome name heuristics
- [x] `potential_leakage` / `requires_review` severities with rule id, explanation,
      recommended action; nothing is removed silently
- [x] `GET/PUT /experiments/{id}/features` persisting `selected_features.json`
- [x] Feature review UI: per-feature toggle, type, missing %, cardinality, warnings, risk,
      reason, recommended action, filtering, sorting, guarded bulk actions

## Phase F — splitting ✅

- [x] Stratified split for classification, random split for regression
- [x] `validation_fraction` (default 0.2) and `random_seed` configurable
- [x] `SplitStrategy` abstraction with room for temporal/grouped splitting

## Phase G — metrics ✅

- [x] Binary: ROC AUC, PR AUC, accuracy, precision, recall, F1, log loss, confusion matrix,
      positive class rate, predicted positive rate
- [x] Multiclass: accuracy, macro/weighted F1, macro precision/recall, confusion matrix, log loss
- [x] Regression: MAE, RMSE, R², MAPE (zero-safe), SMAPE, bias
- [x] Primary metric selection with correct optimization direction; failures degrade to warnings

## Phase H — class imbalance ✅

- [x] Imbalance detection in EDA target analysis (imbalance ratio + severity warning)
- [x] `class_weighting: auto | none` honoured per plugin (`class_weight`, `scale_pos_weight`,
      CatBoost class weights). No SMOTE.

## Phase I — AWS training execution ✅

- [x] Step Functions ASL: preparation → `Map` over models (Training jobs) → evaluation
- [x] Per-model `Catch` so one model failure does not fail the experiment
- [x] Full status enum incl. `COMPLETED_WITH_WARNINGS`
- [x] Ephemeral CPU compute; instance types configurable; no endpoints

## Phase J — artifacts and reproducibility ✅

- [x] S3 layout per `architecture.md` §7; source dataset referenced, never copied
- [x] Dataset identity (version id / etag / size / last modified), seeds, package versions,
      environment metadata, hyperparameters captured in `experiment_config.json`
- [x] `ModelMetadata` per model; normalized + raw feature importance
- [x] `comparison.json` with primary metric, direction, best model, per-model status
- [x] `experiment_summary.json` for reporting and reasoning input

## Phase K — application state ✅

- [x] DynamoDB experiment record (metadata/state only) with an in-memory implementation for
      local development and tests

## Phase L — data dictionary ✅

- [x] JSON / CSV / XLSX ingestion normalized into `DataDictionary`
- [x] Upload + retrieval endpoints; normalized artifact stored beside the experiment config

## Phase M — Bedrock semantic reasoning ✅

- [x] Prompt assembly from deterministic artifacts + data dictionary only
- [x] Strict JSON schema validation of model output (`ReasoningReport`)
- [x] Semantic leakage findings, feature explanations, data-quality interpretation,
      model-behaviour narrative, proposed follow-up experiments (proposals only)
- [x] Hard guarantees: no metric invention, no feature mutation, no code execution — the
      reasoning artifact is read-only output stored under `reasoning/`

## Deliberately deferred (not in v1.0)

| Item | Reason |
|---|---|
| Time-series / forecasting | requires temporal splitting, backtesting and horizon semantics |
| Hyperparameter search | first establish trustworthy baselines and reproducibility |
| SHAP | infrastructure cost and runtime variance; native importance covers v1 needs |
| SMOTE / resampling | class weighting first; resampling changes evaluation semantics |
| NLP vectorization | text columns are detected, warned about and excluded |
| Visual S3 browsing | validated URI entry + allow-list is sufficient and safer |
| Automated model deployment / endpoints | out of scope by mandate |
| Autonomous experiment execution by the LLM | v1 proposes; a human starts the next experiment |

## Verification

```
make install      # python venv + npm install
make lint         # ruff: clean
make test         # pytest: 312 tests, all green
make build-front  # tsc --noEmit + vite build: clean
make demo         # a complete local experiment, end to end, without AWS
```

The suite covers the deterministic engine unit by unit, the job entrypoints (including the
rule that failures become artifacts rather than stack traces), the control plane's validation
and state transitions, the documented architecture boundaries, and a full local experiment
from creation to comparison. The frontend was additionally verified by driving a real browser
through create → EDA → feature review → training → comparison against the running API.
