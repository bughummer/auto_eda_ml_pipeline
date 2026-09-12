# AGENTS.md — working rules for this repository

Read `architecture.md` before changing anything structural. It is authoritative; if you have a
materially better design, change the document, the contracts, the code and the tests together.

## Layout and dependency direction

```
ml_engine → pandas/numpy/sklearn/model libs only      (no boto3, no fastapi, no backend, no jobs)
jobs      → ml_engine
backend   → ml_engine.contracts + aws adapters        (never trains a model in-process on AWS mode)
frontend  → backend HTTP API only                     (never AWS)
```

`tests/test_architecture_boundaries.py` enforces this. Do not weaken it; fix the import instead.

## Non-negotiables

1. **No LLM arithmetic.** Bedrock reads deterministic artifacts. It never computes a statistic,
   a metric, or a ranking, and it never mutates experiment state or feature selection.
2. **No fitting on validation data.** Imputers, encoders, scalers and any learned transform fit
   on the training fold only. `ml_engine/preprocessing` is the only place that fits transforms.
3. **The browser never calls AWS.** All AWS access goes through the FastAPI control plane.
4. **The control plane does no heavy compute.** In `aws` mode FastAPI only validates, records,
   starts executions and reads artifacts.
5. **Status comes from the backend.** The frontend renders `status`/`current_stage`; it never
   derives, guesses, or advances workflow state.
6. **Nothing is auto-excluded except the target.** Deterministic checks recommend; the user
   decides. Recommendations must carry rule id, severity, explanation and recommended action.
7. **Every artifact is a Pydantic contract** from `ml_engine/contracts`. Do not hand-build dicts
   for anything written to S3 or returned from the API.
8. **No secrets in the repository.** Configuration comes from environment variables; see
   `.env.example`.

## Conventions

* Python 3.12, Pydantic v2, `from __future__ import annotations` not required (3.12 baseline).
* Line length 100, ruff for lint and format (`make lint`, `make format`).
* Type hints on all public functions. `Any` needs a reason.
* Contracts are frozen-ish: additive changes are fine; renames require updating the frontend
  types, the artifact readers and the tests in the same commit.
* Artifact file names and S3 prefixes are defined once, in `ml_engine/io/layout.py`. Never
  hardcode a path string elsewhere.
* Model plugins register themselves in `ml_engine/models/registry.py`. Orchestration code must
  never branch on a model name.
* Tests: `tests/<package>/test_<module>.py`. New behaviour ships with a test. The end-to-end
  local experiment test (`tests/backend/test_end_to_end_local.py`) must stay green.

## Adding things

**A model plugin**: implement `ModelPlugin` in `ml_engine/models/<name>.py`, register it,
add it to `tests/ml_engine/test_models.py::test_all_registered_plugins_train`. Nothing else
changes — not the state machine, not the API, not the UI.

**An EDA warning**: add the rule in `ml_engine/profiling/warnings.py` with a new stable
`rule` id and a `WarningCategory`. Never reuse a rule id with different semantics.

**A leakage rule**: add it in `ml_engine/leakage/rules.py` as a pure function
`(context) -> list[LeakageFinding]` and register it in `RULES`.

**An API endpoint**: route in `backend/api/routes/`, logic in `backend/services/`, schema in
`backend/schemas/` (API-only) or `ml_engine/contracts` (artifact-shaped). Update
`architecture.md` §6 and the frontend client in the same change.

## Local development

`ML_FACTORY_MODE=local` (default) uses `LocalObjectStore` + `LocalOrchestrator`: the real job
entrypoints run in a background thread pool against a directory tree that mirrors the S3
layout. Use it for everything except AWS integration work. `make demo` runs a full experiment.
