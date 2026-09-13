# Driving the engine from a notebook

The platform is the front door: a data scientist picks a dataset and a target, reviews the
feature recommendations, and gets ranked models. It covers the repeatable work, which is most
of it.

This document covers the rest — the run that needs a custom loss, an unusual split, a plot
nobody anticipated. `ml_engine` is an ordinary Python package and the SageMaker job entrypoints
are ordinary functions, so any notebook can call them directly: the same profiling, the same
leakage rules, the same artifact layout, the same S3 bucket. Nothing is reimplemented and
nothing is bypassed.

Every snippet below is executed by `tests/test_notebook_recipes.py`. If the engine's API moves,
that test fails and this file gets fixed.

## Setup

```bash
pip install -e ".[models]"     # models extra pulls in xgboost and catboost
```

Credentials are boto3's own: a profile in `~/.aws`, or `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` in the environment. The engine builds a plain `boto3.client("s3")` when
you construct `S3ObjectStore()` with no argument, so if `aws s3 ls` works in your shell, this
works too. On a SageMaker notebook the attached execution role is picked up automatically.

Two objects appear in every recipe:

| | What it is |
|---|---|
| `store` | an `ObjectStore` — `S3ObjectStore()` for S3, `LocalObjectStore()` for plain paths and `file://` URIs |
| `layout` | the one definition of where artifacts go; never build a path by hand |

```python
from ml_engine.io import ExperimentLayout, S3ObjectStore

store  = S3ObjectStore()
layout = ExperimentLayout.for_experiment("s3://my-ml-factory-artifacts", "notebook-001")
```

`layout.base` is `s3://my-ml-factory-artifacts/ml-factory/experiments/notebook-001`. Use a
distinct experiment id per run — the platform never overwrites another experiment's prefix and
neither should you.

`LocalObjectStore()` takes no root: pass `ExperimentLayout.for_experiment("/tmp/scratch", …)`
and it resolves the same layout on disk. Useful for trying something out before writing to the
shared bucket — but only the S3 root the platform is configured with shows up in the UI.

## Recipe A — EDA and leakage screening

The smallest useful thing. Profiles the dataset and runs every deterministic leakage check,
writing `eda/eda.json` and `eda/leakage.json`.

```python
from jobs.profiling.main import run_profiling

eda, leakage = run_profiling(
    store, layout,
    experiment_id="notebook-001",
    dataset_uri="s3://my-approved-data/curated/churn.csv",
    target_column="churned",
)

eda.dataset.row_count          # 400
eda.target.inferred_problem_type
[f.rule for f in leakage.findings]
```

Both are Pydantic models, so `eda.model_dump()` gives you a dict and every field is typed. The
column profiles are in `eda.columns`; per-feature risk is in `leakage.feature_risks`.

Nothing is excluded for you. `leakage.findings` carries a rule id, a severity and a recommended
action, and the decision stays yours — the same contract the UI shows a data scientist.

## Recipe B — make the run appear in the UI

Recipe A writes artifacts but the experiment list will not show them. The list scans the bucket
for prefixes containing `experiment.json`, so write an `ExperimentDefinition` and the run
becomes a first-class experiment: it appears in the list, and its EDA, features and comparison
tabs open like any other.

```python
from datetime import UTC, datetime

from ml_engine.contracts.config import DatasetReference
from ml_engine.contracts.experiment import ControlPlaneState, ExperimentDefinition
from ml_engine.io import write_model

write_model(store, layout.definition, ExperimentDefinition(
    experiment_id="notebook-001",
    name="Churn baseline (notebook)",
    created_by="your.name",
    created_at=datetime.now(UTC),
    dataset=DatasetReference(uri="s3://my-approved-data/curated/churn.csv", file_format="csv"),
    target_column="churned",
    artifact_prefix=layout.base,
))

write_model(store, layout.control_state, ControlPlaneState(
    experiment_id="notebook-001",
    updated_at=datetime.now(UTC),
))
```

`experiment.json` is write-once — the platform never rewrites it, and neither should you.
Status lives in the two state files beside it, which is why `control.json` is separate.

## Recipe C — the whole pipeline

Preparation, one training run per model, then evaluation. This is exactly what the training
state machine does; the difference is that the `Map` state runs the models in parallel on
separate SageMaker instances, while your kernel runs them one after another.

The pipeline reads its frozen configuration from `config/experiment_config.json`, so write that
first. It is the reproducibility record: dataset identity, selected features, split, seed,
hyperparameters, package versions.

```python
from ml_engine.contracts.common import ClassWeighting, ProblemType, RequestedProblemType
from ml_engine.contracts.config import (
    ComputeConfig, DatasetReference, EnvironmentCapture, ExperimentConfig,
    FeatureSelection, ModelSpec, PreprocessingConfig, SplitConfig,
)

FEATURES = ["tenure_months", "monthly_charges", "contract"]
MODELS = ["logistic_regression", "random_forest"]

write_model(store, layout.experiment_config, ExperimentConfig(
    experiment_id="notebook-001",
    name="Churn baseline (notebook)",
    created_at=datetime.now(UTC),
    dataset=DatasetReference(uri="s3://my-approved-data/curated/churn.csv", file_format="csv"),
    target_column="churned",
    problem_type=ProblemType.BINARY_CLASSIFICATION,
    requested_problem_type=RequestedProblemType.AUTO,
    primary_metric="roc_auc",
    feature_selection=FeatureSelection(selected_features=FEATURES),
    split=SplitConfig(),
    preprocessing=PreprocessingConfig(),
    models=[ModelSpec(name=name) for name in MODELS],
    class_weighting=ClassWeighting.AUTO,
    compute=ComputeConfig(),
    environment=EnvironmentCapture(
        python_version="3.12",
        platform="notebook",
        ml_factory_version="1.0.0",
        captured_at=datetime.now(UTC),
    ),
    artifact_prefix=layout.base,
))
```

Then run the three stages:

```python
from jobs._common.experiment_state import ArtifactExperimentStateWriter
from jobs.evaluation.main import run_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.training.main import run_training

prep = run_preparation(store, layout, experiment_id="notebook-001")
prep.rows_after, prep.usable_features, prep.dropped_features

for name in MODELS:
    meta = run_training(store, layout, experiment_id="notebook-001", model_name=name)
    print(meta.model_name, meta.status, meta.primary_metric, meta.primary_score)

comparison = run_evaluation(
    store, layout,
    experiment_id="notebook-001",
    state_writer=ArtifactExperimentStateWriter(store, layout),
)
comparison.best_model, comparison.best_score, comparison.succeeded_count
```

`run_preparation` fits every preprocessing pipeline **on the training fold only** and writes the
materialized folds, so a notebook cannot accidentally leak the validation set through a scaler
or an encoder. That property is in the engine, not in the orchestration.

`state_writer` is what makes the UI show the run as completed with its best model. Leave it out
and the experiment stays in whatever state Recipe B left it.

## Where a notebook is the wrong tool

| | Notebook | Platform |
|---|---|---|
| A model raising | kills the cell; the run is lost | `Catch` per branch; the other models still finish |
| Five models | sequential, in your kernel | parallel, one SageMaker job each |
| Compute | your instance, sized for the worst case, billed while you think | per job, sized per job, billed per second |
| Who can run it | you | anyone with a browser |

Use a notebook for the run that does not fit. Use the platform for the ninety that do.

## What still holds

These are properties of the engine, so they apply to a notebook exactly as they do to a
SageMaker job:

* Preprocessing is fitted on the training fold only.
* Every number comes from pandas or scikit-learn. If you add Bedrock reasoning, it reads the
  artifacts and interprets them — it never computes a statistic or a metric.
* Nothing is dropped from the feature set except the target unless you drop it.
* Every artifact is a versioned Pydantic contract, so what you write is what the UI reads.

Two things a notebook does *not* inherit, because they live in the control plane rather than
the engine: the dataset allow-list (`ML_FACTORY_ALLOWED_DATASET_PREFIXES`) and the artifact-root
check. `run_profiling` will read any URI your credentials can reach. Your IAM policy is the
boundary that still applies.
