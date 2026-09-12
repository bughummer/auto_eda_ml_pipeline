"""Run a complete experiment locally, end to end, without AWS.

Exercises the same path the platform uses in production — create, EDA, feature review,
training, comparison — against the local object store and the in-process orchestrator.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import Settings
from backend.container import build_container
from backend.schemas.experiments import (
    CreateExperimentRequest,
    TrainingConfigRequest,
    UpdateFeatureSelectionRequest,
)
from ml_engine.contracts.common import (
    ExperimentStatus,
    LeakageRiskLevel,
    RecommendedAction,
)
from scripts.sample_data import write_sample_datasets

ROOT = Path("var/ml-factory-demo")


def main() -> int:
    datasets = write_sample_datasets(Path("var/sample-data"))
    settings = Settings(mode="local", local_root=ROOT)
    container = build_container(settings)
    service = container.experiments

    print("1. Creating experiment")
    record = service.create(
        CreateExperimentRequest(
            name="Customer churn baseline",
            dataset_uri=str(datasets["churn"].resolve()),
            target_column="churned",
        ),
        user="demo@corp.example",
    )
    print(f"   experiment_id={record.experiment_id} status={record.status.value}")

    print("2. Waiting for EDA")
    container.orchestrator.wait_for_idle(timeout=300)
    eda = service.eda(record.experiment_id)
    print(
        f"   rows={eda.dataset.row_count} columns={eda.dataset.column_count} "
        f"problem_type={eda.target.inferred_problem_type.value} warnings={len(eda.warnings)}"
    )
    for warning in eda.warnings[:5]:
        print(f"   - [{warning.severity.value}] {warning.rule} {warning.column or ''}")

    print("3. Feature review")
    review = service.feature_review(record.experiment_id)
    flagged = [f for f in review.features if f.leakage_risk is not LeakageRiskLevel.NONE]
    print(f"   {review.selected_count} selected, {review.excluded_count} excluded by default")
    for item in flagged:
        print(f"   - {item.feature}: {item.leakage_risk.value} -> {item.recommended_action.value}")
    # The platform recommends; this demo plays the role of a data scientist who accepts the
    # exclusion recommendations. Nothing was removed automatically.
    excluded_actions = {
        RecommendedAction.CONSIDER_EXCLUDING,
        RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
    }
    keep = [
        f.feature
        for f in review.features
        if f.selected and f.recommended_action not in excluded_actions
    ]
    service.save_features(
        record.experiment_id,
        UpdateFeatureSelectionRequest(
            selected_features=keep,
            exclusion_reasons={
                f.feature: (f.reasons[0] if f.reasons else "excluded during review")
                for f in review.features
                if f.feature not in keep
            },
        ),
        user="demo@corp.example",
    )
    print(f"   training on {len(keep)} features after acting on recommendations")

    print("4. Training")
    started = time.perf_counter()
    service.start_training(record.experiment_id, TrainingConfigRequest(), user="demo@corp.example")
    container.orchestrator.wait_for_idle(timeout=1800)
    status = service.training_status(record.experiment_id)
    print(f"   status={status.status.value} in {time.perf_counter() - started:.1f}s")

    print("5. Comparison")
    comparison = service.comparison(record.experiment_id)
    print(f"   primary metric {comparison.primary_metric} ({comparison.direction.value})")
    for entry in comparison.models:
        score = "n/a" if entry.primary_score is None else f"{entry.primary_score:.4f}"
        print(f"   {entry.rank or '-':>2} {entry.model_name:<24} {entry.status.value:<10} {score}")
    print(f"   best: {comparison.best_model}")

    summary = service.summary(record.experiment_id)
    print(f"6. Summary artifact written with {len(summary.warnings)} warnings")
    print(f"   artifacts: {record.artifact_prefix}")
    container.shutdown()
    return 0 if status.status is not ExperimentStatus.FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
