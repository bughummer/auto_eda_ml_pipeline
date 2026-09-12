"""Terminal experiment state, written by the component that computes it.

Step Functions owns sequencing and every intermediate transition. The final state, however,
depends on the comparison the evaluation job produces — whether every model succeeded, and
whether the run carries warnings — so the evaluation job writes it rather than re-deriving
that judgement in the state machine. ``ml_engine.reporting.final_status`` remains the single
place the decision is made.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from ml_engine.contracts.comparison import ComparisonReport
from ml_engine.reporting import final_status

LOGGER = logging.getLogger("ml_factory.jobs.state")


class ExperimentStateWriter(Protocol):
    def record_completion(self, experiment_id: str, comparison: ComparisonReport) -> None: ...


class DynamoExperimentStateWriter:
    """Writes the terminal status, best model and score onto the experiment record."""

    def __init__(self, table_name: str, region: str, table: Any | None = None) -> None:
        if table is None:
            import boto3

            table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self._table = table

    def record_completion(self, experiment_id: str, comparison: ComparisonReport) -> None:
        status = final_status(comparison)
        expression_values: dict[str, Any] = {
            ":status": status.value,
            ":stage": "completed",
            ":metric": comparison.primary_metric,
            ":now": datetime.now(UTC).isoformat(),
        }
        update = (
            "SET #status = :status, current_stage = :stage, primary_metric = :metric, "
            "updated_at = :now"
        )
        if comparison.best_model:
            update += ", best_model = :best_model, best_score = :best_score"
            expression_values[":best_model"] = comparison.best_model
            expression_values[":best_score"] = str(comparison.best_score)
        self._table.update_item(
            Key={"experiment_id": experiment_id},
            UpdateExpression=update,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues=expression_values,
        )
        LOGGER.info("Recorded terminal state %s for %s", status.value, experiment_id)


def build_state_writer(table_name: str | None, region: str | None) -> ExperimentStateWriter | None:
    """No table configured (local mode, or a dry run) means no record to update."""
    if not table_name:
        return None
    return DynamoExperimentStateWriter(table_name, region or "eu-central-1")
