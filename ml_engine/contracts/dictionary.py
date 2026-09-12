"""Data dictionary (business column documentation) contracts.

Source formats (JSON / CSV / XLSX) are normalized at the edge, in
``ml_engine.dictionary``. No spreadsheet-specific concept appears past that boundary.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import StrictModel

DICTIONARY_SCHEMA_VERSION = "1.0"


class ColumnDocumentation(StrictModel):
    column: str
    business_definition: str | None = None
    source_system: str | None = None
    collection_timing: str | None = Field(
        default=None, description="When the value is produced relative to the predicted event."
    )
    update_frequency: str | None = None
    available_at_prediction_time: bool | None = None
    owner: str | None = None
    contact: str | None = None
    notes: str | None = None
    extra: dict[str, str] = Field(
        default_factory=dict, description="Unmapped source columns, preserved verbatim."
    )


class DataDictionary(StrictModel):
    schema_version: str = DICTIONARY_SCHEMA_VERSION
    experiment_id: str | None = None
    source_format: str
    source_name: str | None = None
    uploaded_at: datetime
    columns: list[ColumnDocumentation] = Field(default_factory=list)
    unmatched_columns: list[str] = Field(
        default_factory=list, description="Documented columns absent from the dataset."
    )
    undocumented_columns: list[str] = Field(
        default_factory=list, description="Dataset columns with no documentation."
    )

    def for_column(self, name: str) -> ColumnDocumentation | None:
        return next((c for c in self.columns if c.column == name), None)
