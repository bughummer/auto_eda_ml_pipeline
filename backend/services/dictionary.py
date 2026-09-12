"""Data dictionary use cases: ingest business documentation, reconcile it, store it."""

import logging

from backend.config import Settings
from backend.errors import ArtifactNotReadyError, ValidationError
from backend.repositories import ExperimentRepository
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.dictionary import DictionaryParseError, parse_dictionary
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists, write_model

LOGGER = logging.getLogger("ml_factory.services.dictionary")

_EXTENSION_FORMATS = {"json": "json", "csv": "csv", "xlsx": "xlsx", "xls": "xlsx"}


class DataDictionaryService:
    def __init__(
        self, *, settings: Settings, repository: ExperimentRepository, store: ObjectStore
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store

    def upload(
        self, experiment_id: str, *, filename: str, payload: bytes, source_format: str | None = None
    ) -> DataDictionary:
        """Parse an uploaded dictionary, reconcile it with the dataset and store it."""
        if len(payload) > self._settings.max_dictionary_upload_bytes:
            raise ValidationError(
                "The data dictionary exceeds the maximum upload size.",
                {"max_bytes": self._settings.max_dictionary_upload_bytes},
            )
        fmt = (source_format or _format_from_filename(filename) or "").lower()
        if not fmt:
            raise ValidationError(
                "Could not determine the file format. Upload a .json, .csv or .xlsx file."
            )

        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        dataset_columns = [profile.name for profile in eda.columns] if eda else None

        try:
            dictionary = parse_dictionary(
                payload,
                source_format=fmt,
                source_name=filename,
                experiment_id=experiment_id,
                dataset_columns=dataset_columns,
            )
        except DictionaryParseError as error:
            raise ValidationError(str(error)) from error

        write_model(self._store, layout.data_dictionary, dictionary)
        LOGGER.info(
            "Stored data dictionary for %s: %d documented columns, %d undocumented",
            experiment_id,
            len(dictionary.columns),
            len(dictionary.undocumented_columns),
        )
        return dictionary

    def get(self, experiment_id: str) -> DataDictionary:
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        dictionary = read_model_if_exists(self._store, layout.data_dictionary, DataDictionary)
        if dictionary is None:
            raise ArtifactNotReadyError("No data dictionary has been uploaded for this experiment.")
        return dictionary


def _format_from_filename(filename: str) -> str | None:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _EXTENSION_FORMATS.get(suffix)
