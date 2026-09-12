"""The one definition of the experiment artifact layout.

Never hardcode an artifact path anywhere else — import ``ExperimentLayout``.
Mirrored exactly by ``architecture.md`` §7.
"""

from dataclasses import dataclass

from ml_engine.io.uri import join_uri

ARTIFACT_NAMESPACE = "ml-factory/experiments"

EDA_FILE = "eda/eda.json"
LEAKAGE_FILE = "eda/leakage.json"
EXPERIMENT_CONFIG_FILE = "config/experiment_config.json"
SELECTED_FEATURES_FILE = "config/selected_features.json"
TRAINING_CONFIG_FILE = "config/training_config.json"
DATA_DICTIONARY_FILE = "config/data_dictionary.json"
PREPARATION_FILE = "validation/validation.json"
PREPROCESSOR_FILE = "preprocessing/preprocessor.joblib"
PREPROCESSING_METADATA_FILE = "preprocessing/preprocessing.json"
TRAIN_DATASET_FILE = "datasets/train.parquet"
VALIDATION_DATASET_FILE = "datasets/validation.parquet"
COMPARISON_FILE = "comparison/comparison.json"
SUMMARY_FILE = "report_data/experiment_summary.json"
REASONING_FILE = "reasoning/reasoning.json"
FAILURE_FILE = "failure.json"


@dataclass(frozen=True, slots=True)
class ExperimentLayout:
    """Resolves every artifact location for one experiment.

    ``base`` is an ``s3://bucket/ml-factory/experiments/<id>`` URI in AWS mode or a local
    directory in local mode. The layout is identical either way.
    """

    base: str

    @classmethod
    def for_experiment(cls, root: str, experiment_id: str) -> "ExperimentLayout":
        """``root`` is the artifact bucket URI (or local root directory)."""
        return cls(base=join_uri(root, ARTIFACT_NAMESPACE, experiment_id))

    def path(self, *parts: str) -> str:
        return join_uri(self.base, *parts)

    @property
    def eda(self) -> str:
        return self.path(EDA_FILE)

    @property
    def leakage(self) -> str:
        return self.path(LEAKAGE_FILE)

    @property
    def experiment_config(self) -> str:
        return self.path(EXPERIMENT_CONFIG_FILE)

    @property
    def training_config(self) -> str:
        return self.path(TRAINING_CONFIG_FILE)

    @property
    def selected_features(self) -> str:
        return self.path(SELECTED_FEATURES_FILE)

    @property
    def data_dictionary(self) -> str:
        return self.path(DATA_DICTIONARY_FILE)

    @property
    def preparation(self) -> str:
        return self.path(PREPARATION_FILE)

    @property
    def preprocessor(self) -> str:
        return self.path(PREPROCESSOR_FILE)

    @property
    def preprocessing_metadata(self) -> str:
        return self.path(PREPROCESSING_METADATA_FILE)

    @property
    def train_dataset(self) -> str:
        return self.path(TRAIN_DATASET_FILE)

    @property
    def validation_dataset(self) -> str:
        return self.path(VALIDATION_DATASET_FILE)

    @property
    def comparison(self) -> str:
        return self.path(COMPARISON_FILE)

    @property
    def summary(self) -> str:
        return self.path(SUMMARY_FILE)

    @property
    def reasoning(self) -> str:
        return self.path(REASONING_FILE)

    @property
    def models_prefix(self) -> str:
        return self.path("models")

    def model_dir(self, model_name: str) -> str:
        return self.path("models", model_name)

    def model_artifact(self, model_name: str) -> str:
        return self.path("models", model_name, "model.joblib")

    def model_metadata(self, model_name: str) -> str:
        return self.path("models", model_name, "metadata.json")

    def model_failure(self, model_name: str) -> str:
        return self.path("models", model_name, FAILURE_FILE)

    def report(self, name: str) -> str:
        return self.path("reports", name)
