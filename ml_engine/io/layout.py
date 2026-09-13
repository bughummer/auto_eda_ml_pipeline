"""The one definition of the experiment artifact layout.

Never hardcode an artifact path anywhere else — import ``ExperimentLayout``.
Mirrored exactly by ``architecture.md`` §7.
"""

from dataclasses import dataclass

from ml_engine.io.uri import join_uri

ARTIFACT_NAMESPACE = "ml-factory/experiments"

DEFINITION_FILE = "experiment.json"
CONTROL_STATE_FILE = "state/control.json"
WORKFLOW_STATE_FILE = "state/workflow.json"
EDA_FILE = "eda/eda.json"
LEAKAGE_FILE = "eda/leakage.json"
EXPERIMENT_CONFIG_FILE = "config/experiment_config.json"
SELECTED_FEATURES_FILE = "config/selected_features.json"
TRAINING_CONFIG_FILE = "config/training_config.json"
DATA_DICTIONARY_FILE = "config/data_dictionary.json"
PREPARATION_FILE = "validation/validation.json"
PREPROCESSOR_FILE_TEMPLATE = "preprocessing/preprocessor_{strategy}.joblib"
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

    ``base`` is an ``s3://bucket/ml-factory/experiments/<id>`` URI. Any ``ObjectStore`` can
    resolve it, so the same layout applies to a SageMaker job, the control plane, a notebook
    driving the engine directly, and the in-memory store used by the tests.
    """

    base: str

    @classmethod
    def for_experiment(cls, root: str, experiment_id: str) -> "ExperimentLayout":
        """``root`` is the artifact bucket URI."""
        return cls(base=join_uri(root, ARTIFACT_NAMESPACE, experiment_id))

    @staticmethod
    def experiments_prefix(root: str) -> str:
        """Where every experiment under an artifact root lives. Used to list them."""
        return join_uri(root, ARTIFACT_NAMESPACE) + "/"

    def path(self, *parts: str) -> str:
        return join_uri(self.base, *parts)

    @property
    def definition(self) -> str:
        return self.path(DEFINITION_FILE)

    @property
    def control_state(self) -> str:
        return self.path(CONTROL_STATE_FILE)

    @property
    def workflow_state(self) -> str:
        return self.path(WORKFLOW_STATE_FILE)

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

    def preprocessor(self, strategy: str) -> str:
        """One fitted pipeline per preprocessing strategy; both fit on the train fold only."""
        return self.path(PREPROCESSOR_FILE_TEMPLATE.format(strategy=strategy))

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
