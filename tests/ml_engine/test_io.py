"""The IO boundary: URI safety, the artifact layout and dataset loading."""

import pandas as pd
import pytest

from ml_engine.contracts.eda import EdaReport
from ml_engine.io import (
    ExperimentLayout,
    InvalidS3UriError,
    LocalObjectStore,
    ObjectNotFoundError,
    detect_format,
    load_dataset,
    parse_s3_uri,
    read_model,
    read_model_if_exists,
    uri_matches_prefix,
    write_model,
    write_parquet,
)
from ml_engine.io.dataset import UnsupportedDatasetFormatError, resolve_format


@pytest.mark.parametrize(
    "uri", ["s3://bucket/key.csv", "s3://my-bucket.name/a/b/c.parquet", "s3://b-1/x"]
)
def test_valid_s3_uris_parse(uri):
    parsed = parse_s3_uri(uri)
    assert str(parsed) == uri


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "bucket/key",
        "s3://bucket",
        "s3://Bucket/key",
        "s3://bucket//../etc/passwd",
        "s3://bad_bucket/key",
        "https://bucket.s3.amazonaws.com/key",
    ],
)
def test_invalid_s3_uris_are_rejected(uri):
    with pytest.raises(InvalidS3UriError):
        parse_s3_uri(uri)


def test_prefix_matching_only_matches_path_boundaries():
    assert uri_matches_prefix("s3://b/data/x.csv", "s3://b/data") is True
    assert uri_matches_prefix("s3://b/data", "s3://b/data") is True
    assert uri_matches_prefix("s3://b/database/x.csv", "s3://b/data") is False


@pytest.mark.parametrize(
    ("uri", "expected"),
    [("a/b.csv", "csv"), ("a/b.parquet", "parquet"), ("a/b.CSV", "csv"), ("a/b.txt", "unknown")],
)
def test_format_detection(uri, expected):
    assert detect_format(uri) == expected


def test_unsupported_format_is_rejected():
    with pytest.raises(UnsupportedDatasetFormatError):
        resolve_format("data.txt")


def test_layout_matches_the_documented_structure():
    layout = ExperimentLayout.for_experiment("s3://bucket", "exp-1")
    base = "s3://bucket/ml-factory/experiments/exp-1"
    assert layout.base == base
    assert layout.eda == f"{base}/eda/eda.json"
    assert layout.leakage == f"{base}/eda/leakage.json"
    assert layout.experiment_config == f"{base}/config/experiment_config.json"
    assert layout.selected_features == f"{base}/config/selected_features.json"
    assert layout.preparation == f"{base}/validation/validation.json"
    assert layout.train_dataset == f"{base}/datasets/train.parquet"
    assert layout.comparison == f"{base}/comparison/comparison.json"
    assert layout.summary == f"{base}/report_data/experiment_summary.json"
    assert layout.model_metadata("xgboost") == f"{base}/models/xgboost/metadata.json"
    assert layout.model_failure("xgboost") == f"{base}/models/xgboost/failure.json"
    assert layout.reasoning == f"{base}/reasoning/reasoning.json"


def test_preprocessor_paths_are_strategy_specific():
    layout = ExperimentLayout.for_experiment("s3://bucket", "exp-1")
    assert layout.preprocessor("dense_numeric") != layout.preprocessor("native_categorical")


@pytest.fixture
def local_store() -> LocalObjectStore:
    """The filesystem adapter, used for SageMaker-mounted paths and these tests."""
    return LocalObjectStore()


def test_local_store_round_trips_bytes(tmp_path, local_store):
    uri = str(tmp_path / "nested" / "file.bin")
    local_store.write_bytes(uri, b"payload")
    assert local_store.read_bytes(uri) == b"payload"
    assert local_store.exists(uri)
    assert local_store.stat(uri).size_bytes == 7


def test_missing_object_raises(tmp_path, local_store):
    with pytest.raises(ObjectNotFoundError):
        local_store.read_bytes(str(tmp_path / "absent.json"))
    assert local_store.stat(str(tmp_path / "absent.json")) is None


def test_contracts_round_trip_through_the_store(tmp_path, local_store, classification_eda):
    uri = str(tmp_path / "eda.json")
    write_model(local_store, uri, classification_eda)
    restored = read_model(local_store, uri, EdaReport)
    assert restored.dataset == classification_eda.dataset
    assert len(restored.columns) == len(classification_eda.columns)
    assert read_model_if_exists(local_store, str(tmp_path / "nope.json"), EdaReport) is None


def test_csv_and_parquet_load_identically(tmp_path, local_store, classification_frame):
    csv_path = tmp_path / "data.csv"
    classification_frame.to_csv(csv_path, index=False)
    parquet_uri = str(tmp_path / "data.parquet")
    write_parquet(local_store, parquet_uri, classification_frame)

    from_csv = load_dataset(local_store, str(csv_path))
    from_parquet = load_dataset(local_store, parquet_uri)
    assert from_csv.frame.shape == from_parquet.frame.shape
    assert list(from_csv.frame.columns) == list(from_parquet.frame.columns)
    assert from_csv.file_format == "csv"
    assert from_parquet.file_format == "parquet"


def test_sampling_is_flagged(tmp_path, local_store, classification_frame):
    path = tmp_path / "data.csv"
    classification_frame.to_csv(path, index=False)
    loaded = load_dataset(local_store, str(path), max_rows=50)
    assert len(loaded.frame) == 50
    assert loaded.sampled is True


def test_parquet_prefix_of_parts_is_loaded(tmp_path, store):
    directory = tmp_path / "parts"
    directory.mkdir()
    pd.DataFrame({"a": [1, 2]}).to_parquet(directory / "part-0.parquet", index=False)
    pd.DataFrame({"a": [3, 4]}).to_parquet(directory / "part-1.parquet", index=False)
    loaded = load_dataset(LocalObjectStore(), str(directory) + "/")
    assert len(loaded.frame) == 4
