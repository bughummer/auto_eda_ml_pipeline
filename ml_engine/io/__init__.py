"""IO boundary: object storage, dataset loading, URI handling and the artifact layout.

This is the only package inside ``ml_engine`` permitted to know about storage backends.
"""

from ml_engine.io.dataset import (
    DatasetLoadError,
    LoadedDataset,
    UnsupportedDatasetFormatError,
    load_dataset,
    load_joblib,
    read_parquet,
    resolve_format,
    save_joblib,
    write_parquet,
)
from ml_engine.io.layout import ExperimentLayout
from ml_engine.io.object_store import (
    LocalObjectStore,
    ObjectMetadata,
    ObjectNotFoundError,
    ObjectStore,
    read_model,
    read_model_if_exists,
    write_json,
    write_model,
)
from ml_engine.io.s3 import S3ObjectStore
from ml_engine.io.uri import (
    InvalidS3UriError,
    S3Uri,
    detect_format,
    is_s3_uri,
    join_uri,
    parse_s3_uri,
    uri_matches_prefix,
)

__all__ = [
    "DatasetLoadError",
    "ExperimentLayout",
    "InvalidS3UriError",
    "LoadedDataset",
    "LocalObjectStore",
    "ObjectMetadata",
    "ObjectNotFoundError",
    "ObjectStore",
    "S3ObjectStore",
    "S3Uri",
    "UnsupportedDatasetFormatError",
    "detect_format",
    "is_s3_uri",
    "join_uri",
    "load_dataset",
    "load_joblib",
    "parse_s3_uri",
    "read_model",
    "read_model_if_exists",
    "read_parquet",
    "resolve_format",
    "save_joblib",
    "uri_matches_prefix",
    "write_json",
    "write_model",
    "write_parquet",
]
