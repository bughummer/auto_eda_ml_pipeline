"""Dataset loading and artifact serialization.

Supported dataset formats in v1.0: CSV (optionally gzipped), TSV and Parquet (single file or
a prefix of part files). Everything is streamed through the :class:`ObjectStore` boundary so
the same code path works against S3, a SageMaker-mounted directory, or a local folder.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_engine.io.object_store import ObjectNotFoundError, ObjectStore
from ml_engine.io.uri import detect_format

SUPPORTED_FORMATS = ("csv", "tsv", "parquet")


class UnsupportedDatasetFormatError(ValueError):
    """Raised for a dataset the platform cannot read."""


class DatasetLoadError(RuntimeError):
    """Raised when a dataset exists but cannot be parsed."""


@dataclass(slots=True)
class LoadedDataset:
    frame: pd.DataFrame
    file_format: str
    source_uri: str
    source_size_bytes: int | None
    sampled: bool = False


def resolve_format(uri: str, explicit: str | None = None) -> str:
    """Determine the dataset format, preferring an explicit override."""
    fmt = (explicit or detect_format(uri)).lower()
    if fmt == "unknown":
        raise UnsupportedDatasetFormatError(
            f"Cannot infer dataset format from {uri!r}. Supported: {', '.join(SUPPORTED_FORMATS)}."
        )
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedDatasetFormatError(
            f"Unsupported dataset format {fmt!r}. Supported: {', '.join(SUPPORTED_FORMATS)}."
        )
    return fmt


def load_dataset(
    store: ObjectStore,
    uri: str,
    file_format: str | None = None,
    max_rows: int | None = None,
) -> LoadedDataset:
    """Load a dataset into memory.

    ``max_rows`` caps how many rows are profiled; when the cap is hit the result is flagged
    as sampled so every downstream artifact can say so honestly.
    """
    fmt = resolve_format(uri, file_format)
    metadata = store.stat(uri) if not uri.endswith("/") else None
    with tempfile.TemporaryDirectory(prefix="mlf-dataset-") as tmp:
        tmp_dir = Path(tmp)
        local_paths = _materialize(store, uri, tmp_dir, fmt)
        try:
            frame = _read_frames(local_paths, fmt, max_rows)
        except UnsupportedDatasetFormatError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise DatasetLoadError(f"Failed to parse {fmt} dataset at {uri}: {exc}") from exc

    sampled = max_rows is not None and len(frame) >= max_rows
    return LoadedDataset(
        frame=frame,
        file_format=fmt,
        source_uri=uri,
        source_size_bytes=metadata.size_bytes if metadata else None,
        sampled=sampled,
    )


def _materialize(store: ObjectStore, uri: str, tmp_dir: Path, fmt: str) -> list[Path]:
    """Copy the dataset locally, expanding a prefix of parquet parts when needed."""
    if uri.endswith("/"):
        members = [u for u in store.list_uris(uri) if not u.endswith("/")]
        if fmt == "parquet":
            members = [u for u in members if u.endswith((".parquet", ".pq"))]
        if not members:
            raise ObjectNotFoundError(f"No dataset files found under {uri}")
        paths = []
        for index, member in enumerate(sorted(members)):
            target = tmp_dir / f"part-{index:05d}"
            paths.append(store.download(member, target))
        return paths
    return [store.download(uri, tmp_dir / "dataset")]


def _read_frames(paths: list[Path], fmt: str, max_rows: int | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    remaining = max_rows
    for path in paths:
        frames.append(_read_one(path, fmt, remaining))
        if remaining is not None:
            remaining = max_rows - sum(len(f) for f in frames)
            if remaining <= 0:
                break
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _read_one(path: Path, fmt: str, max_rows: int | None) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path, nrows=max_rows)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t", nrows=max_rows)
    if fmt == "parquet":
        frame = pd.read_parquet(path)
        return frame.head(max_rows) if max_rows is not None else frame
    raise UnsupportedDatasetFormatError(f"Unsupported dataset format {fmt!r}")


def write_parquet(store: ObjectStore, uri: str, frame: pd.DataFrame) -> str:
    """Persist a dataframe as a parquet artifact."""
    with tempfile.TemporaryDirectory(prefix="mlf-parquet-") as tmp:
        local = Path(tmp) / "frame.parquet"
        frame.to_parquet(local, index=False)
        store.upload(local, uri)
    return uri


def read_parquet(store: ObjectStore, uri: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="mlf-parquet-") as tmp:
        local = store.download(uri, Path(tmp) / "frame.parquet")
        return pd.read_parquet(local)


def save_joblib(store: ObjectStore, uri: str, obj: Any) -> str:
    """Persist a fitted estimator or pipeline."""
    with tempfile.TemporaryDirectory(prefix="mlf-joblib-") as tmp:
        local = Path(tmp) / "artifact.joblib"
        joblib.dump(obj, local)
        store.upload(local, uri)
    return uri


def load_joblib(store: ObjectStore, uri: str) -> Any:
    with tempfile.TemporaryDirectory(prefix="mlf-joblib-") as tmp:
        local = store.download(uri, Path(tmp) / "artifact.joblib")
        return joblib.load(local)
