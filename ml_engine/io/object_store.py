"""Object store boundary.

``ObjectStore`` is the only way ML Factory code touches bytes it did not create. Two
implementations exist: ``LocalObjectStore`` (a directory tree that mirrors the S3 layout,
used for local development, tests and SageMaker-mounted paths) and ``S3ObjectStore``
(``ml_engine.io.s3``). Everything above this boundary is storage-agnostic.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ObjectNotFoundError(FileNotFoundError):
    """Raised when a requested object does not exist."""


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    uri: str
    size_bytes: int
    last_modified: datetime | None = None
    etag: str | None = None
    version_id: str | None = None


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal, testable storage contract."""

    def read_bytes(self, uri: str) -> bytes: ...

    def write_bytes(self, uri: str, data: bytes) -> None: ...

    def exists(self, uri: str) -> bool: ...

    def list_uris(self, prefix: str) -> list[str]: ...

    def stat(self, uri: str) -> ObjectMetadata | None: ...

    def download(self, uri: str, destination: Path) -> Path: ...

    def upload(self, source: Path, uri: str) -> None: ...


def _to_path(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri[len("file://") :])
    return Path(uri)


class LocalObjectStore:
    """Filesystem-backed store. Accepts plain paths and ``file://`` URIs."""

    def read_bytes(self, uri: str) -> bytes:
        path = _to_path(uri)
        if not path.is_file():
            raise ObjectNotFoundError(f"No object at {uri}")
        return path.read_bytes()

    def write_bytes(self, uri: str, data: bytes) -> None:
        path = _to_path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, uri: str) -> bool:
        return _to_path(uri).exists()

    def list_uris(self, prefix: str) -> list[str]:
        root = _to_path(prefix)
        if root.is_file():
            return [str(root)]
        if not root.exists():
            return []
        return sorted(str(p) for p in root.rglob("*") if p.is_file())

    def stat(self, uri: str) -> ObjectMetadata | None:
        path = _to_path(uri)
        if not path.is_file():
            return None
        info = path.stat()
        return ObjectMetadata(
            uri=uri,
            size_bytes=info.st_size,
            last_modified=datetime.fromtimestamp(info.st_mtime, tz=UTC),
        )

    def download(self, uri: str, destination: Path) -> Path:
        source = _to_path(uri)
        if not source.is_file():
            raise ObjectNotFoundError(f"No object at {uri}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return destination

    def upload(self, source: Path, uri: str) -> None:
        destination = _to_path(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)


def write_model(store: ObjectStore, uri: str, model: BaseModel) -> str:
    """Serialize a contract to JSON and store it. Returns the URI for convenience."""
    payload = model.model_dump_json(indent=2, by_alias=True)
    store.write_bytes(uri, payload.encode("utf-8"))
    return uri


def read_model[ModelT: BaseModel](store: ObjectStore, uri: str, model_type: type[ModelT]) -> ModelT:
    """Read and validate a contract artifact. Raises ``ObjectNotFoundError`` if absent."""
    raw = store.read_bytes(uri)
    return model_type.model_validate_json(raw)


def read_model_if_exists[ModelT: BaseModel](
    store: ObjectStore, uri: str, model_type: type[ModelT]
) -> ModelT | None:
    try:
        return read_model(store, uri, model_type)
    except (ObjectNotFoundError, FileNotFoundError):
        return None


def write_json(store: ObjectStore, uri: str, payload: dict) -> str:
    store.write_bytes(uri, json.dumps(payload, indent=2, default=str).encode("utf-8"))
    return uri
