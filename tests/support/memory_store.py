"""An in-memory object store that speaks real ``s3://`` URIs.

Using this instead of a filesystem path means the tests exercise the production code paths:
S3 URI parsing, the dataset allow-list, prefix listing with a delimiter, and the artifact
layout as it is actually addressed in AWS. Only the bytes live somewhere else.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

from ml_engine.io.object_store import ObjectMetadata, ObjectNotFoundError
from ml_engine.io.uri import parse_s3_uri


class InMemoryObjectStore:
    """Implements :class:`ml_engine.io.object_store.ObjectStore` over a dict."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._written_at: dict[str, datetime] = {}

    # --- helpers used by tests -------------------------------------------
    def put_file(self, uri: str, path: Path) -> str:
        """Upload a file produced by a fixture, e.g. a generated dataset."""
        self.write_bytes(uri, path.read_bytes())
        return uri

    @property
    def keys(self) -> list[str]:
        return sorted(self._objects)

    # --- ObjectStore ------------------------------------------------------
    def read_bytes(self, uri: str) -> bytes:
        parse_s3_uri(uri)
        try:
            return self._objects[uri]
        except KeyError as exc:
            raise ObjectNotFoundError(f"No object at {uri}") from exc

    def write_bytes(self, uri: str, data: bytes) -> None:
        parse_s3_uri(uri)
        self._objects[uri] = data
        self._written_at[uri] = datetime.now(UTC)

    def exists(self, uri: str) -> bool:
        return uri in self._objects

    def list_uris(self, prefix: str) -> list[str]:
        normalized = prefix if prefix.endswith("/") else prefix + "/"
        return sorted(uri for uri in self._objects if uri.startswith(normalized))

    def list_prefixes(self, prefix: str) -> list[str]:
        normalized = prefix if prefix.endswith("/") else prefix + "/"
        found = set()
        for uri in self._objects:
            if not uri.startswith(normalized):
                continue
            remainder = uri[len(normalized) :]
            if "/" in remainder:
                found.add(f"{normalized}{remainder.split('/', 1)[0]}/")
        return sorted(found)

    def stat(self, uri: str) -> ObjectMetadata | None:
        if uri not in self._objects:
            return None
        return ObjectMetadata(
            uri=uri,
            size_bytes=len(self._objects[uri]),
            last_modified=self._written_at.get(uri),
            etag=f"{hash(self._objects[uri]) & 0xFFFFFFFF:08x}",
        )

    def download(self, uri: str, destination: Path) -> Path:
        data = self.read_bytes(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    def upload(self, source: Path, uri: str) -> None:
        self.write_bytes(uri, Path(source).read_bytes())

    def copy_tree_from(self, directory: Path, prefix: str) -> None:  # pragma: no cover - helper
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                self.write_bytes(
                    f"{prefix.rstrip('/')}/{path.relative_to(directory)}", path.read_bytes()
                )
        shutil.rmtree(directory, ignore_errors=True)
