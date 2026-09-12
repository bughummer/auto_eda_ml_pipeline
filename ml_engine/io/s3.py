"""S3 implementation of :class:`ml_engine.io.object_store.ObjectStore`.

This is the one module inside ``ml_engine`` that is allowed to touch AWS, and it keeps the
``boto3`` import lazy so importing ``ml_engine`` never requires the SDK. Both the FastAPI
control plane and the SageMaker jobs share this implementation rather than each growing
their own (see ``architecture.md`` AD-8).
"""

from pathlib import Path
from typing import Any

from ml_engine.io.object_store import ObjectMetadata, ObjectNotFoundError
from ml_engine.io.uri import parse_s3_uri


class S3ObjectStore:
    """Thin, dependency-injected S3 adapter.

    ``client`` is a boto3 S3 client. It is injected so tests and the control plane can supply
    a proxy-aware, KMS-aware, role-scoped client built in one place.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import boto3  # imported lazily: ml_engine stays importable without the AWS SDK

            client = boto3.client("s3")
        self._client = client

    @property
    def client(self) -> Any:
        return self._client

    def read_bytes(self, uri: str) -> bytes:
        parsed = parse_s3_uri(uri)
        try:
            response = self._client.get_object(Bucket=parsed.bucket, Key=parsed.key)
        except Exception as exc:
            if _is_missing(exc):
                raise ObjectNotFoundError(f"No object at {uri}") from exc
            raise
        return response["Body"].read()

    def write_bytes(self, uri: str, data: bytes) -> None:
        parsed = parse_s3_uri(uri)
        self._client.put_object(Bucket=parsed.bucket, Key=parsed.key, Body=data)

    def exists(self, uri: str) -> bool:
        return self.stat(uri) is not None

    def list_uris(self, prefix: str) -> list[str]:
        parsed = parse_s3_uri(prefix if prefix.endswith("/") else prefix + "/")
        paginator = self._client.get_paginator("list_objects_v2")
        uris: list[str] = []
        for page in paginator.paginate(Bucket=parsed.bucket, Prefix=parsed.key):
            for item in page.get("Contents", []):
                uris.append(f"s3://{parsed.bucket}/{item['Key']}")
        return sorted(uris)

    def list_prefixes(self, prefix: str) -> list[str]:
        """Common prefixes one level down, so listing experiments costs one call per page
        instead of one per object."""
        parsed = parse_s3_uri(prefix if prefix.endswith("/") else prefix + "/")
        paginator = self._client.get_paginator("list_objects_v2")
        prefixes: list[str] = []
        for page in paginator.paginate(Bucket=parsed.bucket, Prefix=parsed.key, Delimiter="/"):
            for item in page.get("CommonPrefixes", []):
                prefixes.append(f"s3://{parsed.bucket}/{item['Prefix']}")
        return sorted(prefixes)

    def stat(self, uri: str) -> ObjectMetadata | None:
        parsed = parse_s3_uri(uri)
        try:
            head = self._client.head_object(Bucket=parsed.bucket, Key=parsed.key)
        except Exception as exc:
            if _is_missing(exc):
                return None
            raise
        return ObjectMetadata(
            uri=uri,
            size_bytes=int(head.get("ContentLength", 0)),
            last_modified=head.get("LastModified"),
            etag=(head.get("ETag") or "").strip('"') or None,
            version_id=head.get("VersionId"),
        )

    def download(self, uri: str, destination: Path) -> Path:
        parsed = parse_s3_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(parsed.bucket, parsed.key, str(destination))
        except Exception as exc:
            if _is_missing(exc):
                raise ObjectNotFoundError(f"No object at {uri}") from exc
            raise
        return destination

    def upload(self, source: Path, uri: str) -> None:
        parsed = parse_s3_uri(uri)
        self._client.upload_file(str(source), parsed.bucket, parsed.key)


_MISSING_CODES = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}


def _is_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return exc.__class__.__name__ in {"NoSuchKey", "ClientError404"}
    error = response.get("Error", {})
    return str(error.get("Code")) in _MISSING_CODES
