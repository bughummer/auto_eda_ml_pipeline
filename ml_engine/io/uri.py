"""S3 URI parsing and validation.

The only place in the platform that knows how an ``s3://`` string is shaped.
"""

import re
from dataclasses import dataclass

S3_URI_PATTERN = re.compile(r"^s3://(?P<bucket>[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])/(?P<key>.+)$")
BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$")


class InvalidS3UriError(ValueError):
    """Raised when a string is not a usable S3 URI."""


@dataclass(frozen=True, slots=True)
class S3Uri:
    bucket: str
    key: str

    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def prefix(self) -> str:
        """Everything up to the last path segment."""
        return self.key.rsplit("/", 1)[0] if "/" in self.key else ""

    def join(self, *parts: str) -> "S3Uri":
        key = "/".join([self.key.rstrip("/"), *[p.strip("/") for p in parts if p]])
        return S3Uri(self.bucket, key)


def is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def parse_s3_uri(value: str) -> S3Uri:
    """Parse and validate an S3 URI, raising ``InvalidS3UriError`` with a usable message."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidS3UriError("S3 URI must be a non-empty string")
    candidate = value.strip()
    if not candidate.startswith("s3://"):
        raise InvalidS3UriError(f"S3 URI must start with 's3://': {candidate!r}")
    match = S3_URI_PATTERN.match(candidate)
    if not match:
        raise InvalidS3UriError(
            f"Malformed S3 URI {candidate!r}. Expected s3://<bucket>/<key> with a "
            "DNS-compliant lowercase bucket name."
        )
    bucket, key = match.group("bucket"), match.group("key")
    if ".." in key or key.startswith("/"):
        raise InvalidS3UriError(f"S3 key must not be relative or contain '..': {key!r}")
    if "_" in bucket or bucket.endswith("-"):
        raise InvalidS3UriError(f"Invalid S3 bucket name: {bucket!r}")
    return S3Uri(bucket=bucket, key=key)


def join_uri(base: str, *parts: str) -> str:
    """Join URI or path components with a single separator, independent of scheme."""
    cleaned = [p.strip("/") for p in parts if p]
    if not cleaned:
        return base
    return "/".join([base.rstrip("/"), *cleaned])


def uri_matches_prefix(uri: str, allowed_prefix: str) -> bool:
    """Allow-list check. A prefix only matches at a path boundary, never mid-segment."""
    normalized_prefix = allowed_prefix.rstrip("/")
    if uri == normalized_prefix:
        return True
    return uri.startswith(normalized_prefix + "/")


def detect_format(uri: str) -> str:
    """Infer the dataset file format from the URI suffix.

    A trailing slash means a prefix of part files, which the platform reads as parquet.
    """
    if uri.endswith("/"):
        return "parquet"
    lowered = uri.lower()
    for suffix, fmt in (
        (".csv.gz", "csv"),
        (".csv", "csv"),
        (".tsv", "tsv"),
        (".parquet", "parquet"),
        (".pq", "parquet"),
        (".json", "json"),
        (".xlsx", "xlsx"),
        (".xls", "xlsx"),
    ):
        if lowered.endswith(suffix):
            return fmt
    return "unknown"
