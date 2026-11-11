"""Store artifacts in an S3 bucket."""

from __future__ import annotations

from typing import Any

from cv_training.persistence.store import ArtifactStore, validate_relative_key
from cv_training.s3 import create_s3_client, is_not_found, join_key


class S3ArtifactStore(ArtifactStore):
    def __init__(self, bucket: str, prefix: str = "", client: Any | None = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = create_s3_client()
        return self._client

    def save(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._object_key(key), Body=data)

    def load(self, key: str) -> bytes:
        object_key = self._object_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:
            if is_not_found(exc):
                raise FileNotFoundError(key) from exc
            raise
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._object_key(key))
        except Exception as exc:
            if is_not_found(exc):
                return False
            raise
        return True

    def _object_key(self, key: str) -> str:
        return join_key(self.prefix, validate_relative_key(key))
