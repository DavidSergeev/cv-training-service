"""S3 data source using the same images/ and annotations/ layout as local data."""

from __future__ import annotations

from typing import Any

from data_loading.source import DataSource, SampleRef, has_image_suffix
from s3 import create_s3_client, is_not_found, join_key


class S3DataSource(DataSource):
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        images_dir: str = "images",
        annotations_dir: str = "annotations",
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.images_dir = images_dir.strip("/")
        self.annotations_dir = annotations_dir.strip("/")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = create_s3_client()
        return self._client

    def list_samples(self) -> list[SampleRef]:
        image_prefix = join_key(self.prefix, self.images_dir)
        if image_prefix:
            image_prefix += "/"
        samples: list[SampleRef] = []
        for key in self._list_keys(image_prefix):
            if key.endswith("/") or not has_image_suffix(key):
                continue
            relative = key[len(image_prefix) :]
            stem, dot, _suffix = relative.rpartition(".")
            if not dot:
                continue
            annotation_key = join_key(self.prefix, self.annotations_dir, f"{stem}.json")
            sample_id = stem
            samples.append(SampleRef(sample_id=sample_id, image_key=key, annotation_key=annotation_key))
        samples.sort(key=lambda sample: sample.sample_id)
        return samples

    def read_bytes(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if is_not_found(exc):
                raise FileNotFoundError(key) from exc
            raise
        body = response["Body"]
        return body.read()

    def _list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self.client.list_objects_v2(**kwargs)
            keys.extend(item["Key"] for item in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return keys
            token = response.get("NextContinuationToken")
            if not token:
                return keys
