"""Local and S3 artifact stores."""

import io
from pathlib import Path

import pytest

from cv_training.persistence.local import LocalArtifactStore
from cv_training.persistence.s3 import S3ArtifactStore


class _Missing(Exception):
    def __init__(self) -> None:
        super().__init__("missing")
        self.response = {
            "Error": {"Code": "404"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:
        key = (Bucket, Key)
        if key not in self.objects:
            raise _Missing()
        return {"Body": io.BytesIO(self.objects[key])}

    def head_object(self, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise _Missing()
        return {}


def test_local_store_roundtrip_and_rejects_escape(tmp_path: Path) -> None:
    store = LocalArtifactStore(str(tmp_path))
    store.save("runs/metrics.jsonl", b"ok")
    assert store.exists("runs/metrics.jsonl")
    assert store.load("runs/metrics.jsonl") == b"ok"
    with pytest.raises(ValueError):
        store.save("../secret.txt", b"no")


def test_s3_store_uses_prefix_and_reports_missing_keys() -> None:
    client = _FakeS3()
    store = S3ArtifactStore(bucket="models", prefix="runs/job", client=client)
    assert store.exists("weights.pt") is False
    store.save("weights.pt", b"weights")
    assert client.objects[("models", "runs/job/weights.pt")] == b"weights"
    assert store.load("weights.pt") == b"weights"
    with pytest.raises(FileNotFoundError):
        store.load("missing.pt")
