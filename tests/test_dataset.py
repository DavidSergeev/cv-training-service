"""Dataset and S3 listing."""

import io
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from data_loading.annotations import JsonBoxAnnotationReader, JsonLabelAnnotationReader
from data_loading.dataset import SupervisedImageDataset
from data_loading.local import LocalDataSource
from data_loading.s3 import S3DataSource
from processing.preprocessing import Resize


def _png(color: tuple[int, int, int], size: int = 8) -> bytes:
    image = Image.new("RGB", (size, size), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_local_dataset_reads_labels_and_resizes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    (images / "cat.png").write_bytes(_png((255, 0, 0)))
    (annotations / "cat.json").write_text(json.dumps({"label": 2}), encoding="utf-8")

    source = LocalDataSource(str(tmp_path))
    dataset = SupervisedImageDataset(
        source,
        source.list_samples(),
        JsonLabelAnnotationReader(),
        Resize((4, 4)),
    )
    image, label = dataset[0]
    assert image.shape == (3, 4, 4)
    assert int(label) == 2
    assert image.max() <= 1


def test_box_reader_rejects_mismatched_lengths() -> None:
    payload = json.dumps({"boxes": [[0, 0, 1, 1]], "labels": [1, 2]}).encode("utf-8")
    with pytest.raises(ValueError, match="same length"):
        JsonBoxAnnotationReader().read(payload)


class _PagedS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def list_objects_v2(self, Bucket: str, Prefix: str = "", ContinuationToken: str | None = None) -> dict:
        del Bucket
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        if ContinuationToken is None:
            page = keys[:1]
            truncated = len(keys) > 1
            token = "next" if truncated else None
        else:
            page = keys[1:]
            truncated = False
            token = None
        response = {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": truncated,
        }
        if token:
            response["NextContinuationToken"] = token
        return response

    def get_object(self, Bucket: str, Key: str) -> dict:
        del Bucket
        return {"Body": io.BytesIO(self.objects[Key])}


def test_s3_source_pages_and_reads_boxes() -> None:
    objects = {
        "sets/images/notes.txt": b"ignore",
        "sets/images/b.jpg": _png((0, 255, 0)),
        "sets/images/a.jpg": _png((0, 0, 255)),
        "sets/annotations/a.json": json.dumps({"boxes": [[1, 1, 3, 3]], "labels": [0]}).encode("utf-8"),
        "sets/annotations/b.json": json.dumps({"boxes": [], "labels": []}).encode("utf-8"),
    }
    source = S3DataSource(bucket="data", prefix="sets", client=_PagedS3(objects))
    samples = source.list_samples()
    assert [sample.sample_id for sample in samples] == ["a", "b"]

    dataset = SupervisedImageDataset(source, samples, JsonBoxAnnotationReader(), None)
    _image, target = dataset[0]
    assert target["boxes"].shape == (1, 4)
    assert torch.equal(target["labels"], torch.tensor([0]))
