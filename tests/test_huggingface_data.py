"""Hugging Face dataset source."""

import io
import json
import sys
import types

import pytest
from PIL import Image

from data_loading.huggingface import HuggingFaceDataSource


def _png(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (6, 4), color)


def test_source_caps_rows_and_exposes_image_and_label_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"image": _png((255, 0, 0)), "label": 2},
        {"image": _png((0, 255, 0)), "label": 0},
        {"image": _png((0, 0, 255)), "label": 1},
    ]
    seen: dict[str, object] = {}

    def load_dataset(repo_id: str, **kwargs: object) -> list[dict]:
        seen["repo_id"] = repo_id
        seen["kwargs"] = kwargs
        return rows

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)

    source = HuggingFaceDataSource(
        repo_id="SoccerNet/SoccerNet_raw_HQ",
        revision="frames-720p-2fps",
        split="train",
        image_column="image",
        label_column="label",
        max_samples=2,
    )
    samples = source.list_samples()

    assert seen["repo_id"] == "SoccerNet/SoccerNet_raw_HQ"
    assert seen["kwargs"] == {
        "split": "train",
        "revision": "frames-720p-2fps",
        "streaming": True,
    }
    assert [sample.sample_id for sample in samples] == ["0", "1"]
    with Image.open(io.BytesIO(source.read_bytes(samples[0].image_key))) as image:
        assert image.size == (6, 4)
    assert json.loads(source.read_bytes(samples[1].annotation_key)) == {"label": 0}
    with pytest.raises(FileNotFoundError):
        source.read_bytes("image:2")


def test_missing_label_column_names_the_available_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *_args, **_kwargs: [{"image": _png((0, 0, 0)), "class_name": "goal"}]
    monkeypatch.setitem(sys.modules, "datasets", module)

    source = HuggingFaceDataSource(repo_id="org/data", label_column="label", max_samples=1)
    with pytest.raises(ValueError, match="missing label"):
        source.list_samples()
