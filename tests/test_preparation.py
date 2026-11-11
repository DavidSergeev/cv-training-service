"""Preprocessing and splitting."""

from pathlib import Path

import pytest
import torch

from cv_training.config import TrainConfig, load_config
from cv_training.data.source import SampleRef
from cv_training.preparation.split import split_samples
from cv_training.preparation.transforms import (
    Normalize,
    RandomHorizontalFlip,
    Resize,
    build_transform_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample(sample_id: str) -> SampleRef:
    return SampleRef(sample_id=sample_id, image_key=sample_id, annotation_key=sample_id)


def test_split_is_deterministic_and_keeps_a_training_sample() -> None:
    samples = [_sample(str(index)) for index in range(10)]
    first_train, first_val = split_samples(samples, val_ratio=0.2, seed=3)
    second_train, second_val = split_samples(samples, val_ratio=0.2, seed=3)

    assert [sample.sample_id for sample in first_train] == [sample.sample_id for sample in second_train]
    assert [sample.sample_id for sample in first_val] == [sample.sample_id for sample in second_val]
    assert len(first_val) == 2
    assert len(first_train) == 8

    only = split_samples([_sample("a")], val_ratio=0.9, seed=1)
    assert len(only[0]) == 1
    assert only[1] == []


def test_resize_scales_boxes() -> None:
    image = torch.zeros(3, 4, 6)
    target = {"boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]]), "labels": torch.tensor([0])}
    resized, updated = Resize((8, 8))(image, target)

    assert resized.shape == (3, 8, 8)
    expected = torch.tensor([[1.0 * 8 / 6, 2.0, 3.0 * 8 / 6, 6.0]])
    assert torch.allclose(updated["boxes"], expected)


def test_horizontal_flip_mirrors_boxes() -> None:
    image = torch.zeros(3, 4, 6)
    target = {"boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]]), "labels": torch.tensor([1])}
    flipped, updated = RandomHorizontalFlip(1.0)(image, target)
    unchanged, same_target = RandomHorizontalFlip(0.0)(image, target)

    assert flipped.shape == image.shape
    assert torch.allclose(updated["boxes"], torch.tensor([[3.0, 1.0, 5.0, 3.0]]))
    assert torch.equal(unchanged, image)
    assert torch.equal(same_target["boxes"], target["boxes"])


def test_normalize_leaves_labels_unchanged() -> None:
    image = torch.ones(3, 2, 2)
    label = torch.tensor(2)
    normalized, target = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image, label)
    assert torch.allclose(normalized, torch.ones_like(image))
    assert torch.equal(target, label)


def test_gpu_transform_pipeline_is_rejected() -> None:
    raw = load_config(ROOT / "configs" / "classification.yaml").to_dict()
    raw["preparation"]["transform_device"] = "cuda"
    config = TrainConfig.from_dict(raw)
    with pytest.raises(NotImplementedError, match="CPU"):
        build_transform_pipeline(config.preparation, augment=True)
