"""CPU image transforms that also keep detection boxes aligned.

Transforms take a float CHW tensor and a target. A future GPU implementation
can honor the same call signature on CUDA tensors; the dataset applies the
pipeline before the training step.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F

from cv_training.config import PreparationConfig


class SampleTransform(Protocol):
    def __call__(self, image: torch.Tensor, target: Any) -> tuple[torch.Tensor, Any]:
        """Return the transformed image and its aligned target."""


class Compose:
    def __init__(self, transforms: list[SampleTransform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: torch.Tensor, target: Any) -> tuple[torch.Tensor, Any]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class Resize:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = (int(size[0]), int(size[1]))

    def __call__(self, image: torch.Tensor, target: Any) -> tuple[torch.Tensor, Any]:
        _channels, height, width = image.shape
        resized = F.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        boxes = _boxes_of(target)
        if boxes is None or boxes.numel() == 0:
            return resized, target
        new_height, new_width = self.size
        scale = boxes.new_tensor([new_width / width, new_height / height, new_width / width, new_height / height])
        updated = dict(target)
        updated["boxes"] = boxes * scale
        return resized, updated


class RandomHorizontalFlip:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: torch.Tensor, target: Any) -> tuple[torch.Tensor, Any]:
        if torch.rand(()) >= self.probability:
            return image, target
        flipped = image.flip(-1)
        boxes = _boxes_of(target)
        if boxes is None or boxes.numel() == 0:
            return flipped, target
        width = flipped.shape[-1]
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        updated = dict(target)
        updated["boxes"] = torch.stack((width - x2, y1, width - x1, y2), dim=-1)
        return flipped, updated


class Normalize:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        if len(mean) != len(std):
            raise ValueError("Normalize mean and std must have the same length.")
        self.mean = [float(value) for value in mean]
        self.std = [float(value) for value in std]

    def __call__(self, image: torch.Tensor, target: Any) -> tuple[torch.Tensor, Any]:
        mean = image.new_tensor(self.mean).view(-1, 1, 1)
        std = image.new_tensor(self.std).view(-1, 1, 1)
        return (image - mean) / std, target


def build_transform_pipeline(config: PreparationConfig, *, augment: bool) -> Compose:
    """Build the preprocessing pipeline. Augmentation is applied to the train split."""
    if config.transform_device != "cpu":
        raise NotImplementedError(
            "Only CPU transformations are implemented. Set preparation.transform_device to 'cpu'."
        )
    steps: list[SampleTransform] = [Resize(config.image_size)]
    if augment and config.augmentation.horizontal_flip:
        steps.append(RandomHorizontalFlip(config.augmentation.flip_probability))
    if config.normalize is not None:
        steps.append(Normalize(config.normalize.mean, config.normalize.std))
    return Compose(steps)


def _boxes_of(target: Any) -> torch.Tensor | None:
    if isinstance(target, dict) and torch.is_tensor(target.get("boxes")):
        return target["boxes"]
    return None
