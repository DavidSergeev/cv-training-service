"""Decode image bytes and expose them through a PyTorch dataset."""

from __future__ import annotations

import io
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_loading.annotations import AnnotationReader
from data_loading.source import DataSource, SampleRef
from processing.preprocessing import SampleTransform


def decode_image(payload: bytes) -> torch.Tensor:
    """Decode encoded image bytes into a float CHW tensor in the range [0, 1]."""
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Decoded image must have shape height x width x 3.")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class SupervisedImageDataset(Dataset):
    """Load ``(image, target)`` pairs from a source, reader, and transform pipeline."""

    def __init__(
        self,
        source: DataSource,
        samples: Sequence[SampleRef],
        reader: AnnotationReader,
        transform: SampleTransform | None = None,
    ) -> None:
        self.source = source
        self.samples = list(samples)
        self.reader = reader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        sample = self.samples[index]
        image = decode_image(self.source.read_bytes(sample.image_key))
        try:
            annotation = self.source.read_bytes(sample.annotation_key)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Annotation not found for sample {sample.sample_id}: {sample.annotation_key}"
            ) from exc
        target = self.reader.read(annotation)
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
