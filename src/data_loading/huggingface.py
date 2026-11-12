"""Stream image and label rows from a Hugging Face dataset."""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image

from data_loading.source import DataSource, SampleRef


class HuggingFaceDataSource(DataSource):
    """Expose Hub rows through the image-bytes and JSON-label contract.

    Rows are streamed and capped by ``max_samples``. Each kept row is stored as
    encoded image bytes plus ``{"label": <int>}`` so the existing dataset and
    annotation reader can consume them.
    """

    def __init__(
        self,
        repo_id: str,
        revision: str = "main",
        split: str = "train",
        image_column: str = "image",
        label_column: str = "label",
        max_samples: int | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.split = split
        self.image_column = image_column
        self.label_column = label_column
        self.max_samples = max_samples
        self._samples: list[SampleRef] | None = None
        self._payloads: dict[str, bytes] = {}

    def list_samples(self) -> list[SampleRef]:
        self._ensure_loaded()
        assert self._samples is not None
        return list(self._samples)

    def read_bytes(self, key: str) -> bytes:
        self._ensure_loaded()
        try:
            return self._payloads[key]
        except KeyError as exc:
            raise FileNotFoundError(key) from exc

    def _ensure_loaded(self) -> None:
        if self._samples is not None:
            return
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Hugging Face datasets require the datasets package.") from exc
        rows = load_dataset(
            self.repo_id,
            split=self.split,
            revision=self.revision,
            streaming=True,
        )
        samples: list[SampleRef] = []
        payloads: dict[str, bytes] = {}
        for index, row in enumerate(rows):
            if self.max_samples is not None and index >= self.max_samples:
                break
            if not isinstance(row, dict):
                raise TypeError("Hugging Face dataset rows must be mappings.")
            self._require_columns(row)
            image_key = f"image:{index}"
            annotation_key = f"label:{index}"
            payloads[image_key] = _image_bytes(row[self.image_column])
            label = _label_int(row[self.label_column])
            payloads[annotation_key] = json.dumps({"label": label}).encode("utf-8")
            samples.append(SampleRef(sample_id=str(index), image_key=image_key, annotation_key=annotation_key))
        self._samples = samples
        self._payloads = payloads

    def _require_columns(self, row: dict[str, Any]) -> None:
        missing = [name for name in (self.image_column, self.label_column) if name not in row]
        if not missing:
            return
        present = ", ".join(sorted(str(name) for name in row))
        names = ", ".join(missing)
        raise ValueError(f"Hugging Face row is missing {names}. Columns: {present}.")


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return bytes(value["bytes"])
    if isinstance(value, Image.Image):
        image = value
    else:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                image = Image.fromarray(value)
            else:
                raise TypeError
        except TypeError as exc:
            raise TypeError("Hugging Face image column must be a PIL image, encoded bytes, or an array.") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _label_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("Hugging Face label column must be an integer.")
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, int):
        return value
    raise TypeError(f"Hugging Face label column must be an integer, got {type(value).__name__}.")
