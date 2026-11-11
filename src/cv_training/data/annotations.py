"""Annotation parsers. Each reader understands one annotation structure."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import torch


class AnnotationReader(ABC):
    @abstractmethod
    def read(self, payload: bytes) -> Any:
        """Parse annotation bytes into the target returned by the dataset."""


def load_json(payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Annotation payload is not valid JSON.") from exc


class JsonLabelAnnotationReader(AnnotationReader):
    """Read ``{"label": <int>}`` classification annotations."""

    def read(self, payload: bytes) -> torch.Tensor:
        data = load_json(payload)
        if not isinstance(data, dict) or "label" not in data:
            raise ValueError("Classification annotation must contain 'label'.")
        return torch.tensor(int(data["label"]), dtype=torch.long)


class JsonBoxAnnotationReader(AnnotationReader):
    """Read detection annotations shaped as ``{"boxes": [[x1, y1, x2, y2], ...], "labels": [int, ...]}``."""

    def read(self, payload: bytes) -> dict[str, torch.Tensor]:
        data = load_json(payload)
        if not isinstance(data, dict):
            raise ValueError("Detection annotation must be a JSON object.")
        boxes = data.get("boxes", [])
        labels = data.get("labels", [])
        try:
            box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Each box must contain 4 values: x1, y1, x2, y2.") from exc
        try:
            label_tensor = torch.tensor(labels, dtype=torch.long).reshape(-1)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Detection labels must be a list of integers.") from exc
        if box_tensor.shape[0] != label_tensor.shape[0]:
            raise ValueError("Detection boxes and labels must have the same length.")
        return {"boxes": box_tensor, "labels": label_tensor}
