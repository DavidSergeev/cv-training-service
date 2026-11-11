"""Single-grid reference detector.

The head predicts one class distribution and one box per cell. It exists so
``ObjectDetectionTrainer`` can run without a production detection stack.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TinyDetector(nn.Module):
    def __init__(self, num_classes: int, grid_size: int = 8) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1.")
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1.")
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classification = nn.Conv2d(64, num_classes + 1, kernel_size=1)
        self.boxes = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(inputs)
        features = F.interpolate(
            features,
            size=(self.grid_size, self.grid_size),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.classification(features).permute(0, 2, 3, 1)
        boxes = self.boxes(features).permute(0, 2, 3, 1).sigmoid()
        return {"logits": logits, "boxes": boxes}
