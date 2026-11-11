"""Small classifier used to exercise the generic training path."""

from __future__ import annotations

import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1.")
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(1)
        return self.head(features)
