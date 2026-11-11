"""Construct optimizers named by configuration."""

from __future__ import annotations

from typing import Any

import torch

from cv_training.config import OptimizerConfig

OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    key = config.name.lower()
    if key not in OPTIMIZERS:
        choices = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(f"Unknown optimizer {config.name!r}. Expected one of: {choices}.")
    return OPTIMIZERS[key](model.parameters(), lr=config.lr, **config.params)
