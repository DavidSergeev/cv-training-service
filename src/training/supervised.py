"""Generic trainer for models whose loss is a single injected module."""

from __future__ import annotations

from typing import Any

import torch

from training.base import BaseTrainer
from training.losses import GENERIC_LOSSES


class SupervisedTrainer(BaseTrainer):
    """Train a model with ``loss_fn(outputs, targets)``.

    Use this for tasks such as classification, where the batch collates to
    tensors and the loss is one of the configured generic losses.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.loss_fn is None:
            configured = None if self.config.loss is None else self.config.loss.name
            choices = ", ".join(sorted(GENERIC_LOSSES))
            raise ValueError(
                "SupervisedTrainer requires a generic loss or a dotted loss class path. "
                f"Configured loss: {configured!r}. Known generic losses: {choices}."
            )

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError("SupervisedTrainer has no loss function.")
        return self.loss_fn(outputs, targets)
