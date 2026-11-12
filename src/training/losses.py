"""Generic loss constructors. Task-specific losses stay with their trainer."""

from __future__ import annotations

import torch.nn as nn

from config import LossConfig
from imports import load_class

GENERIC_LOSSES: dict[str, type[nn.Module]] = {
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
}


def build_loss(config: LossConfig | None) -> nn.Module | None:
    """Build a generic loss, a dotted loss class, or return ``None``.

    ``None`` means the name is reserved for the concrete trainer. Built-in
    names and dotted paths are constructed here so the training loop can
    receive the module by injection.
    """
    if config is None:
        return None
    if "." in config.name:
        return load_class(config.name)(**config.params)
    if config.name in GENERIC_LOSSES:
        return GENERIC_LOSSES[config.name](**config.params)
    return None
