"""Generic loss constructors. Task-specific losses stay with their trainer."""

from __future__ import annotations

import torch.nn as nn

from config import LossSpec

GENERIC_LOSSES: dict[str, type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def build_loss(config: LossSpec) -> nn.Module | None:
    """Build a generic loss, or return ``None`` when the trainer owns the loss."""
    if config.name in GENERIC_LOSSES:
        return GENERIC_LOSSES[config.name]()
    return None
